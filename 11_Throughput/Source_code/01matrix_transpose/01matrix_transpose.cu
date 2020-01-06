#include <assert.h>
#include <stdio.h>
#include "common/CpuTimer.h"
#include "common/Error.h"
#include "common/GpuTimer.h"
#include "common/Matrix.h"
#include "common/Utilities.h"

#define N 1024
#define K 16

void compareResults(Matrix<int> h_a, Matrix<int> h_b) {
    int i, j;

    for (i = 0; i < h_a.width; i++) {
        for (j = 0; j < h_a.height; j++) {
            assert(h_a.elements[j * h_a.width + i] ==
                   h_b.elements[i * h_b.width + j]);
        }
    }
}

// Kernel v3 using K threads and N/K blocks
// Try this example with 8, 16 and 32 threads by block
__global__ void transposedMatrixKernel_threads_blocks(Matrix<int> d_a,
                                                      Matrix<int> d_b) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    d_b.setElement(i, j, d_a.getElement(j, i));
}

// Kernel v2 using the max number of threads in 1 block
__global__ void transposedMatrixKernel_threads(Matrix<int> d_a,
                                               Matrix<int> d_b,
                                               int THREADS) {
    int i = threadIdx.x;
    int j = 0;

    while (i < N) {
        while (j < N) {
            d_b.setElement(i, j, d_a.getElement(j, i));
            j++;
        }
        i += THREADS;
    }
}

// Kernel v1 using 1 thread and 1 block
__global__ void transposedMatrixKernel(Matrix<int> d_a, Matrix<int> d_b) {
    int i = 0;
    int j = 0;

    while (i < d_a.width) {
        j = 0;
        while (j < d_a.height) {
            d_b.setElement(i, j, d_a.getElement(j, i));
            j++;
        }
        i++;
    }
}

// Host function
void transposedMatrixHost(Matrix<int> d_a, Matrix<int> d_b) {
    // start timer
    CpuTimer timer;
    timer.Start();

    int i, j;

    for (i = 0; i < d_a.width; i++) {
        for (j = 0; j < d_a.height; j++) {
            d_b.setElement(i, j, d_a.getElement(j, i));
        }
    }
    // stop timer
    timer.Stop();

    // print time
    printf("Time Host:  %f ms\n", timer.Elapsed());
}

void onDevice(Matrix<int> h_a, Matrix<int> h_b) {
    // declare GPU data
    Matrix<int> d_a, d_b;
    d_a.width = h_a.width;
    d_a.height = h_a.height;

    d_b.width = h_b.width;
    d_b.height = h_b.height;

    const int ARRAY_BYTES = d_a.width * d_a.height * sizeof(int);

    // allocate  memory on the GPU
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_a.elements, ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_b.elements, ARRAY_BYTES));

    // copy data from CPU the GPU
    HANDLER_ERROR_ERR(cudaMemcpy(d_a.elements, h_a.elements, ARRAY_BYTES,
                                 cudaMemcpyHostToDevice));
    HANDLER_ERROR_ERR(cudaMemcpy(d_b.elements, h_b.elements, ARRAY_BYTES,
                                 cudaMemcpyHostToDevice));

    GpuTimer timer;

    // -*- [1] -*-
    timer.Start();
    transposedMatrixKernel<<<1, 1>>>(d_a, d_b);
    HANDLER_ERROR_MSG("kernel panic!!!");
    timer.Stop();
    printf("Time Device serial:  %f ms\n", timer.Elapsed());
    bandwidth(N, timer.Elapsed());
    // copy data back from the GPU to the CPU
    HANDLER_ERROR_ERR(cudaMemcpy(h_b.elements, d_b.elements, ARRAY_BYTES,
                                 cudaMemcpyDeviceToHost));
    compareResults(h_a, h_b);

    // -*- [2] -*-
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int THREADS = prop.maxThreadsPerBlock;
    timer.Start();
    transposedMatrixKernel_threads<<<1, THREADS>>>(d_a, d_b, THREADS);
    timer.Stop();
    printf("Time Device threads:  %f ms\n", timer.Elapsed());
    bandwidth(N, timer.Elapsed());
    // copy data back from the GPU to the CPU
    HANDLER_ERROR_ERR(cudaMemcpy(h_b.elements, d_b.elements, ARRAY_BYTES,
                                 cudaMemcpyDeviceToHost));
    compareResults(h_a, h_b);

    // -*- [3] -*-
    timer.Start();
    dim3 GridBlocks(N / K, N / K);
    dim3 ThreadsBlocks(K, K);
    transposedMatrixKernel_threads_blocks<<<GridBlocks, ThreadsBlocks>>>(d_a,
                                                                         d_b);
    HANDLER_ERROR_MSG("kernel panic!!!");
    timer.Stop();
    printf("Time Device threads and blocks:  %f ms\n", timer.Elapsed());
    bandwidth(N, timer.Elapsed());
    // copy data back from the GPU to the CPU
    HANDLER_ERROR_ERR(cudaMemcpy(h_b.elements, d_b.elements, ARRAY_BYTES,
                                 cudaMemcpyDeviceToHost));
    compareResults(h_a, h_b);

    // free GPU memory
    HANDLER_ERROR_ERR(cudaFree(d_a.elements));
    HANDLER_ERROR_ERR(cudaFree(d_b.elements));
}

void test(Matrix<int> h_a, Matrix<int> h_b) {
    transposedMatrixHost(h_a, h_b);
    compareResults(h_a, h_b);
}

void onHost() {
    Matrix<int> h_a, h_b, h_c;
    h_a.width = N;
    h_a.height = N;

    h_b.width = N;
    h_b.height = N;

    h_c.width = N;
    h_c.height = N;

    h_a.elements = (int*)malloc(h_a.width * h_b.height * sizeof(int));
    h_b.elements = (int*)malloc(h_b.width * h_b.height * sizeof(int));
    h_c.elements = (int*)malloc(h_b.width * h_b.height * sizeof(int));

    int i, j, k = 0;

    for (i = 0; i < h_a.width; i++) {
        for (j = 0; j < h_a.height; j++) {
            h_a.elements[j * h_a.width + i] = k;
            h_b.elements[j * h_b.width + i] = 0;
            k++;
        }
    }

    // call host function
    test(h_a, h_b);

    // call device configuration
    onDevice(h_a, h_c);

    printf("-: successful execution :-\n");
}

int main() {
    onHost();
}
