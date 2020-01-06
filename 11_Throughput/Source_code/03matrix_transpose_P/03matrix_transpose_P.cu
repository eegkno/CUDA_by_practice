#include <assert.h>
#include <stdio.h>
#include "common/CpuTimer.h"
#include "common/Error.h"
#include "common/GpuTimer.h"
#include "common/Matrix.h"

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

__global__ void transposedMatrixKernelFinal(Matrix<int> d_a, Matrix<int> d_b) {
    // -:YOUR CODE HERE:-
    // (i,j) locations of the tile corners for input & output matrices:
    int in_corner_i = blockIdx.x * blockDim.x,
        in_corner_j = blockIdx.y * blockDim.y;
    int out_corner_i = blockIdx.y * blockDim.y,
        out_corner_j = blockIdx.x * blockDim.x;

    int x = threadIdx.x, y = threadIdx.y;

    __shared__ float tile[K][K];

    // -:YOUR CODE HERE:-

    // coalesced read from global mem, TRANSPOSED write into shared mem:
    tile[y][x] = d_a.elements[(in_corner_i + x) + (in_corner_j + y) * N];
    __syncthreads();
    // read from shared mem, coalesced write to global mem:
    d_b.elements[(out_corner_i + x) + (out_corner_j + y) * N] = tile[x][y];

    // -:YOUR CODE HERE:-
}

__global__ void transposedMatrixKernel_tile_padded(Matrix<int> d_a,
                                                   Matrix<int> d_b) {
    // (i,j) locations of the tile corners for input & output matrices:
    int in_corner_i = blockIdx.x * blockDim.x,
        in_corner_j = blockIdx.y * blockDim.y;
    int out_corner_i = blockIdx.y * blockDim.y,
        out_corner_j = blockIdx.x * blockDim.x;

    int x = threadIdx.x, y = threadIdx.y;

    __shared__ float tile[K][K + 1];

    // coalesced read from global mem, TRANSPOSED write into shared mem:
    tile[y][x] = d_a.elements[(in_corner_i + x) + (in_corner_j + y) * N];
    __syncthreads();
    // read from shared mem, coalesced write to global mem:
    d_b.elements[(out_corner_i + x) + (out_corner_j + y) * N] = tile[x][y];
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

    dim3 GridBlocks(N / K, N / K);
    dim3 ThreadsBlocks(K, K);

    timer.Start();
    transposedMatrixKernel_tile_padded<<<GridBlocks, ThreadsBlocks>>>(d_a, d_b);
    HANDLER_ERROR_MSG("kernel panic!!!");
    timer.Stop();
    printf("Time Device tile-padded:  %f ms\n", timer.Elapsed());
    // copy data back from the GPU to the CPU
    HANDLER_ERROR_ERR(cudaMemcpy(h_b.elements, d_b.elements, ARRAY_BYTES,
                                 cudaMemcpyDeviceToHost));
    compareResults(h_a, h_b);

    // GridBlocks( 4,4 );
    // ThreadsBlocks( 2,2 );

    timer.Start();
    transposedMatrixKernelFinal<<<GridBlocks, ThreadsBlocks>>>(d_a, d_b);
    HANDLER_ERROR_MSG("kernel panic!!!");
    timer.Stop();
    printf("Time Device final:  %f ms\n", timer.Elapsed());
    // copy data back from the GPU to the CPU
    HANDLER_ERROR_ERR(cudaMemcpy(h_b.elements, d_b.elements, ARRAY_BYTES,
                                 cudaMemcpyDeviceToHost));
    compareResults(h_a, h_b);

    // free GPU memory
    HANDLER_ERROR_ERR(cudaFree(d_a.elements));
    HANDLER_ERROR_ERR(cudaFree(d_b.elements));
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

    // call device configuration
    onDevice(h_a, h_c);

    printf("-: successful execution :-\n");
}

int main() {
    onHost();
}
