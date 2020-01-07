#include <assert.h>
#include <stdio.h>
#include "Error.h"

#define N 1024

// if we add
//#define K 16

typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

__device__ float getElement(Matrix A, int row, int col) {
    return A.elements[row * A.width + col];
}

__device__ void setElement(Matrix A, int row, int col, float value) {
    A.elements[row * A.width + col] = value;
}

__global__ void transposedMatrixKernel(Matrix d_a, Matrix d_b) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    while (i < d_a.width) {
        j = threadIdx.y + blockDim.y * blockIdx.y;
        while (j < d_a.width) {
            setElement(d_b, i, j, getElement(d_a, j, i));
            j += blockDim.y * gridDim.y;
        }
        i += blockDim.x * gridDim.x;
    }
}

void onDevice(Matrix h_a, Matrix h_b) {
    // declare GPU data
    Matrix d_a, d_b;
    d_a.width = h_a.width;
    d_a.height = h_a.height;

    d_b.width = h_b.width;
    d_b.height = h_b.height;

    const int ARRAY_BYTES = d_a.width * d_a.height * sizeof(int);

    // create events
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // allocate  memory on the GPU
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_a.elements, ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_b.elements, ARRAY_BYTES));

    // copy data from CPU the GPU
    HANDLER_ERROR_ERR(cudaMemcpy(d_a.elements, h_a.elements, ARRAY_BYTES,
                                 cudaMemcpyHostToDevice));
    HANDLER_ERROR_ERR(cudaMemcpy(d_b.elements, h_b.elements, ARRAY_BYTES,
                                 cudaMemcpyHostToDevice));

    // execution configuration
    dim3 GridBlocks(2, 2);
    dim3 ThreadsBlocks(16, 16);

    // execution configuration
    // dim3 GridBlocks( N/K,N/K );  //(64,64)
    // dim3 ThreadsBlocks( K,K );   //(16,16)

    // run the kernel
    transposedMatrixKernel<<<GridBlocks, ThreadsBlocks>>>(d_a, d_b);
    HANDLER_ERROR_MSG("kernel panic!!!");

    // copy data back from the GPU to the CPU
    HANDLER_ERROR_ERR(cudaMemcpy(h_b.elements, d_b.elements, ARRAY_BYTES,
                                 cudaMemcpyDeviceToHost));

    // stop events
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // print time
    printf("Time :  %f ms\n", elapsedTime);

    // destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // free GPU memory
    HANDLER_ERROR_ERR(cudaFree(d_a.elements));
    HANDLER_ERROR_ERR(cudaFree(d_b.elements));
}

void test() {
    Matrix h_a, h_b;
    h_a.width = N;
    h_a.height = N;

    h_b.width = N;
    h_b.height = N;

    h_a.elements = (float*)malloc(h_a.width * h_b.height * sizeof(int));
    h_b.elements = (float*)malloc(h_b.width * h_b.height * sizeof(int));

    int i, j, k = 0;

    for (i = 0; i < h_a.height; i++) {
        for (j = 0; j < h_a.width; j++) {
            h_a.elements[i * h_a.width + j] = k;
            h_b.elements[i * h_b.width + j] = 0;
            k++;
        }
    }

    // call device configuration
    onDevice(h_a, h_b);

    // test  result
    for (i = 0; i < h_a.height; i++) {
        for (j = 0; j < h_a.height; j++) {
            assert(h_a.elements[j * h_a.width + i] ==
                   h_b.elements[i * h_b.width + j]);
        }
    }

    printf("-: successful execution :-\n");
}

int main() {
    test();
}
