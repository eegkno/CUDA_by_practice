#include <assert.h>
#include <stdio.h>
#include "common/CpuTimer.h"
#include "common/Error.h"
#include "common/GpuTimer.h"
#include "common/Matrix.h"

#define N 4096
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
    // (i,j) locations of the tile corners for input & output matrices:
    int in_corner_i = blockIdx.x * blockDim.x,
        in_corner_j = blockIdx.y * blockDim.y;
    int x = threadIdx.x, y = threadIdx.y;

    __shared__ float tile[K][K];

    while (in_corner_j + x < N) {
        in_corner_i = blockIdx.x * blockDim.x;
        in_corner_i = blockIdx.x * blockDim.x;
        while (in_corner_i + y < N) {
            tile[y][x] =
                d_a.elements[(in_corner_i + x) + (in_corner_j + y) * N];

            __syncthreads();

            d_b.elements[(in_corner_j + x) + (in_corner_i + y) * N] =
                tile[x][y];

            in_corner_i += blockDim.x * gridDim.x;
            in_corner_i += blockDim.x * gridDim.x;
        }
        in_corner_j += gridDim.y * blockDim.y;
        in_corner_j += gridDim.y * blockDim.y;
    }
}

void onDevice(Matrix<int> h_a, Matrix<int> h_b) {
    Matrix<int> d_a, d_b;
    d_a.width = h_a.width;
    d_a.height = h_a.height;

    d_b.width = h_b.width;
    d_b.height = h_b.height;

    dim3 GridBlocks(N / K, N / K);
    dim3 ThreadsBlocks(K, K);

    GpuTimer timer;
    timer.Start();
    HANDLER_ERROR_ERR(cudaHostGetDevicePointer(&d_a.elements, h_a.elements, 0));
    HANDLER_ERROR_ERR(cudaHostGetDevicePointer(&d_b.elements, h_b.elements, 0));

    transposedMatrixKernelFinal<<<GridBlocks, ThreadsBlocks>>>(d_a, d_b);
    HANDLER_ERROR_MSG("kernel panic!!!");
    timer.Stop();

    printf("Time :  %f ms\n", timer.Elapsed());

    compareResults(h_a, h_b);
}

void onHost() {
    Matrix<int> h_a, h_b;
    h_a.width = N;
    h_a.height = N;

    h_b.width = N;
    h_b.height = N;

    HANDLER_ERROR_ERR(cudaSetDeviceFlags(cudaDeviceMapHost));

    HANDLER_ERROR_ERR(cudaHostAlloc(
        (void**)&h_a.elements, h_a.width * h_a.height * sizeof(int),
        cudaHostAllocWriteCombined | cudaHostAllocMapped));
    HANDLER_ERROR_ERR(cudaHostAlloc(
        (void**)&h_b.elements, h_b.width * h_b.height * sizeof(int),
        cudaHostAllocWriteCombined | cudaHostAllocMapped));

    int i, j, k = 0;

    for (i = 0; i < h_a.width; i++) {
        for (j = 0; j < h_a.height; j++) {
            h_a.elements[j * h_a.width + i] = k;
            h_b.elements[j * h_b.width + i] = 0;
            k++;
        }
    }
    // call device configuration
    onDevice(h_a, h_b);

    printf("-: successful execution :-\n");

    // free host memory
    HANDLER_ERROR_ERR(cudaFreeHost(h_a.elements));
    HANDLER_ERROR_ERR(cudaFreeHost(h_b.elements));
}

int main() {
    onHost();
}
