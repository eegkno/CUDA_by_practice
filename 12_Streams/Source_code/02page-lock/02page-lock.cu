#include <stdio.h>
#include "common/Error.h"
#include "common/GpuTimer.h"

#define SIZE (1024 * 1024)
const int ARRAY_BYTES = SIZE * sizeof(int);

float cuda_malloc_test(bool up) {
    int *a, *d_a;
    float elapsedTime;

    GpuTimer timer;

    a = (int*)malloc(ARRAY_BYTES);
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_a, ARRAY_BYTES));

    timer.Start();

    for (int i = 0; i < 100; i++) {
        if (up) {
            HANDLER_ERROR_ERR(
                cudaMemcpy(d_a, a, ARRAY_BYTES, cudaMemcpyHostToDevice));

        } else {
            HANDLER_ERROR_ERR(
                cudaMemcpy(a, d_a, ARRAY_BYTES, cudaMemcpyDeviceToHost));
        }
    }
    timer.Stop();
    elapsedTime = timer.Elapsed();

    free(a);
    HANDLER_ERROR_ERR(cudaFree(d_a));

    return elapsedTime;
}

float cuda_host_alloc_test(bool up) {
    int *a, *d_a;
    float elapsedTime;

    GpuTimer timer;
    timer.Start();

    HANDLER_ERROR_ERR(
        cudaHostAlloc((void**)&a, ARRAY_BYTES, cudaHostAllocDefault));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_a, ARRAY_BYTES));

    for (int i = 0; i < 100; i++) {
        if (up) {
            cudaMemcpy(d_a, a, ARRAY_BYTES, cudaMemcpyHostToDevice);

        } else {
            cudaMemcpy(a, d_a, ARRAY_BYTES, cudaMemcpyDeviceToHost);
        }
    }

    timer.Stop();
    elapsedTime = timer.Elapsed();

    HANDLER_ERROR_ERR(cudaFreeHost(a));
    HANDLER_ERROR_ERR(cudaFree(d_a));

    return elapsedTime;
}

void test() {
    // 1024 bytes = 1k
    // 1 048 576  =  1024 K = 1M
    // 1 M = 1 048 576 bytes
    // 4M  = 4 194 304 bytes

    // 100 = iterations
    float elapsedTime;
    float MB = (float)100 * SIZE * sizeof(int) / 1024 / 1024;

    // try it with cudaMalloc
    elapsedTime = cuda_malloc_test(true);
    printf("Time using cudaMalloc:  %3.1f ms\n\n", elapsedTime);
    printf("\tMB/s during copy up:  %3.1f\n\n", MB / (elapsedTime / 1000));

    elapsedTime = cuda_malloc_test(false);
    printf("Time using cudaMalloc:  %3.1f ms\n\n", elapsedTime);
    printf("\tMB/s during copy down:  %3.1f\n\n", MB / (elapsedTime / 1000));

    // now try it with cudaHostAlloc
    elapsedTime = cuda_host_alloc_test(true);
    printf("Time using cudaHostAlloc:  %3.1f ms\n\n", elapsedTime);
    printf("\tMB/s during copy up:  %3.1f\n\n", MB / (elapsedTime / 1000));

    elapsedTime = cuda_host_alloc_test(false);
    printf("Time using cudaHostAlloc:  %3.1f ms\n\n", elapsedTime);
    printf("\tMB/s during copy down:  %3.1f\n\n", MB / (elapsedTime / 1000));
}

int main(void) {
    test();
}
