#include <math.h>
#include <stdio.h>
#include "common/Error.h"
#include "common/GpuTimer.h"
#include "common/Vector.h"

#define N 10000
#define DIMGRID 10
#define DIMBLOCK 10
#define XMIN -10.0f
#define XMAX 10.0f

const int ARRAY_BYTES = N * sizeof(float);

__host__ __device__ float function1(float x) {
    return x * x;
}

__host__ __device__ float function2(float x) {
    return sinf(x);
}

__global__ void functionKernel1(Vector<float> d_a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    float x, dx;

    dx = (XMAX - (XMIN)) / ((float)N - 1);
    while (i < n) {
        x = XMIN + i * dx;
        d_a.setElement(i, function1(x));
        i += blockDim.x * gridDim.x;
    }
}

__global__ void functionKernel2(Vector<float> d_a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    float x, dx;

    dx = (XMAX - (XMIN)) / ((float)N - 1);

    while (i < n) {
        x = XMIN + i * dx;
        d_a.setElement(i, function2(x));
        i += blockDim.x * gridDim.x;
    }
}

void onDevice(Vector<float> h_a, Vector<float> h_b) {
    Vector<float> d_a, d_b;

    // create the stream
    cudaStream_t stream1;
    HANDLER_ERROR_ERR(cudaStreamCreate(&stream1));

    GpuTimer timer;
    timer.Start(stream1);

    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_a.elements, ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_b.elements, ARRAY_BYTES));

    functionKernel1<<<DIMGRID, DIMBLOCK, 0, stream1>>>(d_a, N);

    HANDLER_ERROR_MSG("kernel panic!!!");

    cudaDeviceSynchronize();

    functionKernel2<<<DIMGRID, DIMBLOCK, 0, stream1>>>(d_b, N);
    HANDLER_ERROR_MSG("kernel panic!!!");

    HANDLER_ERROR_ERR(cudaMemcpy(h_a.elements, d_a.elements, ARRAY_BYTES,
                                 cudaMemcpyDeviceToHost));
    HANDLER_ERROR_ERR(cudaMemcpy(h_b.elements, d_b.elements, ARRAY_BYTES,
                                 cudaMemcpyDeviceToHost));

    // stop timer
    timer.Stop(stream1);

    // print time
    printf("Time :  %f ms\n", timer.Elapsed());

    // destroy stream
    HANDLER_ERROR_ERR(cudaStreamDestroy(stream1));

    // free device memory
    HANDLER_ERROR_ERR(cudaFree(d_a.elements));
    HANDLER_ERROR_ERR(cudaFree(d_b.elements));
}

void checkDeviceProps() {
    cudaDeviceProp prop;
    int whichDevice;
    HANDLER_ERROR_ERR(cudaGetDevice(&whichDevice));
    HANDLER_ERROR_ERR(cudaGetDeviceProperties(&prop, whichDevice));
    if (!prop.deviceOverlap) {
        printf(
            "Device will not handle overlaps, so no speed up from streams\n");
    }
}

void test() {
    Vector<float> h_a, h_b;
    h_a.length = N;
    h_b.length = N;

    h_a.elements = (float*)malloc(ARRAY_BYTES);
    h_b.elements = (float*)malloc(ARRAY_BYTES);

    // call device configuration
    onDevice(h_a, h_b);

    // free host memory
    free(h_a.elements);
    free(h_b.elements);
}

int main() {
    checkDeviceProps();
    test();
}
