#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "common/Error.h"
#include "common/GpuTimer.h"
#include "common/Vector.h"

#define FULL_DATA_SIZE 100000000
#define DIMGRID 10
#define DIMBLOCK 10
#define XMIN -10.0f
#define XMAX 10.0f

const int FULL_ARRAY_BYTES = FULL_DATA_SIZE * sizeof(float);

__host__ __device__ float function1(float x) {
    return x * x;
}

__host__ __device__ float function2(float x) {
    return sinf(x);
}

__global__ void functionKernel1(Vector<float> d_a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    float x, dx;

    dx = (XMAX - (XMIN)) / ((float)FULL_DATA_SIZE - 1);
    while (i < n) {
        x = XMIN + i * dx;
        d_a.setElement(i, function1(x));
        i += blockDim.x * gridDim.x;
    }
}

__global__ void functionKernel2(Vector<float> d_a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    float x, dx;

    dx = (XMAX - (XMIN)) / ((float)FULL_DATA_SIZE - 1);

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

    HANDLER_ERROR_ERR(cudaHostGetDevicePointer(&d_a.elements, h_a.elements, 0));
    HANDLER_ERROR_ERR(cudaHostGetDevicePointer(&d_b.elements, h_b.elements, 0));

    functionKernel1<<<DIMGRID, DIMBLOCK, 0, stream1>>>(d_a, FULL_DATA_SIZE);
    HANDLER_ERROR_MSG("kernel panic!!!");

    functionKernel2<<<DIMGRID, DIMBLOCK, 0, stream1>>>(d_b, FULL_DATA_SIZE);
    HANDLER_ERROR_MSG("kernel panic!!!");

    HANDLER_ERROR_ERR(cudaStreamSynchronize(stream1));

    // stop timer
    timer.Stop(stream1);

    // print time
    printf("Time :  %f ms\n", timer.Elapsed());

    // destroy stream
    HANDLER_ERROR_ERR(cudaStreamDestroy(stream1));
}

void checkDeviceProps() {
    cudaDeviceProp prop;
    int whichDevice;
    HANDLER_ERROR_ERR(cudaGetDevice(&whichDevice));
    HANDLER_ERROR_ERR(cudaGetDeviceProperties(&prop, whichDevice));
    if (!prop.canMapHostMemory) {
        printf("Device cannot map host to device memory\n");
    } else {
        printf("Device test passed\n");
    }
}

void test() {
    Vector<float> h_a, h_b;
    h_a.length = FULL_DATA_SIZE;
    h_b.length = FULL_DATA_SIZE;

    // This flag must be set in order to allocate pinned
    // host memory that is accessible by the device
    HANDLER_ERROR_ERR(cudaSetDeviceFlags(cudaDeviceMapHost));

    // allocate host locked memory
    HANDLER_ERROR_ERR(
        cudaHostAlloc((void**)&h_a.elements, FULL_ARRAY_BYTES,
                      cudaHostAllocWriteCombined | cudaHostAllocMapped));
    HANDLER_ERROR_ERR(
        cudaHostAlloc((void**)&h_b.elements, FULL_ARRAY_BYTES,
                      cudaHostAllocWriteCombined | cudaHostAllocMapped));

    int i;
    for (i = 0; i < FULL_DATA_SIZE; i++) {
        h_a.setElement(i, 1.0);
        h_b.setElement(i, 1.0);
    }

    // call device configuration
    onDevice(h_a, h_b);

    printf("-: successful execution :-\n");

    // free host memory
    HANDLER_ERROR_ERR(cudaFreeHost(h_a.elements));
    HANDLER_ERROR_ERR(cudaFreeHost(h_b.elements));
}

int main() {
    checkDeviceProps();
    test();
}