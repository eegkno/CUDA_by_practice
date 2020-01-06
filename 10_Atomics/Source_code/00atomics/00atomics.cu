#include <assert.h>
#include <time.h>
#include "common/CpuTimer.h"
#include "common/Error.h"
#include "common/GpuTimer.h"
#include "common/Vector.h"

#define SIZE 100

const int ARRAY_BYTES = SIZE * sizeof(int);

__global__ void incrementAtomicsKernel(Vector<int> d_b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    i = i % SIZE;
    atomicAdd(&d_b.elements[i], 1);
}

__global__ void incrementKernel(Vector<int> d_a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    i = i % SIZE;
    d_a.setElement(d_a.getElement(i) + 1, i);
}

void onDevice(Vector<int> h_a, Vector<int> h_b) {
    Vector<int> d_a;
    d_a.length = SIZE;

    Vector<int> d_b;
    d_b.length = SIZE;

    //---------
    // start timer
    GpuTimer timer;
    timer.Start();

    // allocate memory on the GPU for the file's data
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_a.elements, ARRAY_BYTES));

    // copy data from CPU the GPU
    HANDLER_ERROR_ERR(cudaMemcpy(d_a.elements, h_a.elements, ARRAY_BYTES,
                                 cudaMemcpyHostToDevice));

    incrementKernel<<<100, 1024>>>(d_a);
    HANDLER_ERROR_MSG("kernel panic!!!");

    HANDLER_ERROR_ERR(cudaMemcpy(h_a.elements, d_a.elements, ARRAY_BYTES,
                                 cudaMemcpyDeviceToHost));

    // stop timer
    timer.Stop();

    // print time
    printf("GPU Time no atomics:  %f ms\n", timer.Elapsed());

    //-----------

    // start timer
    GpuTimer timerA;
    timerA.Start();

    // allocate memory on the GPU for the file's data
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_b.elements, ARRAY_BYTES));

    // copy data from CPU the GPU
    HANDLER_ERROR_ERR(cudaMemcpy(d_b.elements, h_b.elements, ARRAY_BYTES,
                                 cudaMemcpyHostToDevice));

    incrementAtomicsKernel<<<100, 512>>>(d_b);
    HANDLER_ERROR_MSG("kernel panic!!!");

    HANDLER_ERROR_ERR(cudaMemcpy(h_b.elements, d_b.elements, ARRAY_BYTES,
                                 cudaMemcpyDeviceToHost));

    // stop timer
    timerA.Stop();

    // print time
    printf("GPU Time with Atomics:  %f ms\n", timerA.Elapsed());

    // free device memory
    cudaFree(d_b.elements);
    cudaFree(d_a.elements);
}

void onHost() {
    Vector<int> h_a;
    h_a.length = SIZE;
    h_a.elements = (int*)malloc(ARRAY_BYTES);
    h_a.zerosInit();

    Vector<int> h_b;
    h_b.length = SIZE;
    h_b.elements = (int*)malloc(ARRAY_BYTES);
    h_a.zerosInit();

    onDevice(h_a, h_b);

    h_a.print();
    h_b.print();

    free(h_a.elements);
    free(h_b.elements);
}

int main(void) {
    onHost();

    return 0;
}
