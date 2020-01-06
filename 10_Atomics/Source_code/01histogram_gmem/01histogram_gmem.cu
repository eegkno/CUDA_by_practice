#include <assert.h>
#include <time.h>
#include "common/CpuTimer.h"
#include "common/Error.h"
#include "common/GpuTimer.h"
#include "common/Vector.h"

#define SIZE (100 * 1024 * 1024)
#define SIZE_H 256

const int ARRAY_BYTES = SIZE * sizeof(unsigned int);
const int ARRAY_BYTES_H = SIZE_H * sizeof(int);

__global__ void histo_kernel(Vector<unsigned int> d_a, Vector<int> d_histo) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int tmp = 0;
    while (i < SIZE) {
        tmp = d_a.getElement(i);
        // The kernel serializes the memory access in order to avoid the threads
        // trying to write the same location at the same time which implies
        // a long line of threads waiting to work
        atomicAdd(&d_histo.elements[tmp], 1);
        i += stride;
    }
}

void onDevice(Vector<unsigned int> h_a, Vector<int> h_histo) {
    Vector<unsigned int> d_a;
    d_a.length = SIZE;

    Vector<int> d_histo;
    d_histo.length = SIZE_H;

    // start timer
    GpuTimer timer;
    timer.Start();

    // allocate memory on the GPU for the file's data
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_a.elements, ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_histo.elements, ARRAY_BYTES_H));

    // copy data from CPU the GPU
    HANDLER_ERROR_ERR(cudaMemcpy(d_a.elements, h_a.elements, ARRAY_BYTES,
                                 cudaMemcpyHostToDevice));
    HANDLER_ERROR_ERR(cudaMemcpy(d_histo.elements, h_histo.elements,
                                 ARRAY_BYTES_H, cudaMemcpyHostToDevice));

    // kernel launch - 2x the number of mps gave best timing
    cudaDeviceProp prop;
    HANDLER_ERROR_ERR(cudaGetDeviceProperties(&prop, 0));
    int blocks = prop.multiProcessorCount;

    histo_kernel<<<blocks * 2, SIZE_H>>>(d_a, d_histo);
    HANDLER_ERROR_MSG("kernel panic!!!");

    HANDLER_ERROR_ERR(cudaMemcpy(h_histo.elements, d_histo.elements,
                                 ARRAY_BYTES_H, cudaMemcpyDeviceToHost));

    // stop timer
    timer.Stop();

    // print time
    printf("GPU Time :  %f ms\n", timer.Elapsed());

    // free device memory
    cudaFree(d_histo.elements);
    cudaFree(d_a.elements);
}

void test(Vector<unsigned int> h_a, Vector<int> h_histo, Vector<int> h_test) {
    // start timer
    CpuTimer timer;
    timer.Start();

    for (int i = 0; i < SIZE; i++) {
        h_test.elements[h_a.elements[i]]++;
    }

    // stop timer
    timer.Stop();

    // print time
    printf("CPU Time :  %f ms\n", timer.Elapsed());

    for (int i = 0; i < SIZE_H; i++) {
        // printf(" [%i]  %i | %i \n", i, h_histo.getElement(i),
        // h_test.getElement(i));
        assert(h_histo.getElement(i) == h_test.getElement(i));
    }

    printf("-: successful execution :-\n");
}

void onHost() {
    Vector<unsigned int> h_a;
    h_a.length = SIZE;
    h_a.elements = (unsigned int*)malloc(ARRAY_BYTES);
    h_a.randomInit(0, 255);

    Vector<int> h_histo;
    h_histo.length = SIZE_H;
    h_histo.elements = (int*)malloc(ARRAY_BYTES_H);
    h_histo.zerosInit();

    Vector<int> h_test;
    h_test.length = SIZE_H;
    h_test.elements = (int*)malloc(ARRAY_BYTES_H);
    h_test.zerosInit();

    onDevice(h_a, h_histo);

    test(h_a, h_histo, h_test);

    free(h_a.elements);
    free(h_histo.elements);
    free(h_test.elements);
}

int main(void) {
    onHost();

    return 0;
}
