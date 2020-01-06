#include <assert.h>
#include <stdio.h>
#include "Error.h"
#include "Vector.h"

#define N 64
const int ARRAY_BYTES = N * sizeof(int);

__global__ void staticReverseKernel(Vector<int> d_a) {
    __shared__ int s[64];
    int t = threadIdx.x;
    int tr = N - t - 1;
    s[t] = d_a.getElement(t);
    __syncthreads();
    d_a.setElement(t, s[tr]);
}

__global__ void dynamicReverseKernel(Vector<int> d_a) {
    // -* note the empty brackets and the use of the extern specifier *-
    extern __shared__ int s[];
    int t = threadIdx.x;
    int tr = N - t - 1;
    s[t] = d_a.getElement(t);
    __syncthreads();
    d_a.setElement(t, s[tr]);
}

void test(Vector<int> d_b, Vector<int> d_c) {
    for (int i = 0; i < N; i++) {
        assert(d_b.getElement(i) == d_c.getElement(i));
    }
}

void onDevice(Vector<int> d_a, Vector<int> d_b, Vector<int> d_c) {
    Vector<int> d_d;
    d_d.length = N;

    // allocate  memory on the GPU
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_d.elements, ARRAY_BYTES));

    // copy data from CPU the GPU
    HANDLER_ERROR_ERR(cudaMemcpy(d_d.elements, d_a.elements, ARRAY_BYTES,
                                 cudaMemcpyHostToDevice));

    // run with STATIC shared memory
    staticReverseKernel<<<1, N>>>(d_d);
    HANDLER_ERROR_MSG("kernel panic!!!");

    // copy data back from the GPU to the CPU
    HANDLER_ERROR_ERR(cudaMemcpy(d_c.elements, d_d.elements, ARRAY_BYTES,
                                 cudaMemcpyDeviceToHost));

    // checing results
    test(d_b, d_c);

    // copy data from CPU the GPU
    HANDLER_ERROR_ERR(cudaMemcpy(d_d.elements, d_a.elements, ARRAY_BYTES,
                                 cudaMemcpyHostToDevice));

    // run with DYNAMIC shared memory
    dynamicReverseKernel<<<1, N, N * sizeof(int)>>>(d_d);

    HANDLER_ERROR_MSG("kernel panic!!!");

    // copy data back from the GPU to the CPU
    HANDLER_ERROR_ERR(cudaMemcpy(d_c.elements, d_d.elements, ARRAY_BYTES,
                                 cudaMemcpyDeviceToHost));

    // checing results
    test(d_b, d_c);

    HANDLER_ERROR_ERR(cudaFree(d_d.elements));
}

void onHost() {
    Vector<int> h_a, h_b, h_c;
    h_a.length = N;
    h_b.length = N;
    h_c.length = N;

    h_a.elements = (int*)malloc(ARRAY_BYTES);
    h_b.elements = (int*)malloc(ARRAY_BYTES);
    h_c.elements = (int*)malloc(ARRAY_BYTES);

    for (int i = 0; i < N; i++) {
        h_a.setElement(i, i);
        h_b.setElement(i, N - i - 1);
        h_c.setElement(i, 0);
    }

    // call device configuration
    onDevice(h_a, h_b, h_c);

    printf("-: successful execution :-\n");

    free(h_a.elements);
    free(h_b.elements);
    free(h_c.elements);
}

int main(void) {
    onHost();
}