#include <assert.h>
#include "Error.h"
#include "GpuTimer.h"
#include "Vector.h"

const int BLOCKSIZE = 128;
const int NUMBLOCKS = 1000;
const int N = BLOCKSIZE * NUMBLOCKS;
const int ARRAY_BYTES = N * sizeof(int);

__global__ void tileKernelv1(Vector<int> d_a,
                             Vector<int> d_b,
                             Vector<int> d_c,
                             Vector<int> d_d,
                             Vector<int> d_e,
                             Vector<int> d_out) {
    // Change next operation in order to use the tiling technique
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    d_out.elements[i] =
        (d_a.getElement(i) + d_b.getElement(i) + d_c.getElement(i) +
         d_d.getElement(i) + d_e.getElement(i)) /
        5.0f;

    // -:YOUR CODE HERE:-
}

void onDevice(Vector<int> h_a,
              Vector<int> h_b,
              Vector<int> h_c,
              Vector<int> h_d,
              Vector<int> h_e,
              Vector<int> h_out) {
    Vector<int> d_a, d_b, d_c, d_d, d_e, d_out;
    d_a.length = N;
    d_b.length = N;
    d_c.length = N;
    d_d.length = N;
    d_e.length = N;
    d_out.length = N;

    // allocate  memory on the GPU
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_a.elements, ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_b.elements, ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_c.elements, ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_d.elements, ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_e.elements, ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_out.elements, ARRAY_BYTES));

    // copy data from CPU the GPU
    HANDLER_ERROR_ERR(cudaMemcpy(d_a.elements, h_a.elements, ARRAY_BYTES,
                                 cudaMemcpyHostToDevice));
    HANDLER_ERROR_ERR(cudaMemcpy(d_b.elements, h_b.elements, ARRAY_BYTES,
                                 cudaMemcpyHostToDevice));
    HANDLER_ERROR_ERR(cudaMemcpy(d_c.elements, h_b.elements, ARRAY_BYTES,
                                 cudaMemcpyHostToDevice));
    HANDLER_ERROR_ERR(cudaMemcpy(d_d.elements, h_b.elements, ARRAY_BYTES,
                                 cudaMemcpyHostToDevice));
    HANDLER_ERROR_ERR(cudaMemcpy(d_e.elements, h_b.elements, ARRAY_BYTES,
                                 cudaMemcpyHostToDevice));

    // launch kernel
    tileKernelv1<<<N / BLOCKSIZE, BLOCKSIZE>>>(d_a, d_b, d_c, d_d, d_e, d_out);
    HANDLER_ERROR_MSG("kernel panic!!!");

    // copy data back from the GPU to the CPU
    HANDLER_ERROR_ERR(cudaMemcpy(h_out.elements, d_out.elements, ARRAY_BYTES,
                                 cudaMemcpyDeviceToHost));

    // free GPU memory
    HANDLER_ERROR_ERR(cudaFree(d_a.elements));
    HANDLER_ERROR_ERR(cudaFree(d_b.elements));
    HANDLER_ERROR_ERR(cudaFree(d_c.elements));
    HANDLER_ERROR_ERR(cudaFree(d_d.elements));
    HANDLER_ERROR_ERR(cudaFree(d_e.elements));
    HANDLER_ERROR_ERR(cudaFree(d_out.elements));
}

void test(Vector<int> h_a,
          Vector<int> h_b,
          Vector<int> h_c,
          Vector<int> h_d,
          Vector<int> h_e,
          Vector<int> h_out) {
    int aux = 0;
    for (int i = 0; i < N - 2; i++) {
        aux = (h_a.getElement(i) + h_b.getElement(i) + h_c.getElement(i) +
               h_d.getElement(i) + h_e.getElement(i)) /
              5.0f;
        assert(aux == h_out.getElement(i + 2));
    }
}

void onHost() {
    Vector<int> h_a, h_b, h_c, h_d, h_e, h_out;
    h_a.length = N;
    h_b.length = N;
    h_c.length = N;
    h_d.length = N;
    h_e.length = N;
    h_out.length = N;

    h_a.elements = (int*)malloc(ARRAY_BYTES);
    h_b.elements = (int*)malloc(ARRAY_BYTES);
    h_c.elements = (int*)malloc(ARRAY_BYTES);
    h_d.elements = (int*)malloc(ARRAY_BYTES);
    h_e.elements = (int*)malloc(ARRAY_BYTES);
    h_out.elements = (int*)malloc(ARRAY_BYTES);

    for (int i = 0; i < N; i++) {
        h_a.setElement(i, i);
        h_b.setElement(i, i + 1);
        h_c.setElement(i, i + 2);
        h_d.setElement(i, i + 3);
        h_e.setElement(i, i + 4);
    }

    // call device configuration
    onDevice(h_a, h_b, h_c, h_d, h_e, h_out);

    test(h_a, h_b, h_c, h_d, h_e, h_out);

    printf("-: successful execution :-\n");

    free(h_a.elements);
    free(h_b.elements);
    free(h_c.elements);
    free(h_d.elements);
    free(h_e.elements);
    free(h_out.elements);
}

int main(void) {
    onHost();
    return 0;
}