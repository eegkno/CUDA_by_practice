#include <assert.h>
#include "Error.h"
#include "GpuTimer.h"
#include "Vector.h"

#define N 16
#define BLOCK_SIZE 2
#define STRIDE 4
#define POW(x) (x) * (x)

__global__ void mseKernel(Vector<float> d_a,
                          Vector<float> d_b,
                          Vector<float> d_c) {
    __shared__ float cache[4];

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index (current coefficient)
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Indixes
    int idx = (by * BLOCK_SIZE + ty) * STRIDE + (bx * BLOCK_SIZE + tx);
    int ith = ty * BLOCK_SIZE + tx;

    // operation
    cache[ith] = POW(d_a.getElement(idx) - d_b.getElement(idx));

    __syncthreads();

    int i = 2;
    while (i != 0) {
        if (ith < i)
            cache[ith] += cache[ith + i];
        __syncthreads();
        i /= 2;
    }

    int bidx = by * BLOCK_SIZE + bx;
    if (ith == 0)
        d_c.setElement(bidx, cache[0]);
}

void onDevice(Vector<float> h_a, Vector<float> h_b, Vector<float> h_c) {
    Vector<float> d_a, d_b, d_c;
    d_a.length = N;
    d_b.length = N;
    d_c.length = 4;

    // start timer
    GpuTimer timer;
    timer.Start();

    const int ARRAY_BYTES = N * sizeof(float);

    // allocate  memory on the GPU
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_a.elements, ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_b.elements, ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_c.elements, 4 * sizeof(float)));

    // copy data from CPU the GPU
    HANDLER_ERROR_ERR(cudaMemcpy(d_a.elements, h_a.elements, ARRAY_BYTES,
                                 cudaMemcpyHostToDevice));
    HANDLER_ERROR_ERR(cudaMemcpy(d_b.elements, h_b.elements, ARRAY_BYTES,
                                 cudaMemcpyHostToDevice));

    // execution configuration
    dim3 GridBlocks(2, 2);
    dim3 ThreadsBlocks(2, 2);

    mseKernel<<<GridBlocks, ThreadsBlocks>>>(d_a, d_b, d_c);
    HANDLER_ERROR_MSG("kernel panic!!!");

    // copy data back from the GPU to the CPU
    HANDLER_ERROR_ERR(cudaMemcpy(h_c.elements, d_c.elements, 4 * sizeof(float),
                                 cudaMemcpyDeviceToHost));

    // stop timer
    timer.Stop();

    // print time
    printf("Time :  %f ms\n", timer.Elapsed());

    // free GPU memory
    HANDLER_ERROR_ERR(cudaFree(d_a.elements));
    HANDLER_ERROR_ERR(cudaFree(d_b.elements));
    HANDLER_ERROR_ERR(cudaFree(d_c.elements));
}

void test() {
    Vector<float> h_a, h_b, h_c;
    h_a.length = N;
    h_b.length = N;
    h_c.length = 4;

    h_a.elements = (float*)malloc(h_a.length * sizeof(float));
    h_b.elements = (float*)malloc(h_a.length * sizeof(float));
    h_c.elements = (float*)malloc(4 * sizeof(float));

    int i, j = 16, k = 1;

    for (i = 0; i < h_a.length; i++) {
        h_a.setElement(i, k);
        h_b.setElement(i, j);
        k++;
        j--;
    }

    // call device configuration
    onDevice(h_a, h_b, h_c);

    // verify that the GPU did the work we requested
    float d_mse = 0, h_mse = 0;
    for (int i = 0; i < 4; i++) {
        d_mse += h_c.getElement(i);
        printf(" [%i] = %f \n", i, h_c.getElement(i));
    }
    d_mse /= N;
    printf("mse from device: %f\n", d_mse);

    for (int i = 0; i < N; i++) {
        h_mse += POW(h_a.getElement(i) - h_b.getElement(i));
    }
    h_mse /= N;
    printf("mse from host: %f\n", h_mse);

    assert(d_mse == h_mse);

    printf("-: successful execution :-\n");

    free(h_a.elements);
    free(h_b.elements);
    free(h_c.elements);
}

int main(void) {
    test();
    return 0;
}
