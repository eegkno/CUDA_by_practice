#include <assert.h>
#include "Error.h"
#include "GpuTimer.h"
#include "Vector.h"

// check the constant memory amount

#define N (16 * 1024)
#define THREADS 256
#define BLOCKS 32

const int ARRAY_BYTES = N * sizeof(float);
const int P_ARRAY_BYTES = BLOCKS * sizeof(float);

// constant Vectors in GPU
__constant__ float d_a[N];
__constant__ float d_b[N];

__global__ void dotKernel(Vector<float> d_c) {
    __shared__ float cache[THREADS];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;
    while (tid < N) {
        temp += d_a[tid] * d_b[tid];
        tid += blockDim.x * gridDim.x;
    }

    // set the cache values
    cache[cacheIndex] = temp;

    // synchronize threads in this block
    __syncthreads();

    // for reductions, threadsPerBlock must be a power of 2
    // because of the following code
    int i = blockDim.x / 2;
    while (i != 0) {
        __syncthreads();
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        i /= 2;
    }

    if (cacheIndex == 0)
        d_c.setElement(blockIdx.x, cache[0]);
}

void onDevice(Vector<float> h_a, Vector<float> h_b, Vector<float> h_c) {
    Vector<float> d_c;
    d_c.length = BLOCKS;

    // start timer
    GpuTimer timer;
    timer.Start();

    int i;
    for (i = 0; i < 2; i++) {
        // allocate  memory on the GPU
        HANDLER_ERROR_ERR(cudaMemcpyToSymbol(d_a, h_a.elements, ARRAY_BYTES));
        HANDLER_ERROR_ERR(cudaMemcpyToSymbol(d_b, h_b.elements, ARRAY_BYTES));
        HANDLER_ERROR_ERR(cudaMalloc((void**)&d_c.elements, P_ARRAY_BYTES));

        dotKernel<<<BLOCKS, THREADS>>>(d_c);
        HANDLER_ERROR_MSG("kernel panic!!!");

        // copy data back from the GPU to the CPU
        HANDLER_ERROR_ERR(cudaMemcpy(h_c.elements, d_c.elements, P_ARRAY_BYTES,
                                     cudaMemcpyDeviceToHost));
    }

    // stop timer
    timer.Stop();

    // print time
    printf("Time :  %f ms\n", timer.Elapsed());

    // free GPU memory
    HANDLER_ERROR_ERR(cudaFree(d_c.elements));
}

void test() {
    Vector<float> h_a, h_b, h_c;
    h_a.length = N;
    h_b.length = N;
    h_c.length = BLOCKS;

    h_a.elements = (float*)malloc(ARRAY_BYTES);
    h_b.elements = (float*)malloc(ARRAY_BYTES);
    h_c.elements = (float*)malloc(P_ARRAY_BYTES);

    int i;

    for (i = 0; i < h_a.length; i++) {
        h_a.setElement(i, 1.0);
        h_b.setElement(i, 1.0);
    }

    // call device configuration
    onDevice(h_a, h_b, h_c);
    float d_dot_result = 0.0;

    // verify that the GPU did the work we requested
    for (int i = 0; i < BLOCKS; i++) {
        d_dot_result += h_c.getElement(i);
    }

    printf("Dot result from device = %f \n", d_dot_result);

    float h_dot_result = 0.0;
    for (int i = 0; i < N; i++) {
        h_dot_result += h_a.getElement(i) * h_b.getElement(i);
    }

    printf("Dot result from host = %f \n", h_dot_result);

    assert(d_dot_result == h_dot_result);

    printf("-: successful execution :-\n");

    free(h_a.elements);
    free(h_b.elements);
    free(h_c.elements);
}

int main(void) {
    test();
    return 0;
}
