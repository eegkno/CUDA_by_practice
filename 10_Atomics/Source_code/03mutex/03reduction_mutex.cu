#include <assert.h>
#include "common/Error.h"
#include "common/GpuTimer.h"
#include "common/Lock.h"
#include "common/Vector.h"

#define N (32 * 1024)
#define THREADS 256
#define BLOCKS 32

const int ARRAY_BYTES = N * sizeof(float);

__global__ void dotKernel(Lock lock,
                          Vector<float> d_a,
                          Vector<float> d_b,
                          float* d_dotValue) {
    __shared__ float cache[THREADS];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;
    while (tid < N) {
        temp += d_a.getElement(tid) * d_b.getElement(tid);
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

    /*At this point, the packed data is stored in thc cache array for each block
    created. Now, that value needs to be stored in d_dotValue in order to have
    the final value. It is necessary to lock d_dotValue to avoid the threads get
    the wrong value.
    */

    if (cacheIndex == 0) {
        // Wait until we get the lock. Each thread needs to get the lock value
        // before updating d_dotValue

        lock.lock();
        // we have the lock at this point, update and release
        *d_dotValue += cache[0];
        lock.unlock();
    }
}

void onDevice(Vector<float> h_a, Vector<float> h_b, float& h_dotValue) {
    Vector<float> d_a, d_b;
    d_a.length = N;
    d_b.length = N;

    float* d_dotValue;

    // start timer
    GpuTimer timer;
    timer.Start();

    // allocate  memory on the GPU
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_a.elements, ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_b.elements, ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_dotValue, sizeof(float)));

    // copy data from CPU the GPU
    HANDLER_ERROR_ERR(cudaMemcpy(d_a.elements, h_a.elements, ARRAY_BYTES,
                                 cudaMemcpyHostToDevice));
    HANDLER_ERROR_ERR(cudaMemcpy(d_b.elements, h_b.elements, ARRAY_BYTES,
                                 cudaMemcpyHostToDevice));
    HANDLER_ERROR_ERR(cudaMemcpy(d_dotValue, &h_dotValue, sizeof(float),
                                 cudaMemcpyHostToDevice));

    Lock lock;
    dotKernel<<<BLOCKS, THREADS>>>(lock, d_a, d_b, d_dotValue);
    HANDLER_ERROR_MSG("kernel panic!!!");

    // copy data back from the GPU to the CPU
    HANDLER_ERROR_ERR(cudaMemcpy(&h_dotValue, d_dotValue, sizeof(float),
                                 cudaMemcpyDeviceToHost));

    // stop timer
    timer.Stop();

    // print time
    printf("Time :  %f ms\n", timer.Elapsed());

    // free GPU memory
    HANDLER_ERROR_ERR(cudaFree(d_a.elements));
    HANDLER_ERROR_ERR(cudaFree(d_b.elements));
    HANDLER_ERROR_ERR(cudaFree(d_dotValue));
}

void onHost() {
    float d_dotValue = 0.0f;

    Vector<float> h_a, h_b;
    h_a.length = N;
    h_b.length = N;

    h_a.elements = (float*)malloc(ARRAY_BYTES);
    h_b.elements = (float*)malloc(ARRAY_BYTES);

    int i;

    for (i = 0; i < h_a.length; i++) {
        h_a.setElement(i, 1.0);
        h_b.setElement(i, 1.0);
    }

    // call device configuration
    onDevice(h_a, h_b, d_dotValue);

    printf("Dot result from device %.2f\n", d_dotValue);

    float h_dotValue = 0.0f;
    for (i = 0; i < h_a.length; i++) {
        h_dotValue += h_a.getElement(i) * h_b.getElement(i);
    }

    printf("Dot result from host %.2f\n", h_dotValue);

    assert(d_dotValue == h_dotValue);

    printf("-: successful execution :-\n");

    free(h_a.elements);
    free(h_b.elements);
}

int main(void) {
    onHost();
    return 0;
}
