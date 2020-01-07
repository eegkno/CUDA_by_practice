#include <assert.h>
#include <curand.h>
#include <omp.h>
#include "common/Error.h"
#include "common/GpuTimer.h"
#include "common/Vector.h"

#define N (32 * 1024)  // (32768/2)=16384
#define THREADS 256
#define BLOCKS 32

const int ARRAY_BYTES = N * sizeof(float);
const int P_ARRAY_BYTES = BLOCKS * sizeof(float);
int num_gpus = 0;

struct DataStruct {
    int deviceID;
    int size;
    int offset;
    float* a;
    float* b;
    float returnValue;
};

__global__ void dotKernel(Vector<float> d_a,
                          Vector<float> d_b,
                          Vector<float> d_c,
                          int size) {
    __shared__ float cache[THREADS];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;
    while (tid < size) {
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

    if (cacheIndex == 0)
        d_c.setElement(blockIdx.x, cache[0]);
}

void* onDevice(void* pvoidData) {
    DataStruct* data = (DataStruct*)pvoidData;
    if (data->deviceID != 0) {
        HANDLER_ERROR_ERR(cudaSetDevice(data->deviceID));
        HANDLER_ERROR_ERR(cudaSetDeviceFlags(cudaDeviceMapHost));
    }

    const int PARTIAL_ARRAY_SIZE = data->size;

    Vector<float> h_a, h_b, h_c;
    Vector<float> d_a, d_b, d_c;
    d_c.length = BLOCKS;

    // allocate memory on the CPU side
    h_a.elements = data->a;
    h_b.elements = data->b;
    h_c.elements = (float*)malloc(P_ARRAY_BYTES);

    // start timer
    GpuTimer timer;
    timer.Start();

    // allocate  memory on the GPU
    HANDLER_ERROR_ERR(cudaHostGetDevicePointer(&d_a.elements, h_a.elements, 0));
    HANDLER_ERROR_ERR(cudaHostGetDevicePointer(&d_b.elements, h_b.elements, 0));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_c.elements, P_ARRAY_BYTES));

    // offset 'a' and 'b' to where this GPU is getting its data
    d_a.elements += data->offset;
    d_b.elements += data->offset;

    dotKernel<<<BLOCKS, THREADS>>>(d_a, d_b, d_c, PARTIAL_ARRAY_SIZE);
    HANDLER_ERROR_MSG("kernel panic!!!");

    // copy data back from the GPU to the CPU
    HANDLER_ERROR_ERR(cudaMemcpy(h_c.elements, d_c.elements, P_ARRAY_BYTES,
                                 cudaMemcpyDeviceToHost));

    // finish up on the CPU side
    float partial = 0;
    for (int i = 0; i < BLOCKS; i++) {
        partial += h_c.getElement(i);
    }

    // stop timer
    timer.Stop();

    // print time
    printf("Time :  %f ms\n", timer.Elapsed());

    // free GPU memory
    HANDLER_ERROR_ERR(cudaFree(d_c.elements));

    // free CPU memory
    free(h_c.elements);
    data->returnValue = partial;
    return 0;
}

void test() {
    Vector<float> h_a, h_b;
    h_a.length = N;
    h_b.length = N;

    HANDLER_ERROR_ERR(cudaSetDevice(0));
    HANDLER_ERROR_ERR(cudaSetDeviceFlags(cudaDeviceMapHost));
    HANDLER_ERROR_ERR(cudaHostAlloc((void**)&h_a.elements, ARRAY_BYTES,
                                    cudaHostAllocWriteCombined |
                                        cudaHostAllocPortable |
                                        cudaHostAllocMapped));

    HANDLER_ERROR_ERR(cudaHostAlloc((void**)&h_b.elements, ARRAY_BYTES,
                                    cudaHostAllocWriteCombined |
                                        cudaHostAllocPortable |
                                        cudaHostAllocMapped));

    int i;
    for (i = 0; i < N; i++) {
        h_a.setElement(i, 1.0);
        h_b.setElement(i, 1.0);
    }

    // prepare for multithread
    DataStruct data[num_gpus];

    for (int i = 0; i < num_gpus; i++) {
        data[i].deviceID = i;
        data[i].offset = N / num_gpus * i;
        data[i].size = N / num_gpus;
        data[i].a = h_a.elements;
        data[i].b = h_b.elements;
    }

    printf("Number of GPUs %d, so %d threads created \n", num_gpus, num_gpus);
    omp_set_num_threads(num_gpus);
#pragma omp parallel
    {
        unsigned int cpu_thread_id = omp_get_thread_num();
        unsigned int num_cpu_threads = omp_get_num_threads();
        onDevice(&data[cpu_thread_id]);
    }

    float finalValue = 0;
    for (int i = 0; i < num_gpus; i++) {
        finalValue += data[i].returnValue;
    }

    printf("Dot result = %f \n", finalValue);
    assert(finalValue == N);

    printf("-: successful execution :-\n");

    HANDLER_ERROR_ERR(cudaFreeHost(h_a.elements));
    HANDLER_ERROR_ERR(cudaFreeHost(h_b.elements));
}

void checkDeviceProps() {
    // properties validation
    int deviceCount;
    HANDLER_ERROR_ERR(cudaGetDeviceCount(&deviceCount));
    cudaGetDeviceCount(&num_gpus);
    if (deviceCount < 2) {
        printf(
            "We need at least two compute 1.0 or greater "
            "devices, but only found %d\n",
            deviceCount);
    }

    cudaDeviceProp prop;
    for (int i = 0; i < deviceCount; i++) {
        HANDLER_ERROR_ERR(cudaGetDeviceProperties(&prop, i));
        if (prop.canMapHostMemory != 1) {
            printf("Device %d can not map memory.\n", i);
        }
    }
}

int main(void) {
    checkDeviceProps();
    test();
    return 0;
}
