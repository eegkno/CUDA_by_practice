#include <curand.h>
#include "common/Error.h"
#include "common/GpuTimer.h"
#include "common/Vector.h"

#define N (32 * 1024)
#define THREADS 256
#define BLOCKS 32

const int ARRAY_BYTES = N * sizeof(float);
const int P_ARRAY_BYTES = BLOCKS * sizeof(float);

__global__ void dotKernel(Vector<float> d_a,
                          Vector<float> d_b,
                          Vector<float> d_c) {
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

    if (cacheIndex == 0)
        d_c.setElement(blockIdx.x, cache[0]);
}

int randomNumbersGenerator(Vector<float> d_data, int n) {
    curandGenerator_t gen;

    /* Create pseudo-random number generator */
    HANDLER_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

    /* Set seed 1234ULL = unsigned long long */
    srand48(time(NULL));
    HANDLER_CURAND(curandSetPseudoRandomGeneratorSeed(gen, lrand48()));

    /* Set seed 1234ULL = unsigned long long
    Use this to generate the same random numbers*/
    // HANDLER_CURAND(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

    /* Generate n floats on device */
    HANDLER_CURAND(curandGenerateUniform(gen, d_data.elements, n));

    /* Cleanup */
    HANDLER_CURAND(curandDestroyGenerator(gen));

    return EXIT_SUCCESS;
}

void onDevice(Vector<float> h_c) {
    Vector<float> d_a, d_b, d_c;
    d_a.length = N;
    d_b.length = N;
    d_c.length = BLOCKS;

    // start timer
    GpuTimer timer;
    timer.Start();

    // allocate  memory on the GPU
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_a.elements, ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_b.elements, ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_c.elements, P_ARRAY_BYTES));

    randomNumbersGenerator(d_a, N);
    randomNumbersGenerator(d_b, N);

    dotKernel<<<BLOCKS, THREADS>>>(d_a, d_b, d_c);
    HANDLER_ERROR_MSG("kernel panic!!!");

    // copy data back from the GPU to the CPU
    HANDLER_ERROR_ERR(cudaMemcpy(h_c.elements, d_c.elements, P_ARRAY_BYTES,
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
    Vector<float> h_c;

    // declare vectore to store results
    h_c.length = BLOCKS;

    h_c.elements = (float*)malloc(P_ARRAY_BYTES);

    // call device configuration
    onDevice(h_c);

    float finalValue = 0.0;

    // verify that the GPU did the work we requested
    for (int i = 0; i < BLOCKS; i++) {
        finalValue += h_c.getElement(i);
    }

    printf("Dot result = %f \n", finalValue);

    printf("-: successful execution :-\n");

    free(h_c.elements);
}

int main(void) {
    test();
    return 0;
}