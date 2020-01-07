/*
Source: http://docs.nvidia.com/cuda/curand/index.html#topic_1_2_2
*/

#include <curand.h>
#include <stdio.h>
#include <stdlib.h>
#include "common/Error.h"

int randomNumbersGenerator(int n) {
    size_t i;
    curandGenerator_t gen;
    float *d_data, *h_data;

    const int ARRAY_BYTES = n * sizeof(float);

    /* Allocate n floats on host */
    h_data = (float*)calloc(n, sizeof(float));

    /* Allocate n floats on device */
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_data, ARRAY_BYTES));

    /* Create pseudo-random number generator */
    HANDLER_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

    /* Set seed 1234ULL = unsigned long long */
    srand48(time(NULL));
    HANDLER_CURAND(curandSetPseudoRandomGeneratorSeed(gen, lrand48()));

    /* Set seed 1234ULL = unsigned long long
    Use this to generate the same random numbers*/
    // HANDLER_CURAND(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

    /* Generate n floats on device */
    HANDLER_CURAND(curandGenerateUniform(gen, d_data, n));

    /* Copy device memory to host */
    HANDLER_ERROR_ERR(
        cudaMemcpy(h_data, d_data, ARRAY_BYTES, cudaMemcpyDeviceToHost));

    /* Show result */
    for (i = 0; i < n; i++) {
        printf("%1.4f ", h_data[i]);
    }
    printf("\n");

    /* Cleanup */
    HANDLER_CURAND(curandDestroyGenerator(gen));
    HANDLER_ERROR_ERR(cudaFree(d_data));
    free(h_data);
    return EXIT_SUCCESS;
}

int main(int argc, char* argv[]) {
    randomNumbersGenerator(100);
}