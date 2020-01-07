#include "common/Error.h"
#include "common/GpuTimer.h"
#include "common/Vector.h"

#define MARK 1
#define UNMARK 0
#define ARRAY_SIZE 64

const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);

__global__ void kernelSieve(int k, Vector<int> d_a) {
    // -:YOUR CODE HERE:-
}

void onDevice(Vector<int> h_a) {
    Vector<int> d_a;
    int k;

    // create the streams
    // -:YOUR CODE HERE:-

    HANDLER_ERROR_ERR(cudaMalloc(&d_a.elements, ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMemcpy(d_a.elements, h_a.elements, ARRAY_BYTES,
                                 cudaMemcpyHostToDevice));

    // kernel call
    // -:YOUR CODE HERE:-

    HANDLER_ERROR_ERR(cudaMemcpy(h_a.elements, d_a.elements, ARRAY_BYTES,
                                 cudaMemcpyDeviceToHost));
    HANDLER_ERROR_ERR(cudaFree(d_a.elements));

    // destroy stream
    // -:YOUR CODE HERE:-
}

void onHost() {
    Vector<int> h_a;
    h_a.length = ARRAY_SIZE;

    int j;
    h_a.elements = (int*)malloc(ARRAY_BYTES);

    for (j = 0; j < ARRAY_SIZE; j++) {
        h_a.setElement(j, j);
    }

    onDevice(h_a);

    for (j = 0; j < ARRAY_SIZE; j++) {
        if (h_a.getElement(j) > 1)
            printf("%i \n", h_a.getElement(j));
    }

    free(h_a.elements);
}

void checkDeviceProps() {
    // properties validation
    cudaDeviceProp prop;
    int whichDevice;
    HANDLER_ERROR_ERR(cudaGetDevice(&whichDevice));
    HANDLER_ERROR_ERR(cudaGetDeviceProperties(&prop, whichDevice));
    if (!prop.deviceOverlap) {
        printf(
            "Device will not handle overlaps, so no speed up from streams\n");
    }
}

int main() {
    checkDeviceProps();
    onHost();
}
