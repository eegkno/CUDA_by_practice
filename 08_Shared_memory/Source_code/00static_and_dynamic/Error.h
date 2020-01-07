#ifndef __ERROR_H__
#define __ERROR_H__

#include <stdio.h>

#define HANDLER_ERROR_ERR(err) (HandleError(err, __FILE__, __LINE__))

static void HandleError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

#define HANDLER_ERROR_MSG(msg) (cudaError(msg, __FILE__, __LINE__))

void cudaError(const char* msg, const char* file, int line) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "%s: %s in %s at line %d\n", msg,
                cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

#endif
