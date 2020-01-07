#include <stdio.h>
#include <time.h>
#include "common/CpuTimer.h"
#include "common/Error.h"
#include "common/GpuTimer.h"

#define N 1024
#define ITER 100000

const int ARRAY_BYTES = N * sizeof(int);

typedef struct {
    int a;
    int b;
    int c;
    int d;
} aos;

typedef struct {
    int* a;
    int* b;
    int* c;
    int* d;
} soa;

__global__ void kernelInterleaved(aos* h_in, aos* h_out, int iter) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        for (int i = 0; i < iter; ++i) {
            h_out[tid].a += h_in[tid].a;
            h_out[tid].b += h_in[tid].b;
            h_out[tid].c += h_in[tid].c;
            h_out[tid].d += h_in[tid].d;
        }
    }
}

__global__ void kernelNonInterleaved(soa h_in, soa h_out, int iter) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        for (int i = 0; i < iter; i++) {
            h_out.a[tid] += h_in.a[tid];
            h_out.b[tid] += h_in.b[tid];
            h_out.c[tid] += h_in.c[tid];
            h_out.d[tid] += h_in.d[tid];
        }
    }
}

void onDevice(aos h_inaos[],
              aos h_outaos[],
              soa h_insoa,
              soa h_outsoa,
              int iter) {
    // declare GPU memory pointers (soa)
    soa d_insoa;
    soa d_outsoa;

    // start timer (soa)
    GpuTimer timersoa;
    timersoa.Start();

    // allocate  memory on the GPU (soa)
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_insoa.a, ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_insoa.b, ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_insoa.c, ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_insoa.d, ARRAY_BYTES));

    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_outsoa.a, ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_outsoa.b, ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_outsoa.c, ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_outsoa.d, ARRAY_BYTES));

    // copy data from CPU the GPU (soa)
    HANDLER_ERROR_ERR(
        cudaMemcpy(d_insoa.a, h_insoa.a, ARRAY_BYTES, cudaMemcpyHostToDevice));
    HANDLER_ERROR_ERR(
        cudaMemcpy(d_insoa.b, h_insoa.b, ARRAY_BYTES, cudaMemcpyHostToDevice));
    HANDLER_ERROR_ERR(
        cudaMemcpy(d_insoa.c, h_insoa.c, ARRAY_BYTES, cudaMemcpyHostToDevice));
    HANDLER_ERROR_ERR(
        cudaMemcpy(d_insoa.d, h_insoa.d, ARRAY_BYTES, cudaMemcpyHostToDevice));

    // execution configuration
    dim3 GridBlocks(4096);
    dim3 ThreadsBlocks(256);

    // run the kernel
    kernelNonInterleaved<<<GridBlocks, ThreadsBlocks>>>(d_insoa, d_outsoa,
                                                        ITER);
    HANDLER_ERROR_MSG("kernel panic!!!");

    // copy data back from the GPU to the CPU (soa)
    HANDLER_ERROR_ERR(cudaMemcpy(h_outsoa.a, d_outsoa.a, ARRAY_BYTES,
                                 cudaMemcpyDeviceToHost));
    HANDLER_ERROR_ERR(cudaMemcpy(h_outsoa.b, d_outsoa.b, ARRAY_BYTES,
                                 cudaMemcpyDeviceToHost));
    HANDLER_ERROR_ERR(cudaMemcpy(h_outsoa.c, d_outsoa.c, ARRAY_BYTES,
                                 cudaMemcpyDeviceToHost));
    HANDLER_ERROR_ERR(cudaMemcpy(h_outsoa.d, d_outsoa.d, ARRAY_BYTES,
                                 cudaMemcpyDeviceToHost));

    timersoa.Stop();

    // print time
    printf("SOA on Device took %f ms.\n", timersoa.Elapsed());

    // free GPU memory (soa)
    HANDLER_ERROR_ERR(cudaFree(d_insoa.a));
    HANDLER_ERROR_ERR(cudaFree(d_insoa.b));
    HANDLER_ERROR_ERR(cudaFree(d_insoa.c));
    HANDLER_ERROR_ERR(cudaFree(d_insoa.d));

    HANDLER_ERROR_ERR(cudaFree(d_outsoa.a));
    HANDLER_ERROR_ERR(cudaFree(d_outsoa.b));
    HANDLER_ERROR_ERR(cudaFree(d_outsoa.c));
    HANDLER_ERROR_ERR(cudaFree(d_outsoa.d));

    aos* d_inaos;
    aos* d_outaos;

    const int ARRAY_BYTES_AOS = N * sizeof(aos);

    // start timer (aos)
    GpuTimer timeraos;
    timeraos.Start();

    // allocate  memory on the GPU (aos)
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_inaos, ARRAY_BYTES_AOS));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_outaos, ARRAY_BYTES_AOS));

    // copy data from CPU the GPU (aos)
    HANDLER_ERROR_ERR(
        cudaMemcpy(d_inaos, h_inaos, ARRAY_BYTES_AOS, cudaMemcpyHostToDevice));

    // run the kernel
    kernelInterleaved<<<GridBlocks, ThreadsBlocks>>>(d_inaos, d_outaos, ITER);
    HANDLER_ERROR_MSG("kernel panic!!!");

    // copy data back from the GPU to the CPU (saos)
    HANDLER_ERROR_ERR(
        cudaMemcpy(h_outaos, d_inaos, ARRAY_BYTES_AOS, cudaMemcpyDeviceToHost));

    timeraos.Stop();

    // print time
    printf("AOS on Device took %f ms.\n", timeraos.Elapsed());

    // free GPU memory (aos)
    HANDLER_ERROR_ERR(cudaFree(d_inaos));
    HANDLER_ERROR_ERR(cudaFree(d_outaos));
}

float interleavedHost(aos h_in[], aos h_out[], int iter) {
    CpuTimer timeraos;
    timeraos.Start();

    for (int tid = 0; tid < N; tid++) {
        for (int i = 0; i < iter; i++) {
            h_out[tid].a += h_in[tid].a;
            h_out[tid].b += h_in[tid].b;
            h_out[tid].c += h_in[tid].c;
            h_out[tid].d += h_in[tid].d;
        }
    }

    timeraos.Stop();
    return timeraos.Elapsed();
}

float nonInterleavedHost(soa h_in, soa h_out, int iter) {
    CpuTimer timersoa;
    timersoa.Start();

    for (int tid = 0; tid < N; tid++) {
        for (int i = 0; i < iter; i++) {
            h_out.a[tid] += h_in.a[tid];
            h_out.b[tid] += h_in.b[tid];
            h_out.c[tid] += h_in.c[tid];
            h_out.d[tid] += h_in.d[tid];
        }
    }

    timersoa.Stop();
    return timersoa.Elapsed();
}

void onHost(aos h_inaos[],
            aos h_outaos[],
            soa h_insoa,
            soa h_outsoa,
            int iter) {
    printf("SOA on Host took %f ms.\n",
           ((float)nonInterleavedHost(h_insoa, h_outsoa, ITER)));
    printf("AOS on Host took %f ms.\n",
           ((float)interleavedHost(h_inaos, h_outaos, ITER)));
}

void test() {
    aos h_inaos[N];
    aos h_outaos[N];

    soa h_insoa;
    soa h_outsoa;

    h_insoa.a = (int*)malloc(ARRAY_BYTES);
    h_insoa.b = (int*)malloc(ARRAY_BYTES);
    h_insoa.c = (int*)malloc(ARRAY_BYTES);
    h_insoa.d = (int*)malloc(ARRAY_BYTES);

    h_outsoa.a = (int*)malloc(ARRAY_BYTES);
    h_outsoa.b = (int*)malloc(ARRAY_BYTES);
    h_outsoa.c = (int*)malloc(ARRAY_BYTES);
    h_outsoa.d = (int*)malloc(ARRAY_BYTES);

    for (int j = 0; j < N; j++) {
        h_inaos[j].a = j;
        h_inaos[j].b = j;
        h_inaos[j].c = j;
        h_inaos[j].d = j;

        h_insoa.a[j] = j;
        h_insoa.b[j] = j;
        h_insoa.c[j] = j;
        h_insoa.d[j] = j;
    }

    onHost(h_inaos, h_outaos, h_insoa, h_outsoa, ITER);
    printf("\n");
    onDevice(h_inaos, h_outaos, h_insoa, h_outsoa, ITER);

    free(h_insoa.a);
    free(h_insoa.b);
    free(h_insoa.c);
    free(h_insoa.d);

    free(h_outsoa.a);
    free(h_outsoa.b);
    free(h_outsoa.c);
    free(h_outsoa.d);
}

int main() {
    test();
}
