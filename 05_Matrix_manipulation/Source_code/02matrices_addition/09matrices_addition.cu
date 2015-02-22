#include <stdio.h>
#include <assert.h>
#include "Error.h"

/*
**Execution Config 1
* N = 4
**Execution Config 2
* N = 16 
**Execution Config 3
* N = 32 
**Execution Config 4
* N = 64
*/

#define N 64

__global__ void additionMatricesKernel(int *d_a, int *d_b, int *d_c){


        int i = threadIdx.x+blockIdx.x*blockDim.x;
        int j = threadIdx.y+blockIdx.y*blockDim.y;        

        d_c[ i*N+j ] = d_a[i*N+j] + d_b[i*N+j]; 

}


void onDevice(int h_a[][N], int h_b[][N], int h_c[][N] ){

    // declare GPU memory pointers
	int *d_a, *d_b, *d_c;

	const int ARRAY_BYTES = N * N * sizeof(int);

    // allocate  memory on the GPU
	HANDLER_ERROR_ERR(cudaMalloc((void**)&d_a,ARRAY_BYTES));
	HANDLER_ERROR_ERR(cudaMalloc((void**)&d_b,ARRAY_BYTES));
	HANDLER_ERROR_ERR(cudaMalloc((void**)&d_c,ARRAY_BYTES));

    // copy data from CPU the GPU
	HANDLER_ERROR_ERR(cudaMemcpy(d_a, h_a, ARRAY_BYTES, cudaMemcpyHostToDevice));
	HANDLER_ERROR_ERR(cudaMemcpy(d_b, h_b, ARRAY_BYTES, cudaMemcpyHostToDevice));
	HANDLER_ERROR_ERR(cudaMemcpy(d_c, h_c, ARRAY_BYTES, cudaMemcpyHostToDevice));

    //execution configuration
	dim3 GridBlocks( N/2,N/2 );
	dim3 ThreadsBlocks( N/2,N/2 );
	
	 //run the kernel
	additionMatricesKernel<<<GridBlocks,ThreadsBlocks>>>( d_a, d_b, d_c );
    HANDLER_ERROR_MSG("kernel panic!!!");

    // copy data back from the GPU to the CPU
	HANDLER_ERROR_ERR(cudaMemcpy(h_c, d_c, ARRAY_BYTES, cudaMemcpyDeviceToHost));	

    // free GPU memory
	HANDLER_ERROR_ERR(cudaFree(d_a));
	HANDLER_ERROR_ERR(cudaFree(d_b));
	HANDLER_ERROR_ERR(cudaFree(d_c));
}


void test(int h_a[][N], int h_b[][N], int h_c[][N]){
	
	int i,j;
	for(i=0; i < N; i++){
		for(j = 0; j < N; j++){
			assert(h_a[i][j] + h_b[i][j] == h_c[i][j]);
		}
	}

    printf("-: successful execution :-\n");
}


void onHost(){


	int i,j;
	int h_a[N][N], h_b[N][N], h_c[N][N];

	for(i = 0; i < N; i++){
		for(j = 0; j < N; j++){
			h_a[i][j] = h_b[i][j] = i+j;
			h_c[i][j] = 0;
		}
	}

    // call device configuration
	onDevice(h_a,h_b,h_c);
	test(h_a,h_b,h_c);


}


int main(){

	onHost();
}

