#include <stdio.h>
#include <assert.h>
#include "Error.h"



#define N 4

__global__ void transposedMatrixKernel(int *d_a, int *d_b){

	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;

 	d_b[i*N+j] = d_a[j*N+i];

}


void onDevice(int h_a[][N], int h_b[][N]  ){

    // declare GPU memory pointers
	int *d_a, *d_b;

	const int ARRAY_BYTES = N * N * sizeof(int);

    // allocate  memory on the GPU
	HANDLER_ERROR_ERR(cudaMalloc((void**)&d_a,ARRAY_BYTES));
	HANDLER_ERROR_ERR(cudaMalloc((void**)&d_b,ARRAY_BYTES));


    // copy data from CPU the GPU
	HANDLER_ERROR_ERR(cudaMemcpy(d_a, h_a, ARRAY_BYTES, cudaMemcpyHostToDevice));
	HANDLER_ERROR_ERR(cudaMemcpy(d_b, h_b, ARRAY_BYTES, cudaMemcpyHostToDevice));


    //execution configuration
	dim3 GridBlocks( 1,1 );
	dim3 ThreadsBlocks( 4,4 );
	
	 //run the kernel
	transposedMatrixKernel<<<GridBlocks,ThreadsBlocks>>>( d_a, d_b );
    HANDLER_ERROR_MSG("kernel panic!!!");

    // copy data back from the GPU to the CPU
	HANDLER_ERROR_ERR(cudaMemcpy(h_b, d_b, ARRAY_BYTES, cudaMemcpyDeviceToHost));	

    // free GPU memory
	HANDLER_ERROR_ERR(cudaFree(d_a));
	HANDLER_ERROR_ERR(cudaFree(d_b));

}



void test(int h_a[][N], int h_b[][N] ){

	// test  result
	for(int i=0; i < N; i++){
		for(int j = 0; j < N; j++){
			assert(h_a[j][i] == h_b[i][j]);

		}
	}

    printf("-: successful execution :-\n");

}


void onHost(){


	int i,j,k=0;
	int h_a[N][N], h_b[N][N];

	for(i = 0; i < N; i++){
		for(j = 0; j < N; j++){
			h_a[i][j] = k;
			h_b[i][j] = 0;
			k++;
		}
	}

    // call device configuration
	onDevice(h_a,h_b);
	test(h_a,h_b);

}


int main(){

	onHost();
}

