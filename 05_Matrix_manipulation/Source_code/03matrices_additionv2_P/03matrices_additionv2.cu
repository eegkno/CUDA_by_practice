#include <stdio.h>
#include <assert.h>
#include "Error.h"



#define N 500

__global__ void additionMatricesKernel(int *d_a, int *d_b, int *d_c){


	// -:YOUR CODE HERE:-	

}


void onDevice(int h_a[][N], int h_b[][N], int h_c[][N] ){

    // declare GPU memory pointers
	int *d_a, *d_b, *d_c;

	const int ARRAY_BYTES = N * N * sizeof(int);

    // allocate  memory on the GPU
	// -:YOUR CODE HERE:-		


    // copy data from CPU the GPU
	// -:YOUR CODE HERE:-		


    //execution configuration
	dim3 GridBlocks( 4,4 );
	dim3 ThreadsBlocks( 8,8 );
	
	 //run the kernel
	additionMatricesKernel<<<GridBlocks,ThreadsBlocks>>>( d_a, d_b, d_c );
    HANDLER_ERROR_MSG("kernel panic!!!");

    // copy data back from the GPU to the CPU
	// -:YOUR CODE HERE:-		


    // free GPU memory
	// -:YOUR CODE HERE:-		

}

void test(int h_a[][N], int h_b[][N], int h_c[][N] ){

	for(int i=0; i < N; i++){
		for(int j = 0; j < N; j++){
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

