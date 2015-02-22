#include <stdio.h>
#include <assert.h>
#include "Error.h"



#define N 1000

__global__ void transposedMatrixKernel(int *d_a, int *d_b){


	// -:YOUR CODE HERE:-	
}


void onDevice(int h_a[][N], int h_b[][N]  ){

    // declare GPU memory pointers
	int *d_a, *d_b;

	const int ARRAY_BYTES = N * N * sizeof(int);

    // allocate  memory on the GPU
	// -:YOUR CODE HERE:-	

    // copy data from CPU the GPU
	// -:YOUR CODE HERE:-	


    //execution configuration
	dim3 GridBlocks( 4,4 );
	dim3 ThreadsBlocks( 16,16 );
	
	 //run the kernel
	transposedMatrixKernel<<<GridBlocks,ThreadsBlocks>>>( d_a, d_b );
    HANDLER_ERROR_MSG("kernel panic!!!");

    // copy data back from the GPU to the CPU
	// -:YOUR CODE HERE:-

    // free GPU memory
	// -:YOUR CODE HERE:-

}


void test(int h_a[][N], int h_b[][N] ){

	// test  result
	for(i=0; i < N; i++){
		for(j = 0; j < N; j++){
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

