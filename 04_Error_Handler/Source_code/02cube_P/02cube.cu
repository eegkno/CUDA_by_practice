#include <stdio.h>
#include <assert.h>
#include "Error.h"

__global__ void cubeKernel(float * d_out, float * d_in){

	// -:YOUR CODE HERE:-		

}



void onDevice(float *h_in, float *h_out,  int ARRAY_SIZE,  int ARRAY_BYTES){

	// declare GPU memory pointers
	// -:YOUR CODE HERE:-	

	// allocate GPU memory
	// -:YOUR CODE HERE:-	

	// transfer the array to the GPU
	// -:YOUR CODE HERE:-	

	// launch the kernel
	cubeKernel<<<1, ARRAY_SIZE>>>(d_out, d_in);

	// copy back the result array to the CPU
	// -:YOUR CODE HERE:-	

	// free GPU memory pointers
	// -:YOUR CODE HERE:-	

	
}


void test(float *h_in, float *h_out,  int ARRAY_SIZE,  int ARRAY_BYTES){


	// print out the resulting array
	for (int i =0; i < ARRAY_SIZE; i++) {
		assert( h_out[i] == (h_in[i] * h_in[i] * h_in[i]) );
		printf("%f", h_out[i]);
		printf(((i % 4) != 3) ? "\t" : "\n");

	}

	printf("-: successful execution :-\n");

}


void onHost(){

	const int ARRAY_SIZE = 64;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

	// declare CPU memory pointers
	// -:YOUR CODE HERE:-	

	// Allocate CPU memory pointers
	// -:YOUR CODE HERE:-

	for (int i = 0; i < ARRAY_SIZE; i++) {
		h_in[i] = float(i);
	}

	// call the kernel
	onDevice(h_in, h_out, ARRAY_SIZE, ARRAY_BYTES);
	test(h_in, h_out, ARRAY_SIZE, ARRAY_BYTES);


	// free CPU memory pointers
	// -:YOUR CODE HERE:-
}



int main(int argc, char ** argv) {

	onHost();

	return 0;
}