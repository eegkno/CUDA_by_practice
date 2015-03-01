#include <assert.h>
#include <stdio.h>
#include <cuda.h>

#define ARRAY_SIZE  10
const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);

//kernel
__global__ void addKernel(  int  *d_a, int *d_b, int *d_result){
   int idx = threadIdx.x;
   d_result[idx] = d_a[idx] + d_b[idx];
}


void onDevice( int  *h_a, int *h_b, int *h_result ){

	int *d_a, *d_b, *d_result; 

   //allocate memory on the device
	cudaMalloc( (void**)&d_a, ARRAY_BYTES );
	cudaMalloc( (void**)&d_b, ARRAY_BYTES );
	cudaMalloc( (void**)&d_result, ARRAY_BYTES );

    //copythe arrays 'a' and 'b' to the device
	cudaMemcpy( d_a, h_a, ARRAY_BYTES, cudaMemcpyHostToDevice );
	cudaMemcpy( d_b, h_b, ARRAY_BYTES, cudaMemcpyHostToDevice );

    //run the kernel
    addKernel<<<1,ARRAY_SIZE>>>( d_a, d_b, d_result);

    // copy the array 'result' back from the device to the CPU
	cudaMemcpy( h_result, d_result, ARRAY_BYTES, cudaMemcpyDeviceToHost );

    // free device memory
	cudaFree( d_a );
	cudaFree( d_b );
	cudaFree( d_result );

}

void onHost(){

	int *h_a, *h_b, *h_result;
	
	//allocate memory on the host
	h_a = (int*)malloc(ARRAY_BYTES);
	h_b = (int*)malloc(ARRAY_BYTES);
	h_result = (int*)malloc(ARRAY_BYTES);

    // filling the arrays 'a' and 'b' on the CPU
    for (int i=0; i<ARRAY_SIZE; i++) {
        h_a[i] = i;
        h_b[i] = i*i;
		h_result[i]=0;
    }

    onDevice(h_a, h_b, h_result);

    // check the results
    for (int i=0; i<ARRAY_SIZE; i++) {
		assert( h_a[i] + h_b[i] == h_result[i] );
    }

	printf("-: successful execution :-\n");
    
    // free host memory 
	free(h_a);
	free(h_b);
	free(h_result);
}

int main(){   
	onHost();
    return 0;
}

