#include <stdio.h>
#include <assert.h>

#define ARRAY_SIZE 5
const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);

// Kernel definition
__global__ void addKernel(int* d_a, int* d_b, int* d_c)
{
    int i = threadIdx.x;
    d_c[i] = d_a[i] + d_b[i];
}


void onDevice(int* h_a, int* h_b, int* h_c){
   int *d_c;

  //allocate memory on the device
  cudaMalloc( (void**)&d_c, ARRAY_BYTES );

  addKernel<<<1, ARRAY_SIZE>>>(h_a, h_b, d_c);

  //Copy memory from Device to Host
  cudaMemcpy( &h_c, d_c, ARRAY_BYTES, cudaMemcpyDeviceToHost );

  cudaFree( d_c );
}


void onHost()
{
  int h_a[ARRAY_SIZE];
	int h_b[ARRAY_SIZE];
	int *h_c;

  	//allocate memory on the host
  	h_c = (int*)malloc(ARRAY_BYTES);

  	for(int i =0; i< ARRAY_SIZE; i++){

  		h_a[i] = 1;
  		h_b[i] = 1;
  		h_c[i] = 0;
  	}

  	onDevice(h_a, h_b, h_c);

  	for(int i =0; i< ARRAY_SIZE; i++){
  		printf("%i \n",h_a[0]);
      assert(h_a[i] + h_b[i] == h_c[i]);
	  }

  	printf("-: successful execution :-\n");

  	free(h_c);
}

int main(){

	onHost();
	return 0;
}
