#include <assert.h>
#include <stdio.h>
#include <cuda.h>


#define ARRAY_SIZE 1000
#define THREADS 3
#define BLOCKS 7
const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);


// index of the threads in x        -> threadIdx.x
// index of the blocks in x         -> blockIdx.x
// number of threads per block in x -> blockDim.x
// number of blocks per grid in x   -> gridDim.x

//kernel 
__global__ void addKernel(  int  *d_a, int *d_b, int *d_result){
   
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
    while(idx < ARRAY_SIZE){ 
      d_result[idx] = d_a[idx] + d_b[idx];
      //d_result[idx] = idx;
      idx += THREADS * gridDim.x;
    }    
}


__global__ void addKernel2(  int  *d_a, int *d_b, int *d_result){
   
  int idx = threadIdx.x + blockIdx.x * gridDim.x;
    while(idx < ARRAY_SIZE){ 
      d_result[idx] = d_a[idx] + d_b[idx];
      //d_result[idx] = idx;
      idx += blockDim.x;
    }  
}


__global__ void addKernel3(  int  *d_a, int *d_b, int *d_result){
  
  int SIZE = 143;

  int idx = threadIdx.x + blockIdx.x * SIZE;
    while(idx < (SIZE*(blockIdx.x+1)) && idx < ARRAY_SIZE ){ 
      d_result[idx] = d_a[idx] + d_b[idx];
      //d_result[idx] = idx;
      idx += blockDim.x;
    }  
}

void onDevice(int  *h_a, int *h_b, int *h_result){

  int *d_a, *d_b, *d_result; 

   //allocate memory on the device
  cudaMalloc( (void**)&d_a, ARRAY_BYTES );
  cudaMalloc( (void**)&d_b, ARRAY_BYTES );
  cudaMalloc( (void**)&d_result, ARRAY_BYTES );

    //copythe arrays 'a' and 'b' to the device
  cudaMemcpy( d_a, h_a, ARRAY_BYTES, cudaMemcpyHostToDevice );
  cudaMemcpy( d_b, h_b, ARRAY_BYTES, cudaMemcpyHostToDevice );

  //run the kernel
  addKernel<<<BLOCKS,THREADS>>>( d_a, d_b, d_result);

    // copy the array 'result' back from the device to the CPU
  cudaMemcpy( h_result, d_result, ARRAY_BYTES, cudaMemcpyDeviceToHost );

    // check the results
  for (int i=0; i<ARRAY_SIZE; i++) {
    assert( h_a[i] + h_b[i] == h_result[i] );
    //printf("%i\n", h_result[i] );
  }

  //run the kernel
  addKernel3<<<BLOCKS,THREADS>>>( d_a, d_b, d_result);

    // copy the array 'result' back from the device to the CPU
  cudaMemcpy( h_result, d_result, ARRAY_BYTES, cudaMemcpyDeviceToHost );

    // check the results
  for (int i=0; i<ARRAY_SIZE; i++) {
    assert( h_a[i] + h_b[i] == h_result[i] );
    //printf("%i\n", h_result[i] );
  }

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
        h_a[i] = -i;
        h_b[i] = i * i;
    h_result[i]=0;
  }
    
  onDevice(h_a, h_b, h_result);

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

