#include <assert.h>
#include <stdio.h>
#include <cuda.h>


//define necessary variables
  // -:YOUR CODE HERE:-  


const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);

//kernel
__global__ void addKernel(  int  *d_a, int *d_b, int *d_result){
  // index of the threads in x        -> threadIdx.x
  // index of the blocks in x         -> blockIdx.x
  // number of threads per block in x -> blockDim.x
  // number of blocks per grid in x   -> gridDim.x

  // -:YOUR CODE HERE:-  
}


void onDevice( int  *h_a, int *h_b, int *h_result ){

  int *d_a, *d_b, *d_result; 

  //allocate memory on the device
  // -:YOUR CODE HERE:-  

  //copythe arrays 'a' and 'b' to the device
  // -:YOUR CODE HERE:-  

  //run the kernel
  addKernel<<<BLOCKS,THREADS>>>( d_a, d_b, d_result);

  // copy the array 'result' back from the device to the CPU
  // -:YOUR CODE HERE:-  

  // check the results
  for (int i=0; i<ARRAY_SIZE; i++) {
    assert( h_a[i] + h_b[i] == h_result[i] );
    //printf("%i\n", h_result[i] );
  }

  // free device memory
  // -:YOUR CODE HERE:-  
    
}

void onHost(){

  int *h_a, *h_b, *h_result;

  //allocate memory on the host
  // -:YOUR CODE HERE:-  

  // filling the arrays 'a' and 'b' on the CPU
  for (int i=0; i<ARRAY_SIZE; i++) {
      h_a[i] = -i;
      h_b[i] = i * i;
      h_result[i]=0;
  }

  onDevice(h_a, h_b, h_result);

  // check the results
  for (int i=0; i<ARRAY_SIZE; i++) {
    assert( h_a[i] + h_b[i] == h_result[i] );
  }

  printf("-: successful execution :-\n");

  // free host memory 
  // -:YOUR CODE HERE:-  

}

int main(){   
  onHost();
  return 0;
}

