#include <assert.h>
#include <stdio.h>
#include "Error.h"

int one = 1;
__device__ int two = 0;


// 1. remove __global__

__global__ void addKernel( int a, int b, int *c ) {
    *c = a + b;
// 3. variable one can't be modified in the kernel
    //one = a + b;
    two = a + b;
    
// 5. variale one can't be read in the kernel    
    two = (a + b) * one;
}

int main( void ) {
    int h_c;
    int *d_c;
    int h_one;
    const int C_BYTES = 1 * sizeof(int);


    //Allocate memory
    HANDLER_ERROR_ERR(cudaMalloc( (void**)&d_c, C_BYTES ));

// 2. chage execution configuration
    
    //Call the kernel
    addKernel<<<1,1>>>( 2, 7, d_c );
    HANDLER_ERROR_MSG( "Kernel Panic!!!" );
    cudaDeviceSynchronize();
    //cudaThreadSynchronize();   //used in old driver versions

    //Copy memory from Device to Host
    HANDLER_ERROR_ERR(cudaMemcpy( &h_c, d_c, C_BYTES, cudaMemcpyDeviceToHost ));   
    HANDLER_ERROR_ERR(cudaMemcpy( &h_one, d_c, C_BYTES, cudaMemcpyDeviceToHost ));
 
    assert( 2 + 7 == h_c);  
    assert( 2 + 7 == h_one);  

// 4. variable two can be modified, but with a warning advice
    two = 4;
    printf("%i\n", two);

    printf("-: successful execution :-\n");

    //Free device memory
    HANDLER_ERROR_ERR(cudaFree( d_c ));

    return 0;
}
