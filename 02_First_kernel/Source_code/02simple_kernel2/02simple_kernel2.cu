#include <assert.h>
#include <stdio.h>

__global__ void addKernel( int a, int b, int *c ) {
    *c = a + b;
}

int main( void ) {
    int h_c;
    int *d_c;
    const int C_BYTES = 1 * sizeof(int);


    //Save memory
    cudaMalloc( (void**)&d_c, C_BYTES );

    //Call the kernel
    addKernel<<<1,1>>>( 2, 7, d_c );

    //Copy memory from Device to Host
    cudaMemcpy( &h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost );
    
    assert( 2 + 7 == h_c);   
    printf("-: successful execution :-\n");

    //Free device memory
    cudaFree( d_c );

    return 0;
}
