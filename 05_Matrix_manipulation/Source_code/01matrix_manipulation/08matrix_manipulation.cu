#include "Error.h"


#define BLOCK_SIZE 3
#define STRIDE 6

__global__ void matrixKernel( float *d_in, float *d_out ){

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

       
    // Thread index (current coefficient)
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float dividend =  d_in[ (by * BLOCK_SIZE + 0) * STRIDE + (bx * BLOCK_SIZE + 0) ];
    float divisor =  d_in[ (by * BLOCK_SIZE + ty) * STRIDE + (bx * BLOCK_SIZE + tx) ];

    d_out[ (by * BLOCK_SIZE + ty) * STRIDE + (bx * BLOCK_SIZE + tx) ] =
     dividend/divisor;
    
}


void onDevice( float *h_in, float *h_out, int ARRAY_BYTES){


    // declare GPU memory pointers
    float *d_in;
    float *d_out;

    // allocate the memory on the GPU
    cudaMalloc( (void**)&d_in, ARRAY_BYTES) ;
    cudaMalloc( (void**)&d_out, ARRAY_BYTES) ;

    // copy data from CPU the GPU
    cudaMemcpy( d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice );
    HANDLER_ERROR_MSG("memory H2D d_in");

    //execution configuration
    dim3 ThreadsBlocks( 3, 3 );
    dim3 GridBlocks( 2 , 2); 
 

    //run the kernel
    matrixKernel<<< GridBlocks, ThreadsBlocks >>>( d_in, d_out );
    HANDLER_ERROR_MSG("kernel panic!!!");
    cudaThreadSynchronize();

    // copy data back from the GPU to the CPU
    cudaMemcpy( h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost );
    HANDLER_ERROR_MSG("memory H2D d_out");

    // free GPU memory
    cudaFree( d_in );
    cudaFree( d_out );

}

void test(float *h_out, int ARRAY_SIZE){

    // verify that the GPU did the requested work
    for (int i=0, x=0; i<ARRAY_SIZE; i++) {
            printf( " %f ",h_out[i] );
            if(x==STRIDE-1){
                printf("\n");
                x=0;
            }else{
                x++;
            }
    }

    printf("-: successful execution :-\n");

}

void onHost(){

    const int ARRAY_SIZE = 36;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    float h_in[] ={1.0, 2.0, 3.0,    10.0, 11.0, 12.0,
                4.0, 5.0, 6.0,    13.0, 14.0, 15.0,
                7.0, 8.0, 9.0,    16.0, 17.0, 18.0,
                19.0, 20.0, 21.0, 28.0, 29.0, 30.0,
                22.0, 23.0, 24.0, 31.0, 32.0, 33.0,
                25.0, 26.0, 27.0, 34.0, 35.0, 36.0 };

    //CPU memory pointer
    float *h_out;

    //allocate memory on the host 
    h_out = (float*)malloc( ARRAY_BYTES );

    for(int i = 0; i < ARRAY_SIZE; i++){
        h_out[i] = 0;
    }

    // call device configuration
    onDevice( h_in, h_out, ARRAY_BYTES );
    test(h_out, ARRAY_SIZE);


    // free memory on the CPU
    free( h_out );
}


int main( void ) { 
 
    onHost();

    return 0;
}
