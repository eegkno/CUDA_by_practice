
#ifndef __LOCK_H__
#define __LOCK_H__

struct Lock {
    int* mutex;

    Lock(void) {
        int state = 0;
        HANDLER_ERROR_ERR(cudaMalloc((void**)&mutex, sizeof(int)));
        HANDLER_ERROR_ERR(
            cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice));
    }

    ~Lock(void) { cudaFree(mutex); }

    __device__ void lock(void) {
        // every thread is waiting to get the mutex, once a thread won the
        // mutex, it changes the value to 1
        while (atomicCAS(mutex, 0, 1) != 0)
            ;
    }

    // When the thread end its work, it release the mutex. Now it's ready to be
    // taken.
    __device__ void unlock(void) { atomicExch(mutex, 0); }
};

#endif
