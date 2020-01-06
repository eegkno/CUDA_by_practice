#ifndef PTHREAD_TIMER_H__
#define PTHREAD_TIMER_H__

// POSIX threads.
#include <pthread.h>
#include <stdio.h>

typedef pthread_t CUTThread;
typedef void* (*CUT_THREADROUTINE)(void*);

struct CudaPthread {
    pthread_t thread;

    CudaPthread() {}

    ~CudaPthread() {}

    // Create thread
    CUTThread start_thread(CUT_THREADROUTINE func, void* data) {
        pthread_create(&thread, NULL, func, data);
        return thread;
    }

    // Wait for thread to finish
    void end_thread() { pthread_join(thread, NULL); }

    // Destroy thread
    void destroy_thread() { pthread_cancel(thread); }
};

#endif /* PTHREAD_TIMER_H__ */
