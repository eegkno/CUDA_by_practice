// POSIX threads.
#include <pthread.h>
#include <stdio.h>

typedef pthread_t CUTThread;
typedef void* (*CUT_THREADROUTINE)(void*);

#define CUT_THREADPROC void
#define CUT_THREADEND

// Create thread.
CUTThread start_thread(CUT_THREADROUTINE, void* data);

// Wait for thread to finish.
void end_thread(CUTThread thread);

// Destroy thread.
void destroy_thread(CUTThread thread);

// Wait for multiple threads.
void wait_for_threads(const CUTThread* threads, int num);

// Create thread
CUTThread start_thread(CUT_THREADROUTINE func, void* data) {
    pthread_t thread;
    pthread_create(&thread, NULL, func, data);
    return thread;
}

// Wait for thread to finish
void end_thread(CUTThread thread) {
    pthread_join(thread, NULL);
}

// Destroy thread
void destroy_thread(CUTThread thread) {
    pthread_cancel(thread);
}

// Wait for multiple threads
void wait_for_threads(const CUTThread* threads, int num) {
    for (int i = 0; i < num; i++)
        end_thread(threads[i]);
}
