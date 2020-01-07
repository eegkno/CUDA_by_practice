#include <pthread.h>
#include <stdio.h>
#include <unistd.h>  //sleep

void* hello(void* arg) {
    int* ptrToThreadNumber = (int*)arg;
    sleep(5);
    printf("HelloThread %d\n", *ptrToThreadNumber);
    return 0;
}

int main(void) {
    pthread_t tid;
    int threadNum = 1;

    // pthread_create creates a new thread and makes it executable.
    // This routine can be called any number of times from anywhere within your
    // code.

    pthread_create(&tid, NULL, hello, &threadNum);

    //"Joining" is one way to accomplish synchronization between threads
    // The pthread_join() subroutine blocks the calling thread until the
    // specified threadid thread terminates.
    pthread_join(tid, NULL);
    printf("------------------------\n");
    return (0);
}
