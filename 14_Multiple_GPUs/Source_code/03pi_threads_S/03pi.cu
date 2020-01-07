#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#define THREADS 4

int intervalsT = 100000000;
double store, base;
double partialStore[] = {0.0, 0.0, 0.0, 0.0};

void* threadRoutine(void* param) {
    int i;
    int* threadId = (int*)param;
    int partialInterval = intervalsT / THREADS;
    double height;
    double x;

    for (i = (*threadId) * partialInterval, partialStore[*threadId] = 0.0;
         i < (partialInterval * (*threadId + 1)); i++) {
        x = i * base;
        height = 4 / (1 + x * x);
        partialStore[*threadId] += base * height;
    }
    return 0;
}

void calculatePIHostMultiple() {
    int i;
    pthread_t threads[THREADS];
    int threadId[THREADS];

    for (i = 0; i < THREADS; i++)
        threadId[i] = i;

    base = (double)(1.0 / intervalsT);

    for (i = 0; i < THREADS; i++)
        pthread_create(&threads[i], NULL, threadRoutine, &threadId[i]);

    for (i = 0; i < THREADS; i++)
        pthread_join(threads[i], NULL);

    store = 0.0;
    for (i = 0; i < THREADS; i++)
        store += partialStore[i];

    printf("PI (multiple th) =%f\n", store);
}

void calculatePIHostSingle() {
    int i;
    double height, x;
    double store, base;

    int intervals = 100000000;
    base = (double)(1.0 / intervals);

    for (i = 0, store = 0.0, x = 0.0; i < intervals; i++) {
        x = i * base;
        height = 4 / (1 + x * x);
        store += base * height;
    }

    printf("PI (single th) =%f \n", store);
}

int main() {
    calculatePIHostSingle();

    calculatePIHostMultiple();
}
