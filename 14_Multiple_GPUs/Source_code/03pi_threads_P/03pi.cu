#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#define THREADS 4

int intervalsT = 100000000;
double partialStore[] = {0.0, 0.0, 0.0, 0.0};
// -:YOUR CODE HERE:-

void* threadRoutine(void* param) {
    // -:YOUR CODE HERE:-
    return 0;
}

void calculatePIHostMultiple() {
    // -:YOUR CODE HERE:-
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
