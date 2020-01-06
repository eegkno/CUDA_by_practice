#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include "common/CpuTimer.h"
#include "common/Vector.h"

#define THREADS 2
#define N (32)

const int ARRAY_BYTES = N * sizeof(float);
const int ARRAY_BYTES_H = N / THREADS * sizeof(float);

struct DataStruct {
    int threadID;
    int size;
    float* a;
    float* b;
    float* out;
};

void test(Vector<float> h_sout, Vector<float> h_mout) {
    for (int i = 0; i < N; i++) {
        assert(h_sout.elements[i] == h_mout.elements[i]);
    }
}

void* threadRoutine(void* param) {
    DataStruct* data = (DataStruct*)param;

    printf("ThreadID = %d \n", data->threadID);

    for (int i = 0; i < data->size; i++) {
        data->out[i] = data->a[i] + data->b[i];
    }

    return 0;
}

void addMutiple(Vector<float> h_a, Vector<float> h_b, Vector<float> h_mout) {
    // prepare for multithread
    DataStruct data[THREADS];
    data[0].threadID = 0;
    data[0].size = N / THREADS;
    data[0].a = h_a.elements;
    data[0].b = h_b.elements;
    data[0].out = (float*)malloc(ARRAY_BYTES_H);

    data[1].threadID = 1;
    data[1].size = N / THREADS;
    data[1].a = h_a.elements + N / THREADS;
    data[1].b = h_b.elements + N / THREADS;
    data[1].out = (float*)malloc(ARRAY_BYTES_H);

    for (int i = 0; i < N / THREADS; i++) {
        data[0].out[i] = 0;
        data[1].out[i] = 0;
    }

    pthread_t threads[THREADS];

    for (int i = 0; i < THREADS; i++)
        pthread_create(&threads[i], NULL, threadRoutine, &(data[i]));

    for (int i = 0; i < THREADS; i++)
        pthread_join(threads[i], NULL);

    for (int i = 0; i < data[0].size; i++)
        h_mout.elements[i] = data[0].out[i];

    for (int i = N / THREADS, j = 0; i < data[1].size * 2; i++, j++)
        h_mout.elements[i] = data[1].out[j];
}

void addSingle(Vector<float> h_a, Vector<float> h_b, Vector<float> h_sout) {
    for (int i = 0; i < N; i++) {
        h_sout.setElement(i, h_a.getElement(i) + h_b.getElement(i));
    }
}

void run() {
    Vector<float> h_a, h_b, h_sout, h_mout;
    h_a.length = N;
    h_b.length = N;
    h_sout.length = N;
    h_mout.length = N;

    h_a.elements = (float*)malloc(ARRAY_BYTES);
    h_b.elements = (float*)malloc(ARRAY_BYTES);
    h_sout.elements = (float*)malloc(ARRAY_BYTES);
    h_mout.elements = (float*)malloc(ARRAY_BYTES);

    for (int i = 0; i < N; i++) {
        h_a.elements[i] = i;
        h_b.elements[i] = i;
        h_sout.elements[i] = 0;
        h_mout.elements[i] = 0;
    }

    addSingle(h_a, h_b, h_sout);
    addMutiple(h_a, h_b, h_mout);

    test(h_sout, h_mout);
    printf("-: successful execution :-\n");
}

int main(void) {
    run();
    return 0;
}
