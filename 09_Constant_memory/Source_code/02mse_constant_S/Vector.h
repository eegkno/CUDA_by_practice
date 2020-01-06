#ifndef GPU_VECTOR_H__
#define GPU_VECTOR_H__

#include <iostream>
using namespace std;

template <typename T>
struct Vector {
    int length;
    T* elements;

    __device__ __host__ T getElement(int index) { return elements[index]; }

    __device__ __host__ void setElement(int index, T value) {
        elements[index] = value;
    }

    __host__ void randomInit(int linf, int lsup) {
        // srand(time(0));
        for (int i = 0; i < length; i++) {
            elements[i] = rand() % lsup + linf;
            // elements[i] =rand();
        }
    }

    __host__ void onesInit() {
        for (int i = 0; i < length; i++)
            elements[i] = 1;
    }

    __host__ void zerosInit() {
        for (int i = 0; i < length; i++)
            elements[i] = 0;
    }

    __host__ void valueInit(int value) {
        for (int i = 0; i < length; i++)
            elements[i] = value;
    }

    __host__ void print() {
        for (int i = 0; i < length; i++)
            cout << elements[i] << endl;
        cout << endl;
    }
};

#endif /* GPU_VECTOR_H__ */
