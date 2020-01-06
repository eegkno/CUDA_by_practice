#ifndef GPU_MATRIX_H__
#define GPU_MATRIX_H__
#include <iostream>
using namespace std;

template <typename T>
struct Matrix {
    int width;
    int height;
    T* elements;

    __device__ __host__ T getElement(int row, int col) {
        return elements[row * width + col];
    }

    __device__ __host__ void setElement(int row, int col, T value) {
        elements[row * width + col] = value;
    }

    __host__ void print() {
        int i, j;
        cout << "\n" << endl;
        for (i = 0; i < width; i++) {
            for (j = 0; j < height; j++) {
                cout << getElement(j, i) << " ";
            }
            cout << "\n" << endl;
        }
    }
};

#endif /* GPU_MATRIX_H__ */
