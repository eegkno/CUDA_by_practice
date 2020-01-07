#ifndef GPU_MATRIX_H__
#define GPU_MATRIX_H__

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
};

#endif /* GPU_MATRIX_H__ */
