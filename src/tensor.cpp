#include <iostream>
#include "tensor.hpp"

int Tensor::total_size() const {
    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    return size;
}

float& Tensor::at(std::initializer_list<int> indices) {
    assert(indices.size() == ndim);
    int arr[8];
    int i = 0;
    for (const int& val : indices) {
        arr[i++] = val;
    }
    int index = 0;
    int multiple = 1;
    for (int i = 0; i < indices.size(); i++) {
        assert(arr[i] < shape[i] && arr[i] >= 0);
        for (int j = i + 1; j < indices.size(); j++) {
            multiple *= shape[j];
        }
        index += arr[i] * multiple;
        multiple = 1;
    }
    return data[index];
}

void Tensor::print() const {
    std::cout << "Shape: [";
    for (int i = 0; i < ndim; i++) {
        std::cout << shape[i];
        if (i < ndim - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;

    for (int i = 0; i < total_size(); i++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}

Tensor Tensor::matmul(const Tensor& other) {
    assert(ndim == other.ndim);
    assert(shape[ndim - 1] == other.shape[ndim - 2]);
    for (int i = 0; i < ndim - 2; i++) {
        assert(shape[i] == other.shape[i]);
    }

    int out_shape[8];
    for (int i = 0; i < ndim - 2; i++) {
        out_shape[i] = shape[i];
    }
    out_shape[ndim-2] = shape[ndim-2];
    out_shape[ndim-1] = other.shape[ndim-1];
    Tensor C(out_shape, ndim);

    // compute strides for A, B, C
    int stride_A[8], stride_B[8], stride_C[8];
    stride_A[ndim-1] = 1;
    stride_B[ndim-1] = 1;
    stride_C[ndim-1] = 1;
    for (int i = ndim-2; i >= 0; i--) {
        stride_A[i] = stride_A[i+1] * shape[i+1];
        stride_B[i] = stride_B[i+1] * other.shape[i+1];
        stride_C[i] = stride_C[i+1] * C.shape[i+1];
    }

    // compute total batch size
    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) {
        batch_size *= shape[i];
    }

    int M = shape[ndim-2];
    int K = shape[ndim-1];
    int N = other.shape[ndim-1];

    // iterate over all batch index combinations
    for (int b = 0; b < batch_size; b++) {
        // compute batch offsets for A, B, C
        int offset_A = 0, offset_B = 0, offset_C = 0;
        int tmp = b;
        for (int d = ndim-3; d >= 0; d--) {
            int idx = tmp % shape[d];
            tmp /= shape[d];
            offset_A += idx * stride_A[d];
            offset_B += idx * stride_B[d];
            offset_C += idx * stride_C[d];
        }

        // matmul for this batch
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += data[offset_A + i * stride_A[ndim-2] + k * stride_A[ndim-1]] *
                           other.data[offset_B + k * stride_B[ndim-2] + j * stride_B[ndim-1]];
                }
                C.data[offset_C + i * stride_C[ndim-2] + j * stride_C[ndim-1]] = sum;
            }
        }
    }
    return C;
}