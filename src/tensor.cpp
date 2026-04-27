#include <iostream>
#include <cmath>
#include "tensor.hpp"

int Tensor::total_size() const {
    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    return size;
}

void Tensor::init_strides() {
    if (ndim == 0) return;
    strides[ndim-1] = 1;
    for (int i = ndim-2; i >= 0; i--) {
        strides[i] = strides[i+1] * shape[i+1];
    }
}

void Tensor::compute_strides(int* out_strides) const {
    std::memcpy(out_strides, strides, sizeof(strides));
}

int Tensor::compute_offset(int b, const int* strides, int skip_axis) const {
    int offset = 0;
    int tmp = b;
    for (int d = ndim-1; d >= 0; d--) {
        if (d == skip_axis) continue;
        int idx = tmp % shape[d];
        tmp /= shape[d];
        offset += idx * strides[d];
    }
    return offset;
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

Tensor Tensor::matmul(const Tensor& other) const {
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
    compute_strides(stride_A);
    other.compute_strides(stride_B);
    C.compute_strides(stride_C);

    // compute total batch size
    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) {
        batch_size *= shape[i];
    }

    int M = shape[ndim-2];
    int K = shape[ndim-1];
    int N = other.shape[ndim-1];

    for (int b = 0; b < batch_size; b++) {
        int offset_A = 0, offset_B = 0, offset_C = 0;
        int tmp = b;
        for (int d = ndim-3; d >= 0; d--) {
            int idx = tmp % shape[d];
            tmp /= shape[d];
            offset_A += idx * stride_A[d];
            offset_B += idx * stride_B[d];
            offset_C += idx * stride_C[d];
        }
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

Tensor Tensor::softmax(int axis) const {
    // compute total batch size
    int batch_size = 1;
    for (int i = 0; i < ndim; i++) {
        if (i == axis) continue;
        batch_size *= shape[i];
    }

    // compute strides
    int stride[8];
    compute_strides(stride);

    Tensor result(shape, ndim);
    int dim_size = shape[axis];
    int axis_stride = stride[axis];

    for (int b = 0; b < batch_size; b++) {
        int offset = compute_offset(b, stride, axis);

        // find max for numerical stability
        float max_val = data[offset];
        for (int i = 1; i < dim_size; i++) {
            float val = data[offset + i * axis_stride];
            if (val > max_val) {
                max_val = val;
            }
        }

        float sum_exp = 0.0f;
        for (int i = 0; i < dim_size; i++) {
            float val = data[offset + i * axis_stride];
            float exp_val = std::exp(val - max_val);
            result.data[offset + i * axis_stride] = exp_val;
            sum_exp += exp_val;
        }

        for (int i = 0; i < dim_size; i++) {
            result.data[offset + i * axis_stride] /= sum_exp;
        }
    }
    return result;
}

void Tensor::print() const {
    if (name[0] != '\0') std::cout << "Name: " << name << "\n";
    std::cout << "Shape: [";
    for (int i = 0; i < ndim; i++) {
        std::cout << shape[i];
        if (i < ndim - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    for (int i = 0; i < total_size(); i++) std::cout << data[i] << " ";
    std::cout << std::endl;
}