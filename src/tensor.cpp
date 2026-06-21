#include <iostream>
#include <cmath>
#include "tensor.hpp"

size_t Tensor::numel() const {
    if (ndim == 0) return 0;
    size_t size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= static_cast<size_t>(shape[i]);
    }
    return size;
}

int Tensor::dim() const {
    return ndim;
}

int Tensor::size(int axis) const {
    assert(axis >= 0 && axis < ndim);
    return shape[axis];
}

int Tensor::stride(int axis) const {
    assert(axis >= 0 && axis < ndim);
    return strides[axis];
}

bool Tensor::is_contiguous() const {
    if (ndim == 0) return true;
    int expected_stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        if (expected_stride != stride(i)) return false;
        expected_stride *= size(i);
    }
    return true;
}

void Tensor::init_contiguous_strides() {
    if (ndim == 0) return;
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

int Tensor::compute_offset(int b, const int* strides, int skip_axis) const {
    assert(skip_axis == -1 || (skip_axis >= 0 && skip_axis < ndim));
    int offset = 0;
    int tmp = b;
    for (int d = ndim - 1; d >= 0; d--) {
        if (d == skip_axis) continue;
        int idx = tmp % shape[d];
        tmp /= shape[d];
        offset += idx * strides[d];
    }
    return offset;
}

float* Tensor::data_ptr() {
    return data;
}

const float* Tensor::data_ptr() const {
    return data;
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
    assert(ndim >= 2);
    assert(shape[ndim - 1] == other.shape[ndim - 2]);
    assert(data != nullptr && other.data != nullptr);
    bool broadcast = false;
    for (int i = 0; i < ndim - 2; i++) {
        bool compatible = shape[i] != other.shape[i] && (shape[i] == 1 || other.shape[i] == 1);
        if (compatible) broadcast = true;
        assert(shape[i] == other.shape[i] || compatible);
    }

    int output_shape[8] = {0};
    if (!broadcast) {
        for (int i = 0; i < ndim - 2; i++) {
            output_shape[i] = shape[i];
        }
    } else {
        for (int i = 0; i < ndim - 2; i++) {
            if (shape[i] != other.shape[i]) {
                output_shape[i] = (other.shape[i] == 1) ? shape[i] : other.shape[i];
                continue;
            }
            output_shape[i] = shape[i];
        }
    }

    output_shape[ndim - 2] = shape[ndim - 2];
    output_shape[ndim - 1] = other.shape[ndim - 1];
    Tensor C(output_shape, ndim);

    // strides for A, B, C
    const int* stride_A = strides;
    const int* stride_B = other.strides;
    const int* stride_C = C.strides;

    // compute total batch size
    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) {
        batch_size *= output_shape[i];
    }

    int M = shape[ndim - 2];
    int K = shape[ndim - 1];
    int N = other.shape[ndim - 1];

    for (int b = 0; b < batch_size; b++) {
        int offset_A = 0, offset_B = 0, offset_C = 0;
        int temp = b;
        for (int d = ndim - 3; d >= 0; d--) {
            int index = temp % output_shape[d];
            temp /= output_shape[d];
            offset_C += index * stride_C[d];

            bool A_broadcasted = shape[d] == 1 && output_shape[d] != 1;
            bool B_broadcasted = other.shape[d] == 1 && output_shape[d] != 1;

            if (!A_broadcasted) {
                offset_A += index * stride_A[d];
            }
            if (!B_broadcasted) {
                offset_B += index * stride_B[d];
            }
        }
        int row_A, row_C, row_B;
        float val_A;
        for (int i = 0; i < M; i++) {
            row_A = offset_A + i * stride_A[ndim - 2];
            row_C = offset_C + i * stride_C[ndim - 2];
            for (int k = 0; k < K; k++) {
                val_A = data[row_A + k * stride_A[ndim - 1]];
                row_B = offset_B + k * stride_B[ndim - 2];
                for (int j = 0; j < N; j++) {
                    C.data[row_C + j * stride_C[ndim - 1]] += val_A * other.data[row_B + j * stride_B[ndim - 1]];
                }
            }
        }
    }
    return C;
}

Tensor Tensor::softmax(int axis) const {
    assert(axis >= 0 && axis < ndim);
    assert(data != nullptr);

    // compute total batch size
    int batch_size = 1;
    for (int i = 0; i < ndim; i++) {
        if (i == axis) continue;
        batch_size *= shape[i];
    }

    Tensor out(*this);
    int dim_size = shape[axis];
    int axis_stride = strides[axis];

    for (int b = 0; b < batch_size; b++) {
        int offset = compute_offset(b, strides, axis);

        // find max for numerical stability
        float max = data[offset];
        for (int i = 1; i < dim_size; i++) {
            float curr = data[offset + i * axis_stride];
            max = (curr > max) ? curr : max;
        }

        // find sum of exponentials
        float sum = 0.0;
        for (int i = 0; i < dim_size; i++) {
            sum += std::exp(data[offset + i * axis_stride] - max);
        }

        for (int i = 0; i < dim_size; i++) {
            out.data[offset + i * axis_stride] = (std::exp(data[offset + i * axis_stride] - max)) / sum;
        }
    }
    return out;
}

void Tensor::print() const {
    if (name[0] != '\0') std::cout << "Name: " << name << "\n";

    std::cout << "Shape: [";
    for (int i = 0; i < ndim; i++) {
        std::cout << shape[i];
        if (i < ndim - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Strides: [";
    for (int i = 0; i < ndim; i++) {
        std::cout << strides[i];
        if (i < ndim - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Data: ";
    for (size_t i = 0; i < numel(); i++) std::cout << data[i] << " ";
    std::cout << std::endl;
}

void Tensor::print_shape() const {
    if (name[0] != '\0') std::cout << "Name: " << name << "\n";

    std::cout << "Shape: [";
    for (int i = 0; i < ndim; i++) {
        std::cout << shape[i];
        if (i < ndim - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Strides: [";
    for (int i = 0; i < ndim; i++) {
        std::cout << strides[i];
        if (i < ndim - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}