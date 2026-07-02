#include <iostream>
#include <cmath>
#include <thread>
#include <vector>

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

Tensor Tensor::zeros(std::initializer_list<int> shape) {
    Tensor out(shape);
    return out;
}

Tensor Tensor::ones(std::initializer_list<int> shape) {
    Tensor out(shape);
    for (size_t i = 0; i < out.numel(); i++) {
        out.data[i] = 1;
    }
    return out;
}

Tensor Tensor::full(std::initializer_list<int> shape, float value) {
    Tensor out(shape);
    for (size_t i = 0; i < out.numel(); i++) {
        out.data[i] = value;
    }
    return out;
}

Tensor Tensor::from_data(std::initializer_list<int> shape, std::initializer_list<float> values) {
    Tensor out(shape);
    assert(out.numel() == values.size());
    size_t i = 0;
    for (float val : values) {
        out.data[i++] = val;
    }
    return out;
}

float& Tensor::at(std::initializer_list<int> indices) {
    assert(indices.size() == static_cast<size_t>(ndim));
    assert(data != nullptr);

    int flat_index = 0;
    size_t axis = 0;

    for (int idx : indices) {
        assert(idx >= 0 && idx < shape[axis]);
        flat_index += strides[axis++] * idx;
    }

    return data[flat_index];
}

void threaded_matmul_worker(const Tensor& A, const Tensor& B, Tensor& C, int start, int end) {
    int ndim = A.ndim;

    // strides for A, B, C
    const int* stride_A = A.strides;
    const int* stride_B = B.strides;
    const int* stride_C = C.strides;

    int I = A.shape[ndim - 2];
    int J = B.shape[ndim - 1];
    int K = A.shape[ndim - 1];

    for (int task = start; task < end; task++) {
        int offset_A = 0, offset_B = 0, offset_C = 0;
        int temp = task / I;
        int row = task % I;
        for (int d = ndim - 3; d >= 0; d--) {
            int index = temp % C.shape[d];
            temp /= C.shape[d];
            offset_C += index * stride_C[d];

            bool A_broadcasted = A.shape[d] == 1 && C.shape[d] != 1;
            bool B_broadcasted = B.shape[d] == 1 && C.shape[d] != 1;

            if (!A_broadcasted) {
                offset_A += index * stride_A[d];
            }
            if (!B_broadcasted) {
                offset_B += index * stride_B[d];
            }
        }

        const float* a_row = A.data + offset_A + row * stride_A[ndim - 2];
        float* c_row = C.data + offset_C + row * stride_C[ndim - 2];
        for (int k = 0; k < K; k++) {
            float val_A = a_row[k * stride_A[ndim - 1]];
            const float* b_row = B.data + offset_B + k * stride_B[ndim - 2];
            for (int j = 0; j < J; j++) {
                c_row[j * stride_C[ndim - 1]] += val_A * b_row[j * stride_B[ndim - 1]];
            }
        }

    }
}

Tensor Tensor::matmul(const Tensor& other, int thread_count) const {
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

    // compute total batch size
    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) {
        batch_size *= output_shape[i];
    }

    int I = shape[ndim - 2];
    // int J = other.shape[ndim - 1];
    // int K = shape[ndim - 1];

    int total_rows = batch_size * I;

    if (thread_count == 0) {
        thread_count = (static_cast<int>(std::thread::hardware_concurrency()) == 0) ? 1 : static_cast<int>(static_cast<int>(std::thread::hardware_concurrency()));
    }

    assert(thread_count > 0);

    if (thread_count > total_rows) {
        thread_count = total_rows;
    }

    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    int rows_per_thread = total_rows / thread_count;
    int remainder = total_rows % thread_count;

    for (int i = 0; i < remainder; i++) {
        threads.emplace_back(threaded_matmul_worker, std::cref(*this), std::cref(other), std::ref(C), (rows_per_thread * i) + i, (rows_per_thread * (i + 1)) + (i + 1));
    }

    for (int i = remainder; i < thread_count; i++) {
        threads.emplace_back(threaded_matmul_worker, std::cref(*this), std::cref(other), std::ref(C), rows_per_thread * i + remainder, rows_per_thread * (i + 1) + remainder);
    }

    for (std::thread& thread : threads) {
        thread.join();
    }

    return C;
}

Tensor Tensor::matmul_naive(const Tensor& other) const {
    assert(ndim == other.ndim);
    assert(ndim >= 2);
    assert(shape[ndim - 1] == other.shape[ndim - 2]);
    assert(data != nullptr && other.data != nullptr);

    if (ndim == 2 && is_contiguous() && other.is_contiguous()) {
        // 2d matmul fast path
        int output_shape[8] = {0};

        int I = shape[0];
        int J = other.shape[1];
        int K = shape[1];

        output_shape[0] = I;
        output_shape[1] = J;
        Tensor C(output_shape, 2);

        for (int i = 0; i < I; i++) {
            const float* a_row = data + i * K;
            float* c_row = C.data + i * J;
            for (int k = 0; k < K; k++) {
                float val_A = a_row[k];
                const float* b_row = other.data + k * J;
                for (int j = 0; j < J; j++) {
                    c_row[j] += val_A * b_row[j];
                }
            }
        }

        return C;
    }

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

Tensor Tensor::matmul_2d_blocked(const Tensor& other, int block_size) const {
    assert(ndim == other.ndim);
    assert(ndim == 2);
    assert(shape[ndim - 1] == other.shape[ndim - 2]);
    assert(data != nullptr && other.data != nullptr);
    assert(is_contiguous() && other.is_contiguous());
    assert(block_size > 0);

    int output_shape[8] = {0};

    int I = shape[0];
    int J = other.shape[1];
    int K = shape[1];

    int BI = block_size;
    int BJ = block_size;
    int BK = block_size;

    assert(I % BI == 0);
    assert(J % BJ == 0);
    assert(K % BK == 0);

    output_shape[0] = I;
    output_shape[1] = J;
    Tensor C(output_shape, 2);

    for (int ii = 0; ii < I; ii += BI) {
        for (int jj = 0; jj < J; jj += BJ) {
            for (int kk = 0; kk < K; kk += BK) {

                for (int i = ii; i < ii + BI; i++) {
                    const float* a_row = data + i * K;
                    int row_C = i * J;
                    for (int k = kk; k < kk + BK; k++) {
                        float val_A = a_row[k];
                        float* c_row = C.data + row_C;
                        const float* b_row = other.data + k * J;
                        for (int j = jj; j < jj + BJ; j++) {
                            c_row[j] += val_A * b_row[j];
                        }
                    }
                }

            }
        }
    }

    return C;
}

void matmul_worker(const Tensor& A, const Tensor& B, Tensor& C, int start, int end) {
    int J = B.shape[1];
    int K = A.shape[1];
    
    for (int i = start; i < end; i++) {
        const float* a_row = A.data + i * K;
        float* c_row = C.data + i * J;
        for (int k = 0; k < K; k++) {
            float val_A = a_row[k];
            const float* b_row = B.data + k * J;
            for (int j = 0; j < J; j++) {
                c_row[j] += val_A * b_row[j];
            }
        }
    }
}

Tensor Tensor::matmul_2d_threaded(const Tensor& other, int thread_count) const {
    assert(ndim == 2);
    assert(other.ndim == 2);
    assert(is_contiguous());
    assert(other.is_contiguous());
    assert(shape[ndim - 1] == other.shape[ndim - 2]);
    assert(data != nullptr && other.data != nullptr);

    int I = shape[0];
    int J = other.shape[1];

    if (thread_count == 0) {
        thread_count = (static_cast<int>(std::thread::hardware_concurrency()) == 0) ? 1 : static_cast<int>(static_cast<int>(std::thread::hardware_concurrency()));
    }

    assert(thread_count > 0);

    if (thread_count > I) {
        thread_count = I;
    }

    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    int output_shape[8] = {0};
    output_shape[0] = I;
    output_shape[1] = J;
    Tensor C(output_shape, 2);

    int rows_per_thread = I / thread_count;
    int remainder = I % thread_count;

    for (int i = 0; i < remainder; i++) {
        threads.emplace_back(matmul_worker, std::cref(*this), std::cref(other), std::ref(C), (rows_per_thread * i) + i, (rows_per_thread * (i + 1)) + (i + 1));
    }

    for (int i = remainder; i < thread_count; i++) {
        threads.emplace_back(matmul_worker, std::cref(*this), std::cref(other), std::ref(C), rows_per_thread * i + remainder, rows_per_thread * (i + 1) + remainder);
    }

    for (std::thread& thread : threads) {
        thread.join();
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