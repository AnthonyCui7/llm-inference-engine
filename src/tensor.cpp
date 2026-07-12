#include <iostream>
#include <cmath>
#include <thread>
#include <vector>

#include "tensor.hpp"
#include "thread_pool.hpp"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#elif defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

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

int Tensor::compute_offset(int b, int skip_axis) const {
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

Tensor Tensor::add(float scalar) const {
    assert(data != nullptr);

    Tensor out = *this;

    for (size_t i = 0; i < numel(); i++) {
        out.data[i] += scalar;
    }

    return out;
}

Tensor Tensor::mul(float scalar) const {
    assert(data != nullptr);

    Tensor out = *this;

    for (size_t i = 0; i < numel(); i++) {
        out.data[i] *= scalar;
    }

    return out;
}

Tensor Tensor::sub(float scalar) const {
    assert(data != nullptr);

    Tensor out = *this;

    for (size_t i = 0; i < numel(); i++) {
        out.data[i] -= scalar;
    }

    return out;
}

Tensor Tensor::div(float scalar) const {
    assert(data != nullptr);

    Tensor out = *this;

    for (size_t i = 0; i < numel(); i++) {
        out.data[i] /= scalar;
    }

    return out;
}

Tensor Tensor::add(const Tensor& other) const {
    assert(ndim == other.ndim);
    assert(data != nullptr);
    assert(other.data != nullptr);

    bool broadcast = false;
    for (int i = 0; i < ndim; i++) {
        bool compatible = shape[i] != other.shape[i] && (shape[i] == 1 || other.shape[i] == 1);
        if (compatible) broadcast = true;
        assert(shape[i] == other.shape[i] || compatible);
    }

    int output_shape[8] = {0};
    if (!broadcast) {
        for (int i = 0; i < ndim; i++) {
            output_shape[i] = shape[i];
        }
    } else {
        for (int i = 0; i < ndim; i++) {
            if (shape[i] != other.shape[i]) {
                output_shape[i] = (other.shape[i] == 1) ? shape[i] : other.shape[i];
                continue;
            }
            output_shape[i] = shape[i];
        }
    }

    Tensor C(output_shape, ndim);

    // strides for A, B, C
    const int* stride_A = strides;
    const int* stride_B = other.strides;
    const int* stride_C = C.strides;

    if (!broadcast && is_contiguous() && other.is_contiguous()) {
        for (size_t i = 0; i < numel(); i++) {
            C.data[i] = data[i] + other.data[i];
        }
    } else {
        int total_elements = static_cast<int>(C.numel());

        for (int elem = 0; elem < total_elements; elem++) {
            int offset_A = 0, offset_B = 0, offset_C = 0;
            int temp = elem;

            for (int d = ndim - 1; d >= 0; d--) {
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

            C.data[offset_C] = data[offset_A] + other.data[offset_B];
        }
    }

    return C;
}

Tensor Tensor::mul(const Tensor& other) const {
    assert(ndim == other.ndim);
    assert(data != nullptr);
    assert(other.data != nullptr);

    bool broadcast = false;
    for (int i = 0; i < ndim; i++) {
        bool compatible = shape[i] != other.shape[i] && (shape[i] == 1 || other.shape[i] == 1);
        if (compatible) broadcast = true;
        assert(shape[i] == other.shape[i] || compatible);
    }

    int output_shape[8] = {0};
    if (!broadcast) {
        for (int i = 0; i < ndim; i++) {
            output_shape[i] = shape[i];
        }
    } else {
        for (int i = 0; i < ndim; i++) {
            if (shape[i] != other.shape[i]) {
                output_shape[i] = (other.shape[i] == 1) ? shape[i] : other.shape[i];
                continue;
            }
            output_shape[i] = shape[i];
        }
    }

    Tensor C(output_shape, ndim);

    // strides for A, B, C
    const int* stride_A = strides;
    const int* stride_B = other.strides;
    const int* stride_C = C.strides;

    if (!broadcast && is_contiguous() && other.is_contiguous()) {
        for (size_t i = 0; i < numel(); i++) {
            C.data[i] = data[i] * other.data[i];
        }
    } else {
        int total_elements = static_cast<int>(C.numel());

        for (int elem = 0; elem < total_elements; elem++) {
            int offset_A = 0, offset_B = 0, offset_C = 0;
            int temp = elem;

            for (int d = ndim - 1; d >= 0; d--) {
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

            C.data[offset_C] = data[offset_A] * other.data[offset_B];
        }
    }
    
    return C;
}

Tensor Tensor::sub(const Tensor& other) const {
    assert(ndim == other.ndim);
    assert(data != nullptr);
    assert(other.data != nullptr);

    bool broadcast = false;
    for (int i = 0; i < ndim; i++) {
        bool compatible = shape[i] != other.shape[i] && (shape[i] == 1 || other.shape[i] == 1);
        if (compatible) broadcast = true;
        assert(shape[i] == other.shape[i] || compatible);
    }

    int output_shape[8] = {0};
    if (!broadcast) {
        for (int i = 0; i < ndim; i++) {
            output_shape[i] = shape[i];
        }
    } else {
        for (int i = 0; i < ndim; i++) {
            if (shape[i] != other.shape[i]) {
                output_shape[i] = (other.shape[i] == 1) ? shape[i] : other.shape[i];
                continue;
            }
            output_shape[i] = shape[i];
        }
    }

    Tensor C(output_shape, ndim);

    // strides for A, B, C
    const int* stride_A = strides;
    const int* stride_B = other.strides;
    const int* stride_C = C.strides;

    if (!broadcast && is_contiguous() && other.is_contiguous()) {
        for (size_t i = 0; i < numel(); i++) {
            C.data[i] = data[i] - other.data[i];
        }
    } else {
        int total_elements = static_cast<int>(C.numel());

        for (int elem = 0; elem < total_elements; elem++) {
            int offset_A = 0, offset_B = 0, offset_C = 0;
            int temp = elem;

            for (int d = ndim - 1; d >= 0; d--) {
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

            C.data[offset_C] = data[offset_A] - other.data[offset_B];
        }
    }
    
    return C;
}

/*
SIMD kernel for one output row of a matmul: c[j] += sum over k of a[k] * b[k][j].
Requires contiguous B and C rows; A's column stride can be anything. Walks 4 rows of B
at a time, so B is read sequentially and the C row is only loaded/stored once per 4 k's.
Per-element accumulation stays in ascending k order, so output matches the scalar path.
NEON and AVX2 versions, scalar fallback elsewhere.
*/
static void matmul_row_kernel(const float* a_row, int a_stride,
                              const float* B, int b_row_stride,
                              float* c_row, int K, int J) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    int k = 0;
    for (; k + 4 <= K; k += 4) {
        float32x4_t a0 = vdupq_n_f32(a_row[k * a_stride]);
        float32x4_t a1 = vdupq_n_f32(a_row[(k + 1) * a_stride]);
        float32x4_t a2 = vdupq_n_f32(a_row[(k + 2) * a_stride]);
        float32x4_t a3 = vdupq_n_f32(a_row[(k + 3) * a_stride]);
        const float* b0 = B + k * b_row_stride;
        const float* b1 = b0 + b_row_stride;
        const float* b2 = b1 + b_row_stride;
        const float* b3 = b2 + b_row_stride;

        int j = 0;
        for (; j + 8 <= J; j += 8) {
            float32x4_t c0 = vld1q_f32(c_row + j);
            float32x4_t c1 = vld1q_f32(c_row + j + 4);
            c0 = vfmaq_f32(c0, a0, vld1q_f32(b0 + j));
            c1 = vfmaq_f32(c1, a0, vld1q_f32(b0 + j + 4));
            c0 = vfmaq_f32(c0, a1, vld1q_f32(b1 + j));
            c1 = vfmaq_f32(c1, a1, vld1q_f32(b1 + j + 4));
            c0 = vfmaq_f32(c0, a2, vld1q_f32(b2 + j));
            c1 = vfmaq_f32(c1, a2, vld1q_f32(b2 + j + 4));
            c0 = vfmaq_f32(c0, a3, vld1q_f32(b3 + j));
            c1 = vfmaq_f32(c1, a3, vld1q_f32(b3 + j + 4));
            vst1q_f32(c_row + j, c0);
            vst1q_f32(c_row + j + 4, c1);
        }
        for (; j + 4 <= J; j += 4) {
            float32x4_t c0 = vld1q_f32(c_row + j);
            c0 = vfmaq_f32(c0, a0, vld1q_f32(b0 + j));
            c0 = vfmaq_f32(c0, a1, vld1q_f32(b1 + j));
            c0 = vfmaq_f32(c0, a2, vld1q_f32(b2 + j));
            c0 = vfmaq_f32(c0, a3, vld1q_f32(b3 + j));
            vst1q_f32(c_row + j, c0);
        }
        for (; j < J; j++) {
            float sum = c_row[j];
            sum += a_row[k * a_stride] * b0[j];
            sum += a_row[(k + 1) * a_stride] * b1[j];
            sum += a_row[(k + 2) * a_stride] * b2[j];
            sum += a_row[(k + 3) * a_stride] * b3[j];
            c_row[j] = sum;
        }
    }
    for (; k < K; k++) {
        float32x4_t a0 = vdupq_n_f32(a_row[k * a_stride]);
        const float* b0 = B + k * b_row_stride;
        int j = 0;
        for (; j + 4 <= J; j += 4) {
            float32x4_t c0 = vld1q_f32(c_row + j);
            c0 = vfmaq_f32(c0, a0, vld1q_f32(b0 + j));
            vst1q_f32(c_row + j, c0);
        }
        for (; j < J; j++) {
            c_row[j] += a_row[k * a_stride] * b0[j];
        }
    }
#elif defined(__AVX2__) && defined(__FMA__)
    int k = 0;
    for (; k + 4 <= K; k += 4) {
        __m256 a0 = _mm256_set1_ps(a_row[k * a_stride]);
        __m256 a1 = _mm256_set1_ps(a_row[(k + 1) * a_stride]);
        __m256 a2 = _mm256_set1_ps(a_row[(k + 2) * a_stride]);
        __m256 a3 = _mm256_set1_ps(a_row[(k + 3) * a_stride]);
        const float* b0 = B + k * b_row_stride;
        const float* b1 = b0 + b_row_stride;
        const float* b2 = b1 + b_row_stride;
        const float* b3 = b2 + b_row_stride;

        int j = 0;
        for (; j + 16 <= J; j += 16) {
            __m256 c0 = _mm256_loadu_ps(c_row + j);
            __m256 c1 = _mm256_loadu_ps(c_row + j + 8);
            c0 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(b0 + j), c0);
            c1 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(b0 + j + 8), c1);
            c0 = _mm256_fmadd_ps(a1, _mm256_loadu_ps(b1 + j), c0);
            c1 = _mm256_fmadd_ps(a1, _mm256_loadu_ps(b1 + j + 8), c1);
            c0 = _mm256_fmadd_ps(a2, _mm256_loadu_ps(b2 + j), c0);
            c1 = _mm256_fmadd_ps(a2, _mm256_loadu_ps(b2 + j + 8), c1);
            c0 = _mm256_fmadd_ps(a3, _mm256_loadu_ps(b3 + j), c0);
            c1 = _mm256_fmadd_ps(a3, _mm256_loadu_ps(b3 + j + 8), c1);
            _mm256_storeu_ps(c_row + j, c0);
            _mm256_storeu_ps(c_row + j + 8, c1);
        }
        for (; j + 8 <= J; j += 8) {
            __m256 c0 = _mm256_loadu_ps(c_row + j);
            c0 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(b0 + j), c0);
            c0 = _mm256_fmadd_ps(a1, _mm256_loadu_ps(b1 + j), c0);
            c0 = _mm256_fmadd_ps(a2, _mm256_loadu_ps(b2 + j), c0);
            c0 = _mm256_fmadd_ps(a3, _mm256_loadu_ps(b3 + j), c0);
            _mm256_storeu_ps(c_row + j, c0);
        }
        for (; j < J; j++) {
            float sum = c_row[j];
            sum += a_row[k * a_stride] * b0[j];
            sum += a_row[(k + 1) * a_stride] * b1[j];
            sum += a_row[(k + 2) * a_stride] * b2[j];
            sum += a_row[(k + 3) * a_stride] * b3[j];
            c_row[j] = sum;
        }
    }
    for (; k < K; k++) {
        __m256 a0 = _mm256_set1_ps(a_row[k * a_stride]);
        const float* b0 = B + k * b_row_stride;
        int j = 0;
        for (; j + 8 <= J; j += 8) {
            __m256 c0 = _mm256_loadu_ps(c_row + j);
            c0 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(b0 + j), c0);
            _mm256_storeu_ps(c_row + j, c0);
        }
        for (; j < J; j++) {
            c_row[j] += a_row[k * a_stride] * b0[j];
        }
    }
#else
    for (int k = 0; k < K; k++) {
        float val_A = a_row[k * a_stride];
        const float* b_row = B + k * b_row_stride;
        for (int j = 0; j < J; j++) {
            c_row[j] += val_A * b_row[j];
        }
    }
#endif
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

    // the SIMD row kernel needs contiguous B and C rows
    bool inner_contiguous = stride_B[ndim - 1] == 1 && stride_C[ndim - 1] == 1;

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

        if (inner_contiguous) {
            matmul_row_kernel(a_row, stride_A[ndim - 1], B.data + offset_B, stride_B[ndim - 2], c_row, K, J);
            continue;
        }

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

    // thread_count caps the parallelism; 0 uses every pool lane
    assert(thread_count >= 0);

    ThreadPool::instance().parallel_for(total_rows, thread_count, [this, &other, &C](int start, int end) {
        threaded_matmul_worker(*this, other, C, start, end);
    });

    return C;
}

Tensor Tensor::gelu() const {
    assert(data != nullptr);

    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;

    Tensor out = *this;

    for (size_t i = 0; i < numel(); i++) {
        float x = out.data[i];
        out.data[i] = 0.5f * x * (1.0f + std::tanh(SQRT_2_OVER_PI * (x + 0.044715f * x * x * x)));
    }

    return out;
}

Tensor Tensor::layernorm(float eps) const {
    assert(data != nullptr);
    assert(ndim >= 1);

    Tensor out = *this;

    int hidden_dim = shape[ndim - 1];
    int hidden_stride = strides[ndim - 1];
    int total_tokens = static_cast<int>(numel()) / hidden_dim;

    for (int token = 0; token < total_tokens; token++) {
        int offset = compute_offset(token, ndim - 1);

        // find mean and variance
        float sum = 0.0f;
        float sum_squares = 0.0f;

        float x;
        for (int i = 0; i < hidden_dim; i++) {
            x = data[offset + i * hidden_stride];
            sum += x;
            sum_squares += x * x;
        }

        float mean = sum / hidden_dim;
        float var = (sum_squares / hidden_dim) - (mean * mean);
        if (var < 0.0f) var = 0.0f;

        // normalize
        float inv_std = 1.0f / std::sqrt(var + eps);

        for (int i = 0; i < hidden_dim; i++) {
            out.data[offset + i * hidden_stride] =
                (data[offset + i * hidden_stride] - mean)
                * inv_std;
        }
    }

    return out;
}

Tensor Tensor::layernorm(const Tensor& gamma, const Tensor& beta, float eps) const {
    assert(data != nullptr);
    assert(ndim >= 1);

    int hidden_dim = shape[ndim - 1];
    int hidden_stride = strides[ndim - 1];

    assert(gamma.data != nullptr);
    assert(beta.data != nullptr);
    assert(gamma.ndim == 1);
    assert(beta.ndim == 1);
    assert(gamma.shape[0] == hidden_dim);
    assert(beta.shape[0] == hidden_dim);

    Tensor out = *this;

    int total_tokens = static_cast<int>(numel()) / hidden_dim;

    for (int token = 0; token < total_tokens; token++) {
        int offset = compute_offset(token, ndim - 1);

        // find mean and variance
        float sum = 0.0f;
        float sum_squares = 0.0f;

        float x;
        for (int i = 0; i < hidden_dim; i++) {
            x = data[offset + i * hidden_stride];
            sum += x;
            sum_squares += x * x;
        }

        float mean = sum / hidden_dim;
        float var = (sum_squares / hidden_dim) - (mean * mean);
        if (var < 0.0f) var = 0.0f;

        // normalize
        float inv_std = 1.0f / std::sqrt(var + eps);

        for (int i = 0; i < hidden_dim; i++) {
            out.data[offset + i * hidden_stride] =
                (gamma.data[i * gamma.strides[0]]
                * (data[offset + i * hidden_stride] - mean))
                * inv_std
                + beta.data[i * beta.strides[0]];
        }
    }

    return out;
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
        int offset = compute_offset(b, axis);

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

Tensor Tensor::embedding(const Tensor& token_ids) const {
    assert(data != nullptr);
    assert(ndim == 2);

    assert(token_ids.data != nullptr);
    assert(token_ids.ndim == 2);

    int vocab_size = shape[0];
    int hidden_dim = shape[1];
    int batches = token_ids.shape[0];
    int total_tokens = token_ids.shape[1];

    Tensor out({batches, total_tokens, hidden_dim});

    for (int b = 0; b < batches; b++) {
        int token_batch = b * token_ids.strides[0];
        int out_batch = b * out.strides[0];

        for (int i = 0; i < total_tokens; i++) {
            int token_offset = token_batch + i * token_ids.strides[1];
            float token_value = token_ids.data[token_offset];

            assert(token_value == static_cast<int>(token_value));

            int token_id = static_cast<int>(token_value);
            assert(token_id >= 0);
            assert(token_id < vocab_size);

            int embedding_offset = token_id * strides[0];
            int out_row = i * out.strides[1];

            for (int j = 0; j < hidden_dim; j++) {
                out.data[out_batch + out_row + j * out.strides[2]] = data[embedding_offset + j * strides[1]];
            }
        }
    }

    return out;
}

Tensor Tensor::reshape(std::initializer_list<int> new_shape) const {
    assert(data != nullptr);
    assert(is_contiguous());

    Tensor out(new_shape);
    assert(out.numel() == numel());
    
    for (size_t i = 0; i < numel(); i++) {
        out.data[i] = data[i];
    }

    return out;
}

Tensor Tensor::transpose(int axis1, int axis2) const {
    assert(data != nullptr);
    assert(axis1 >= 0 && axis1 < ndim);
    assert(axis2 >= 0 && axis2 < ndim);
    assert(axis1 != axis2);

    Tensor out = *this;

    int temp = out.shape[axis1];
    out.shape[axis1] = out.shape[axis2];
    out.shape[axis2] = temp;

    out.init_contiguous_strides();

    int transposed_strides[8];
    std::memcpy(transposed_strides, strides, sizeof(strides));
    temp = transposed_strides[axis1];
    transposed_strides[axis1] = transposed_strides[axis2];
    transposed_strides[axis2] = temp;

    for (size_t elem = 0; elem < out.numel(); elem++) {
        int offset = 0;
        temp = static_cast<int>(elem);

        for (int d = ndim - 1; d >= 0; d--) {
            int idx = temp % out.shape[d];
            temp /= out.shape[d];
            offset += idx * transposed_strides[d];
        }
        
        out.data[elem] = data[offset];
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

            for (int j = 0; j < J; j++) {
                float sum = 0.0f;

                for (int k = 0; k < K; k++) {
                    sum += a_row[k] * other.data[k * J + j];
                }

                c_row[j] += sum;
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