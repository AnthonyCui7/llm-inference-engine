/*
Header file for a tensor class. Designed to be memory efficient for large scale operations later with LLMs.
*/
#pragma once

#include <utility>
#include <cstring>
#include <cassert>
#include <cstddef>
#include <initializer_list>

struct Tensor {
    float* data;        // pointer to the data buffer
    int    shape[8];    // support up to 8 dimensions
    int    strides[8];  // strides for each dimension, used for indexing
    int    ndim;        // number of dimensions (8 or less)
    char   name[32];    // optional name for debugging

    Tensor(std::initializer_list<int> shape, const char* tensor_name = "") {
        assert(shape.size() <= 8);
        ndim = static_cast<int>(shape.size());
        assert(ndim >= 0 && ndim <= 8);

        std::memset(this->shape, 0, sizeof(this->shape));
        std::memset(strides, 0, sizeof(strides));

        int i = 0;
        for (const int& size : shape) {
            assert(size > 0);
            this->shape[i++] = size;
        }

        std::strncpy(name, tensor_name, 31);
        name[31] = '\0';
        
        data = new float[numel()]();
        init_contiguous_strides();
    }

    Tensor(const int* shape, int n, const char* tensor_name = "") {
        assert(n >= 0 && n <= 8);
        assert(shape != nullptr || n == 0);

        ndim = n;

        std::memset(strides, 0, sizeof(strides));
        std::memset(this->shape, 0, sizeof(this->shape));

        for (int i = 0; i < n; i++) {
            assert(shape[i] > 0);
            this->shape[i] = shape[i];
        }

        std::strncpy(name, tensor_name, 31);
        name[31] = '\0';
        
        data = new float[numel()]();
        init_contiguous_strides();
    }

    Tensor() {
        data = nullptr;
        ndim = 0;
        std::memset(shape, 0, sizeof(shape));
        std::memset(strides, 0, sizeof(strides));
        std::memset(name, 0, sizeof(name));
    }

    void swap(Tensor& other) noexcept {
        std::swap(data, other.data);
        std::swap(ndim, other.ndim);

        for (int i = 0; i < 8; i++) {
            std::swap(shape[i], other.shape[i]);
            std::swap(strides[i], other.strides[i]);
        }

        for (int i = 0; i < 32; i++) {
            std::swap(name[i], other.name[i]);
        }
    }

    ~Tensor() {
        delete[] data;
    }

    // copy constructor
    Tensor(const Tensor& other) {
        ndim = other.ndim;

        std::memcpy(shape, other.shape, sizeof(shape));
        std::memcpy(strides, other.strides, sizeof(strides));

        std::strncpy(name, other.name, 31);
        name[31] = '\0';

        size_t size = numel();
        if (other.data == nullptr || size == 0) {
            data = nullptr;
        } else {
            data = new float[size];
            std::memcpy(data, other.data, size * sizeof(float));
        }
    }

    // copy assignment operator
    Tensor& operator=(const Tensor& other) {
        if (this == &other) return *this;
        Tensor temp(other);
        swap(temp);
        return *this;
    }

    // move constructor
    Tensor(Tensor&& other) noexcept {
        data = other.data;
        ndim = other.ndim;

        std::memcpy(shape, other.shape, sizeof(shape));
        std::memcpy(strides, other.strides, sizeof(strides));

        std::strncpy(name, other.name, 31);
        name[31] = '\0';

        other.data = nullptr;
        other.ndim = 0;
        std::memset(other.shape, 0, sizeof(other.shape));
        std::memset(other.strides, 0, sizeof(other.strides));
        std::memset(other.name, 0, sizeof(other.name));
    }

    // move assignment operator
    Tensor& operator=(Tensor&& other) noexcept {
        if (this == &other) return *this;

        delete[] data;

        data = other.data;
        ndim = other.ndim;
        std::memcpy(shape, other.shape, sizeof(shape));
        std::memcpy(strides, other.strides, sizeof(strides));
        std::memcpy(name, other.name, sizeof(name));

        other.data = nullptr;
        other.ndim = 0;
        std::memset(other.shape, 0, sizeof(other.shape));
        std::memset(other.strides, 0, sizeof(other.strides));
        std::memset(other.name, 0, sizeof(other.name));

        return *this;
    }

    // internal
    size_t numel() const;
    int dim() const;
    int size(int axis) const;
    int stride(int axis) const;
    bool is_contiguous() const;

    void init_contiguous_strides();
    int compute_offset(int b, const int* strides, int skip_axis = -1) const;

    float* data_ptr();
    const float* data_ptr() const;

    // factories
    static Tensor zeros(std::initializer_list<int> shape);
    static Tensor ones(std::initializer_list<int> shape);
    static Tensor full(std::initializer_list<int> shape, float value);
    static Tensor from_data(std::initializer_list<int> shape, std::initializer_list<float> values);

    // ops
    Tensor add(float scalar) const;
    Tensor mul(float scalar) const;
    Tensor sub(float scalar) const;
    Tensor div(float scalar) const;
    Tensor add(const Tensor& other) const;
    Tensor mul(const Tensor& other) const;
    Tensor sub(const Tensor& other) const;
    Tensor matmul(const Tensor& other, int thread_count = 0) const;
    Tensor softmax(int axis) const;

    // old/unused test ops
    float& at(std::initializer_list<int> indices);
    Tensor matmul_naive(const Tensor& other) const;
    Tensor matmul_2d_blocked(const Tensor& other, int block_size) const;
    Tensor matmul_2d_threaded(const Tensor& other, int thread_count = 0) const;

    // debug
    void print() const;
    void print_shape() const;
};