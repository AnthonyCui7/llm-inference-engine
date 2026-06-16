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

    Tensor(std::initializer_list<int> dimensions, const char* tensor_name = "") {
        assert(dimensions.size() <= 8);
        ndim = static_cast<int>(dimensions.size());
        assert(ndim >= 0 && ndim <= 8);
        std::memset(shape, 0, sizeof(shape));
        std::memset(strides, 0, sizeof(strides));
        int i = 0;
        for (const int& dim : dimensions) {
            assert(dim > 0);
            shape[i++] = dim;
        }
        std::strncpy(name, tensor_name, 31);
        name[31] = '\0';
        data = new float[total_size()]();
        init_strides();
    }

    Tensor(const int* dimensions, int n, const char* tensor_name = "") {
        assert(n >= 0 && n <= 8);
        assert(dimensions != nullptr || n == 0);
        ndim = n;
        std::memset(strides, 0, sizeof(strides));
        std::memset(shape, 0, sizeof(shape));
        for (int i = 0; i < n; i++) {
            assert(dimensions[i] > 0);
            shape[i] = dimensions[i];
        }
        std::strncpy(name, tensor_name, 31);
        name[31] = '\0';
        data = new float[total_size()]();
        init_strides();
    }

    Tensor() {
        data = nullptr;
        ndim = 0;
        name[0] = '\0';
        std::memset(shape, 0, sizeof(shape));
        std::memset(strides, 0, sizeof(strides));
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
        size_t size = total_size();
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
        other.name[0] = '\0';
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
        other.name[0] = '\0';

        return *this;
    }

    // internal
    size_t total_size() const;
    void   init_strides();
    void   compute_strides(int* out_strides) const;
    int    compute_offset(int b, const int* strides, int skip_axis = -1) const;

    // ops
    float& at(std::initializer_list<int> indices);
    Tensor matmul(const Tensor& other) const;
    Tensor softmax(int axis) const;

    // debug
    void   print() const;
};