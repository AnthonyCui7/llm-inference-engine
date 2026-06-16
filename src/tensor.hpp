/*
Header file for a tensor class. Designed to be memory efficient for large scale operations later with LLMs.
*/
#pragma once

#include <cstring>
#include <cassert>
#include <utility>
#include <initializer_list>

struct Tensor {
    float* data;        // pointer to the data buffer
    int    shape[8];    // support up to 8 dimensions
    int    strides[8];  // strides for each dimension, used for indexing
    int    ndim;        // number of dimensions (8 or less)
    char   name[64];    // optional name for debugging

    Tensor(std::initializer_list<int> dimensions, const char* tensor_name = "") {
        ndim = dimensions.size();
        assert(ndim <= 8);
        int i = 0;
        for (const int& dim : dimensions) {
            shape[i++] = dim;
        }
        strncpy(name, tensor_name, 63);
        name[63] = '\0';
        data = new float[total_size()]();
        init_strides();
    }

    Tensor(const int* dimensions, int n, const char* tensor_name = "") {
        ndim = n;
        assert(ndim <= 8);
        std::memcpy(shape, dimensions, n * sizeof(int));
        strncpy(name, tensor_name, 63);
        name[63] = '\0';
        data = new float[total_size()]();
        init_strides();
    }

    Tensor() {
        data = nullptr;
        ndim = 0;
        name[0] = '\0';
        std::memset(strides, 0, sizeof(strides));
    }

    void swap(Tensor& other) noexcept {
        std::swap(data, other.data);
        std::swap(ndim, other.ndim);

        for (int i = 0; i < 8; i++) {
            std::swap(shape[i], other.shape[i]);
            std::swap(strides[i], other.strides[i]);
        }

        for (int i = 0; i < 64; i++) {
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
        strncpy(name, other.name, 63);
        name[63] = '\0';
        int size = total_size();
        data = new float[size];
        std::memcpy(data, other.data, size * sizeof(float));
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
        ndim = other.ndim;
        std::memcpy(shape, other.shape, sizeof(shape));
        std::memcpy(strides, other.strides, sizeof(strides));
        strncpy(name, other.name, 63);
        name[63] = '\0';
        data = other.data;
        other.data = nullptr;
    }

    // move assignment operator
    Tensor& operator=(Tensor&& other) noexcept {
        if (this == &other) return *this;
        Tensor temp(std::move(other));
        swap(temp);
        return *this;
    }

    // internal
    int    total_size() const;
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