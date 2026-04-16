/*
Header file for a tensor class. Designed to be memory efficient for large scale operations later with LLMs.
*/
#pragma once

#include <cstring>
#include <cassert>


struct Tensor {
    float* data;
    int shape[8];
    int ndim;

    Tensor(std::initializer_list<int> dimensions) {
        ndim = dimensions.size();
        assert(ndim <= 8);
        int i = 0;
        for (const int& dim : dimensions) {
            shape[i++] = dim;
        }
        data = new float[total_size()]();
    }

    Tensor() {
        data = nullptr;
        ndim = 0;
    }

    ~Tensor() {
        delete[] data;
    }

    // copy constructor
    Tensor(const Tensor& other) {
        ndim = other.ndim;
        std::memcpy(shape, other.shape, sizeof(shape));
        int size = total_size();
        data = new float[size];
        std::memcpy(data, other.data, size * sizeof(float));
    }

    // copy assignment operator
    Tensor& operator=(const Tensor& other) {
        if (this == &other) {
            return *this;
        }
        ndim = other.ndim;
        delete[] data;
        std::memcpy(shape, other.shape, sizeof(shape));
        int size = total_size();
        data = new float[size];
        std::memcpy(data, other.data, size * sizeof(float));
        return *this;
    }

    // methods
    int total_size() const;
    float& at(std::initializer_list<int> indices);
    void print() const;
};