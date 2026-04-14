/*
Header file for a tensor class. Designed to be memory efficient for large scale operations later with LLMs.
*/
#pragma once
#include <vector>
#include <cstring>


struct Tensor {
    float* data;
    std::vector<int> dims;

    Tensor(const std::vector<int>& dimensions) {
        dims = dimensions;
        data = new float[total_size()];
    }

    ~Tensor() {
        delete[] data;
    }

    // copy constructor
    Tensor(const Tensor& other) {
        dims = other.dims;
        int size = total_size();
        data = new float[size];
        std::memcpy(data, other.data, size * sizeof(float));
    }

    // copy assignment operator
    Tensor& operator=(const Tensor& other) {
        if (this == &other) {
            return *this;
        }
        delete[] data;
        dims = other.dims;
        int size = total_size();
        data = new float[size];
        std::memcpy(data, other.data, size * sizeof(float));
        return *this;
    }

    // methods
    int total_size() const;

    float& at(int i, int j);
    float& at(const std::vector<int>& indices);

    int dimensions() const;
};