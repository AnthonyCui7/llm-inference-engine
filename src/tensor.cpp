#include <cassert>

#include "tensor.hpp"

int Tensor::total_size() const {
    int size = 1;
    for (const int& dim : dims) {
        size *= dim;
    }
    return size;
}

float& Tensor::at(int i, int j) {
    return data[i * dims[1] + j];
}

float& Tensor::at(const std::vector<int>& indices) {
    assert(indices.size() == dims.size());
    int index = 0;
    int multiple = 1;
    for (int i = 0; i < indices.size(); i++) {
        assert(indices[i] < dims[i] && indices[i] >= 0);
        for (int j = i + 1; j < indices.size(); j++) {
            multiple *= dims[j];
        }
        index += indices[i] * multiple;
        multiple = 1;
    }
    return data[index];
}

int Tensor::dimensions() const {
    return dims.size();
}