#include <iostream>
#include "tensor.hpp"

int Tensor::total_size() const {
    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    return size;
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

void Tensor::print() const {
    std::cout << "Shape: [";
    for (int i = 0; i < ndim; i++) {
        std::cout << shape[i];
        if (i < ndim - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;

    for (int i = 0; i < total_size(); i++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}