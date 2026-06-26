#include <utility>
#include <cassert>
#include <cstring>
#include <iostream>

#include "tensor.hpp"

int main() {
    // test default tensor
    Tensor A;

    assert(A.data == nullptr && A.ndim == 0);

    for (int i = 0; i < 8; i++) {
        assert(A.shape[i] == 0);
        assert(A.strides[i] == 0);
    }

    for (int i = 0; i < 32; i++) {
        assert(A.name[i] == 0);
    }

    assert(A.numel() == 0);
    assert(A.dim() == 0);
    assert(A.is_contiguous() == true);
    assert(A.data_ptr() == nullptr);

    // test 2D tensor metadata
    Tensor B({2, 3});

    assert(B.data != nullptr && B.ndim == 2);

    assert(B.shape[0] == 2);
    assert(B.shape[1] == 3);
    assert(B.size(0) == 2);
    assert(B.size(1) == 3);

    assert(B.strides[0] == 3);
    assert(B.strides[1] == 1);
    assert(B.stride(0) == 3);
    assert(B.stride(1) == 1);

    for (int i = 0; i < 6; i++) {
        B.data[i] = i;
        int row = i / 3;
        int col = i % 3;
        assert(B.data[i] == B.at({row, col}));
    }

    for (int i = 2; i < 8; i++) {
        assert(B.shape[i] == 0);
        assert(B.strides[i] == 0);
    }

    for (int i = 0; i < 32; i++) {
        assert(B.name[i] == 0);
    }

    assert(B.numel() == 6);
    assert(B.dim() == 2);
    assert(B.is_contiguous() == true);
    assert(B.data_ptr() != nullptr);

    // test copy constructor/assignment deep copy
    Tensor C = B;
    C.data[0] = 10;
    assert(B.data[0] != 10 && B.data_ptr() != C.data_ptr());

    B = C;
    assert(B.data[0] == 10 && B.data_ptr() != C.data_ptr());

    // test move constructor
    float* stolen_data = B.data;
    int stolen_shape[8];
    int stolen_strides[8];
    int stolen_ndim = B.ndim;
    char stolen_name[32];
    std::memcpy(stolen_shape, B.shape, sizeof(B.shape));
    std::memcpy(stolen_strides, B.strides, sizeof(B.strides));
    std::memcpy(stolen_name, B.name, sizeof(B.name));


    Tensor D = std::move(B);
    assert(B.data == nullptr);
    assert(B.ndim == 0);
    assert(D.data == stolen_data);
    assert(D.ndim == stolen_ndim);

    for (int i = 0; i < 8; i++) {
        assert(B.shape[i] == 0);
        assert(B.strides[i] == 0);
        assert(D.shape[i] == stolen_shape[i]);
        assert(D.strides[i] == stolen_strides[i]);
    }

    for (int i = 0; i < 32; i++) {
        assert(B.name[i] == 0);
        assert(D.name[i] == stolen_name[i]);
    }

    // test move assignment
    stolen_data = C.data;
    stolen_ndim = C.ndim;
    std::memcpy(stolen_shape, C.shape, sizeof(C.shape));
    std::memcpy(stolen_strides, C.strides, sizeof(C.strides));
    std::memcpy(stolen_name, C.name, sizeof(C.name));

    A = std::move(C);

    assert(C.data == nullptr);
    assert(C.ndim == 0);
    assert(A.data == stolen_data);
    assert(A.ndim == stolen_ndim);

    for (int i = 0; i < 8; i++) {
        assert(C.shape[i] == 0);
        assert(C.strides[i] == 0);
        assert(A.shape[i] == stolen_shape[i]);
        assert(A.strides[i] == stolen_strides[i]);
    }

    for (int i = 0; i < 32; i++) {
        assert(C.name[i] == 0);
        assert(A.name[i] == stolen_name[i]);
    }

    assert(A.data[0] == 10);

    // test factory methods
    Tensor Z = Tensor::zeros({2, 3});
    for (size_t i = 0; i < Z.numel(); i++) {
        assert(Z.data[i] == 0.0f);
    }

    Tensor O = Tensor::ones({2, 3});
    for (size_t i = 0; i < O.numel(); i++) {
        assert(O.data[i] == 1.0f);
    }

    Tensor F = Tensor::full({2, 3}, 7.0f);
    for (size_t i = 0; i < F.numel(); i++) {
        assert(F.data[i] == 7.0f);
    }

    Tensor G = Tensor::from_data({2, 3}, {1, 2, 3, 4, 5, 6});
    assert(G.at({0, 0}) == 1.0f);
    assert(G.at({0, 1}) == 2.0f);
    assert(G.at({1, 2}) == 6.0f);

    // test matmul
    Tensor X1 = Tensor::from_data({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor Y1 = Tensor::from_data({3, 2}, {1, 2, 3, 4, 5, 6});
    Tensor Z1 = X1.matmul(Y1);
    
    assert(Z1.at({0, 0}) == 22);
    assert(Z1.at({0, 1}) == 28);
    assert(Z1.at({1, 0}) == 49);
    assert(Z1.at({1, 1}) == 64);

    std::cout << "Test finished" << std::endl;

    return 0;
}