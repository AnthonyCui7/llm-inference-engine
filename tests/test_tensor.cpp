#include <utility>
#include <cassert>
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

    std::cout << "Test finished" << std::endl;

    return 0;
}