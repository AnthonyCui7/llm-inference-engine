#include <utility>
#include <cassert>
#include <cstring>
#include <iostream>

#include "tensor.hpp"

int main() {
    // test default tensor
    Tensor default_tensor;

    assert(default_tensor.data == nullptr && default_tensor.ndim == 0);

    for (int i = 0; i < 8; i++) {
        assert(default_tensor.shape[i] == 0);
        assert(default_tensor.strides[i] == 0);
    }

    for (int i = 0; i < 32; i++) {
        assert(default_tensor.name[i] == 0);
    }

    assert(default_tensor.numel() == 0);
    assert(default_tensor.dim() == 0);
    assert(default_tensor.is_contiguous() == true);
    assert(default_tensor.data_ptr() == nullptr);

    // test 2D tensor metadata
    Tensor matrix_2d({2, 3});

    assert(matrix_2d.data != nullptr && matrix_2d.ndim == 2);

    assert(matrix_2d.shape[0] == 2);
    assert(matrix_2d.shape[1] == 3);
    assert(matrix_2d.size(0) == 2);
    assert(matrix_2d.size(1) == 3);

    assert(matrix_2d.strides[0] == 3);
    assert(matrix_2d.strides[1] == 1);
    assert(matrix_2d.stride(0) == 3);
    assert(matrix_2d.stride(1) == 1);

    for (int i = 0; i < 6; i++) {
        matrix_2d.data[i] = i;
        int row = i / 3;
        int col = i % 3;
        assert(matrix_2d.data[i] == matrix_2d.at({row, col}));
    }

    for (int i = 2; i < 8; i++) {
        assert(matrix_2d.shape[i] == 0);
        assert(matrix_2d.strides[i] == 0);
    }

    for (int i = 0; i < 32; i++) {
        assert(matrix_2d.name[i] == 0);
    }

    assert(matrix_2d.numel() == 6);
    assert(matrix_2d.dim() == 2);
    assert(matrix_2d.is_contiguous() == true);
    assert(matrix_2d.data_ptr() != nullptr);

    // test copy constructor/assignment deep copy
    Tensor copied_matrix = matrix_2d;
    copied_matrix.data[0] = 10;
    assert(matrix_2d.data[0] != 10 && matrix_2d.data_ptr() != copied_matrix.data_ptr());

    matrix_2d = copied_matrix;
    assert(matrix_2d.data[0] == 10 && matrix_2d.data_ptr() != copied_matrix.data_ptr());

    // test move constructor
    float* stolen_data = matrix_2d.data;
    int stolen_shape[8];
    int stolen_strides[8];
    int stolen_ndim = matrix_2d.ndim;
    char stolen_name[32];
    std::memcpy(stolen_shape, matrix_2d.shape, sizeof(matrix_2d.shape));
    std::memcpy(stolen_strides, matrix_2d.strides, sizeof(matrix_2d.strides));
    std::memcpy(stolen_name, matrix_2d.name, sizeof(matrix_2d.name));

    Tensor moved_matrix = std::move(matrix_2d);
    assert(matrix_2d.data == nullptr);
    assert(matrix_2d.ndim == 0);
    assert(moved_matrix.data == stolen_data);
    assert(moved_matrix.ndim == stolen_ndim);

    for (int i = 0; i < 8; i++) {
        assert(matrix_2d.shape[i] == 0);
        assert(matrix_2d.strides[i] == 0);
        assert(moved_matrix.shape[i] == stolen_shape[i]);
        assert(moved_matrix.strides[i] == stolen_strides[i]);
    }

    for (int i = 0; i < 32; i++) {
        assert(matrix_2d.name[i] == 0);
        assert(moved_matrix.name[i] == stolen_name[i]);
    }

    // test move assignment
    stolen_data = copied_matrix.data;
    stolen_ndim = copied_matrix.ndim;
    std::memcpy(stolen_shape, copied_matrix.shape, sizeof(copied_matrix.shape));
    std::memcpy(stolen_strides, copied_matrix.strides, sizeof(copied_matrix.strides));
    std::memcpy(stolen_name, copied_matrix.name, sizeof(copied_matrix.name));

    default_tensor = std::move(copied_matrix);

    assert(copied_matrix.data == nullptr);
    assert(copied_matrix.ndim == 0);
    assert(default_tensor.data == stolen_data);
    assert(default_tensor.ndim == stolen_ndim);

    for (int i = 0; i < 8; i++) {
        assert(copied_matrix.shape[i] == 0);
        assert(copied_matrix.strides[i] == 0);
        assert(default_tensor.shape[i] == stolen_shape[i]);
        assert(default_tensor.strides[i] == stolen_strides[i]);
    }

    for (int i = 0; i < 32; i++) {
        assert(copied_matrix.name[i] == 0);
        assert(default_tensor.name[i] == stolen_name[i]);
    }

    assert(default_tensor.data[0] == 10);

    // test factory methods
    Tensor zeros_tensor = Tensor::zeros({2, 3});
    for (size_t i = 0; i < zeros_tensor.numel(); i++) {
        assert(zeros_tensor.data[i] == 0.0f);
    }

    Tensor ones_tensor = Tensor::ones({2, 3});
    for (size_t i = 0; i < ones_tensor.numel(); i++) {
        assert(ones_tensor.data[i] == 1.0f);
    }

    Tensor full_tensor = Tensor::full({2, 3}, 7.0f);
    for (size_t i = 0; i < full_tensor.numel(); i++) {
        assert(full_tensor.data[i] == 7.0f);
    }

    Tensor data_tensor = Tensor::from_data({2, 3}, {1, 2, 3, 4, 5, 6});
    assert(data_tensor.at({0, 0}) == 1.0f);
    assert(data_tensor.at({0, 1}) == 2.0f);
    assert(data_tensor.at({1, 2}) == 6.0f);

    // test matmul
    Tensor matmul_lhs_2d = Tensor::from_data({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor matmul_rhs_2d = Tensor::from_data({3, 2}, {1, 2, 3, 4, 5, 6});
    Tensor matmul_out_2d = matmul_lhs_2d.matmul(matmul_rhs_2d);
    
    assert(matmul_out_2d.at({0, 0}) == 22);
    assert(matmul_out_2d.at({0, 1}) == 28);
    assert(matmul_out_2d.at({1, 0}) == 49);
    assert(matmul_out_2d.at({1, 1}) == 64);

    Tensor matmul_lhs_3d = Tensor::from_data({2, 2, 3}, {
        1, 2, 3,
        4, 5, 6,

        7, 8, 9,
        10, 11, 12
    });

    Tensor matmul_rhs_3d = Tensor::from_data({2, 3, 2}, {
        1, 2,
        3, 4,
        5, 6,

        7, 8,
        9, 10,
        11, 12
    });

    Tensor matmul_out_3d = matmul_lhs_3d.matmul(matmul_rhs_3d);

    assert(matmul_out_3d.at({0, 0, 0}) == 22);
    assert(matmul_out_3d.at({0, 0, 1}) == 28);
    assert(matmul_out_3d.at({0, 1, 0}) == 49);
    assert(matmul_out_3d.at({0, 1, 1}) == 64);

    assert(matmul_out_3d.at({1, 0, 0}) == 220);
    assert(matmul_out_3d.at({1, 0, 1}) == 244);
    assert(matmul_out_3d.at({1, 1, 0}) == 301);
    assert(matmul_out_3d.at({1, 1, 1}) == 334);

    Tensor matmul_lhs_4d = Tensor::from_data({2, 1, 2, 3}, {
        1, 2, 3,
        4, 5, 6,

        7, 8, 9,
        10, 11, 12
    });

    Tensor matmul_rhs_4d = Tensor::from_data({2, 1, 3, 2}, {
        1, 2,
        3, 4,
        5, 6,

        7, 8,
        9, 10,
        11, 12
    });

    Tensor matmul_out_4d = matmul_lhs_4d.matmul(matmul_rhs_4d);

    assert(matmul_out_4d.at({0, 0, 0, 0}) == 22);
    assert(matmul_out_4d.at({0, 0, 0, 1}) == 28);
    assert(matmul_out_4d.at({0, 0, 1, 0}) == 49);
    assert(matmul_out_4d.at({0, 0, 1, 1}) == 64);

    assert(matmul_out_4d.at({1, 0, 0, 0}) == 220);
    assert(matmul_out_4d.at({1, 0, 0, 1}) == 244);
    assert(matmul_out_4d.at({1, 0, 1, 0}) == 301);
    assert(matmul_out_4d.at({1, 0, 1, 1}) == 334);

    // test softmax
    Tensor softmax_input = Tensor::from_data({2, 3}, {
        1, 2, 3,
        1, 2, 3
    });

    Tensor softmax_out = softmax_input.softmax(1);

    assert(std::abs(softmax_out.at({0, 0}) - 0.09003057f) < 1e-5f);
    assert(std::abs(softmax_out.at({0, 1}) - 0.24472847f) < 1e-5f);
    assert(std::abs(softmax_out.at({0, 2}) - 0.66524096f) < 1e-5f);

    assert(std::abs(softmax_out.at({1, 0}) - 0.09003057f) < 1e-5f);
    assert(std::abs(softmax_out.at({1, 1}) - 0.24472847f) < 1e-5f);
    assert(std::abs(softmax_out.at({1, 2}) - 0.66524096f) < 1e-5f);

    assert(std::abs(softmax_out.at({0, 0}) + softmax_out.at({0, 1}) + softmax_out.at({0, 2}) - 1.0f) < 1e-5f);
    assert(std::abs(softmax_out.at({1, 0}) + softmax_out.at({1, 1}) + softmax_out.at({1, 2}) - 1.0f) < 1e-5f);

        Tensor softmax_axis0_input = Tensor::from_data({2, 3}, {
        1, 2, 3,
        4, 5, 6
    });

    Tensor softmax_axis0_out = softmax_axis0_input.softmax(0);

    assert(std::abs(softmax_axis0_out.at({0, 0}) - 0.04742587f) < 1e-5f);
    assert(std::abs(softmax_axis0_out.at({1, 0}) - 0.95257413f) < 1e-5f);

    assert(std::abs(softmax_axis0_out.at({0, 1}) - 0.04742587f) < 1e-5f);
    assert(std::abs(softmax_axis0_out.at({1, 1}) - 0.95257413f) < 1e-5f);

    assert(std::abs(softmax_axis0_out.at({0, 2}) - 0.04742587f) < 1e-5f);
    assert(std::abs(softmax_axis0_out.at({1, 2}) - 0.95257413f) < 1e-5f);

    assert(std::abs(softmax_axis0_out.at({0, 0}) + softmax_axis0_out.at({1, 0}) - 1.0f) < 1e-5f);
    assert(std::abs(softmax_axis0_out.at({0, 1}) + softmax_axis0_out.at({1, 1}) - 1.0f) < 1e-5f);
    assert(std::abs(softmax_axis0_out.at({0, 2}) + softmax_axis0_out.at({1, 2}) - 1.0f) < 1e-5f);

    std::cout << "Test finished" << std::endl;

    return 0;
}