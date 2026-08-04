#include <utility>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>

#include "tensor.hpp"
#include "attention.hpp"
#include "transformer.hpp"
#include "model.hpp"
#include "weights.hpp"
#include "generate.hpp"

static void write_u32(std::FILE* f, uint32_t value) {
    std::fwrite(&value, sizeof(value), 1, f);
}

static void write_tensor(std::FILE* f, const std::string& name, const Tensor& t) {
    write_u32(f, static_cast<uint32_t>(name.size()));
    std::fwrite(name.data(), 1, name.size(), f);
    write_u32(f, static_cast<uint32_t>(t.ndim));
    for (int d = 0; d < t.ndim; d++) {
        write_u32(f, static_cast<uint32_t>(t.shape[d]));
    }
    std::fwrite(t.data, sizeof(float), t.numel(), f);
}

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

    // test threaded 2d contiguous matmul
    Tensor matmul_lhs_threaded_2d = Tensor::from_data({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor matmul_rhs_threaded_2d = Tensor::from_data({3, 2}, {1, 2, 3, 4, 5, 6});
    Tensor matmul_out_threaded_2d = matmul_lhs_2d.matmul_2d_threaded(matmul_rhs_2d, 2);
    
    assert(matmul_out_threaded_2d.at({0, 0}) == 22);
    assert(matmul_out_threaded_2d.at({0, 1}) == 28);
    assert(matmul_out_threaded_2d.at({1, 0}) == 49);
    assert(matmul_out_threaded_2d.at({1, 1}) == 64);

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

    // test matmul against naive with K crossing the k block boundary and odd J
    Tensor matmul_lhs_wide({3, 301});
    Tensor matmul_rhs_wide({301, 5});

    for (size_t i = 0; i < matmul_lhs_wide.numel(); i++) {
        matmul_lhs_wide.data[i] = (static_cast<int>(i % 23) - 11) * 0.125f;
    }
    for (size_t i = 0; i < matmul_rhs_wide.numel(); i++) {
        matmul_rhs_wide.data[i] = (static_cast<int>(i % 19) - 9) * 0.0625f;
    }

    Tensor matmul_out_wide = matmul_lhs_wide.matmul(matmul_rhs_wide);
    Tensor matmul_gold_wide = matmul_lhs_wide.matmul_naive(matmul_rhs_wide);

    for (size_t i = 0; i < matmul_gold_wide.numel(); i++) {
        assert(matmul_out_wide.data[i] == matmul_gold_wide.data[i]);
    }

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

    // test causal scaled dot product attention
    Tensor attn_q = Tensor::from_data({2, 2}, {1, 0, 0, 1});
    Tensor attn_k = Tensor::from_data({2, 2}, {1, 0, 0, 1});
    Tensor attn_v = Tensor::from_data({2, 2}, {1, 2, 3, 4});

    Tensor attn_out = scaled_dot_product_attention(attn_q, attn_k, attn_v);

    // row 0 attends only to itself, row 1 weights = softmax(0, 1/sqrt(2))
    assert(attn_out.at({0, 0}) == 1);
    assert(attn_out.at({0, 1}) == 2);
    assert(std::abs(attn_out.at({1, 0}) - 2.3395231f) < 1e-5f);
    assert(std::abs(attn_out.at({1, 1}) - 3.3395231f) < 1e-5f);

    // test multi head attention with one head matches single head attention exactly
    Tensor mha_q = Tensor::from_data({1, 2, 2}, {1, 0, 0, 1});
    Tensor mha_k = Tensor::from_data({1, 2, 2}, {1, 0, 0, 1});
    Tensor mha_v = Tensor::from_data({1, 2, 2}, {1, 2, 3, 4});

    Tensor mha_out_single = multi_head_attention(mha_q, mha_k, mha_v, 1);

    assert(mha_out_single.ndim == 3);
    assert(mha_out_single.numel() == attn_out.numel());

    for (size_t i = 0; i < attn_out.numel(); i++) {
        assert(mha_out_single.data[i] == attn_out.data[i]);
    }

    // test two heads, each head fed the same values as the single head case,
    // so both halves of every output row must match the single head output
    Tensor mha_q2 = Tensor::from_data({1, 2, 4}, {1, 0, 1, 0, 0, 1, 0, 1});
    Tensor mha_k2 = Tensor::from_data({1, 2, 4}, {1, 0, 1, 0, 0, 1, 0, 1});
    Tensor mha_v2 = Tensor::from_data({1, 2, 4}, {1, 2, 1, 2, 3, 4, 3, 4});

    Tensor mha_out_two = multi_head_attention(mha_q2, mha_k2, mha_v2, 2);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            float expected = attn_out.at({i, j});
            assert(std::abs(mha_out_two.at({0, i, j}) - expected) < 1e-5f);
            assert(std::abs(mha_out_two.at({0, i, j + 2}) - expected) < 1e-5f);
        }
    }

    // test the query offset path: the last query alone, offset past the
    // earlier keys, must reproduce the last row of the full output bitwise
    Tensor attn_q_last = Tensor::from_data({1, 2}, {0, 1});
    Tensor attn_out_last = scaled_dot_product_attention(attn_q_last, attn_k, attn_v, 1);

    assert(attn_out_last.shape[0] == 1 && attn_out_last.shape[1] == 2);
    assert(attn_out_last.data[0] == attn_out.at({1, 0}));
    assert(attn_out_last.data[1] == attn_out.at({1, 1}));

    Tensor mha_q_last = Tensor::from_data({1, 1, 4}, {0, 1, 0, 1});
    Tensor mha_out_last = multi_head_attention(mha_q_last, mha_k2, mha_v2, 2, 1);

    assert(mha_out_last.shape[1] == 1 && mha_out_last.shape[2] == 4);
    for (int j = 0; j < 4; j++) {
        assert(mha_out_last.at({0, 0, j}) == mha_out_two.at({0, 1, j}));
    }

    // test self attention with identity projections and zero biases matches
    // multi head attention exactly
    Tensor sa_x = Tensor::from_data({1, 2, 2}, {1, 2, 3, 4});
    Tensor sa_eye = Tensor::from_data({2, 2}, {1, 0, 0, 1});
    Tensor sa_zero_bias = Tensor::zeros({2});

    Tensor sa_out = self_attention(sa_x, sa_eye, sa_zero_bias, sa_eye, sa_zero_bias,
                                   sa_eye, sa_zero_bias, sa_eye, sa_zero_bias, 1);
    Tensor sa_expected = multi_head_attention(sa_x, sa_x, sa_x, 1);

    assert(sa_out.ndim == 3);
    assert(sa_out.shape[0] == 1 && sa_out.shape[1] == 2 && sa_out.shape[2] == 2);

    for (size_t i = 0; i < sa_expected.numel(); i++) {
        assert(sa_out.data[i] == sa_expected.data[i]);
    }

    // doubling the value projection doubles the output exactly, since the
    // attention weights only depend on q and k
    Tensor sa_eye_doubled = Tensor::from_data({2, 2}, {2, 0, 0, 2});
    Tensor sa_out_doubled = self_attention(sa_x, sa_eye, sa_zero_bias, sa_eye, sa_zero_bias,
                                           sa_eye_doubled, sa_zero_bias, sa_eye, sa_zero_bias, 1);

    for (size_t i = 0; i < sa_out.numel(); i++) {
        assert(sa_out_doubled.data[i] == 2 * sa_out.data[i]);
    }

    // test the attention block reduces to the bare residual when the output
    // projection is zero; would fail if the residual added layernorm(x) instead of x
    Tensor block_gamma = Tensor::ones({2});
    Tensor block_beta = Tensor::zeros({2});
    Tensor block_w_o_zero = Tensor::zeros({2, 2});

    Tensor block_out = attention_block(sa_x, block_gamma, block_beta, sa_eye, sa_zero_bias,
                                       sa_eye, sa_zero_bias, sa_eye, sa_zero_bias,
                                       block_w_o_zero, sa_zero_bias, 1);

    assert(block_out.ndim == 3);
    assert(block_out.shape[0] == 1 && block_out.shape[1] == 2 && block_out.shape[2] == 2);

    for (size_t i = 0; i < sa_x.numel(); i++) {
        assert(block_out.data[i] == sa_x.data[i]);
    }

    // with the output projection zeroed, its bias passes straight through,
    // so the block becomes x + b_o exactly
    Tensor bias_shift = Tensor::from_data({2}, {5, 7});
    Tensor block_out_bias = attention_block(sa_x, block_gamma, block_beta, sa_eye, sa_zero_bias,
                                            sa_eye, sa_zero_bias, sa_eye, sa_zero_bias,
                                            block_w_o_zero, bias_shift, 1);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            assert(block_out_bias.at({0, i, j}) == sa_x.at({0, i, j}) + bias_shift.at({j}));
        }
    }

    // test the mlp block: identity weights reduce it to x + gelu(layernorm(x)),
    // and layernorm maps every row of sa_x to roughly [-1, 1]
    Tensor mlp_out = mlp_block(sa_x, block_gamma, block_beta, sa_eye, sa_zero_bias, sa_eye, sa_zero_bias);

    assert(mlp_out.ndim == 3);
    assert(std::abs(mlp_out.at({0, 0, 0}) - 0.8411920f) < 1e-4f);
    assert(std::abs(mlp_out.at({0, 0, 1}) - 2.8411920f) < 1e-4f);
    assert(std::abs(mlp_out.at({0, 1, 0}) - 2.8411920f) < 1e-4f);
    assert(std::abs(mlp_out.at({0, 1, 1}) - 4.8411920f) < 1e-4f);

    // zero second linear reduces the mlp block to the bare residual
    Tensor mlp_out_zero = mlp_block(sa_x, block_gamma, block_beta, sa_eye, sa_zero_bias,
                                    block_w_o_zero, sa_zero_bias);

    for (size_t i = 0; i < sa_x.numel(); i++) {
        assert(mlp_out_zero.data[i] == sa_x.data[i]);
    }

    // and its bias passes through the zeroed linear, giving x + b_proj exactly
    Tensor mlp_out_bias = mlp_block(sa_x, block_gamma, block_beta, sa_eye, sa_zero_bias,
                                    block_w_o_zero, bias_shift);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            assert(mlp_out_bias.at({0, i, j}) == sa_x.at({0, i, j}) + bias_shift.at({j}));
        }
    }

    // test the full transformer block and stacking: zeroing both output
    // projections makes every block a no-op, so input round trips unchanged
    TransformerBlockWeights noop_block;
    noop_block.attn_gamma = Tensor::ones({2});
    noop_block.attn_beta = Tensor::zeros({2});
    noop_block.w_q = sa_eye;
    noop_block.b_q = sa_zero_bias;
    noop_block.w_k = sa_eye;
    noop_block.b_k = sa_zero_bias;
    noop_block.w_v = sa_eye;
    noop_block.b_v = sa_zero_bias;
    noop_block.w_o = Tensor::zeros({2, 2});
    noop_block.b_o = sa_zero_bias;
    noop_block.mlp_gamma = Tensor::ones({2});
    noop_block.mlp_beta = Tensor::zeros({2});
    noop_block.w_fc = sa_eye;
    noop_block.b_fc = sa_zero_bias;
    noop_block.w_proj = Tensor::zeros({2, 2});
    noop_block.b_proj = sa_zero_bias;

    Tensor single_block_out = transformer_block(sa_x, noop_block, 1);

    assert(single_block_out.ndim == 3);
    for (size_t i = 0; i < sa_x.numel(); i++) {
        assert(single_block_out.data[i] == sa_x.data[i]);
    }

    std::vector<TransformerBlockWeights> noop_layers(3, noop_block);
    Tensor stack_out = transformer_stack(sa_x, noop_layers, 1);

    assert(stack_out.ndim == 3);
    assert(stack_out.shape[0] == 1 && stack_out.shape[1] == 2 && stack_out.shape[2] == 2);
    for (size_t i = 0; i < sa_x.numel(); i++) {
        assert(stack_out.data[i] == sa_x.data[i]);
    }

    // a stack with live weights just needs to come out the right shape
    TransformerBlockWeights live_block = noop_block;
    live_block.w_o = sa_eye;
    live_block.w_proj = sa_eye;

    std::vector<TransformerBlockWeights> live_layers(2, live_block);
    Tensor live_out = transformer_stack(sa_x, live_layers, 1);

    assert(live_out.ndim == 3);
    assert(live_out.shape[0] == 1 && live_out.shape[1] == 2 && live_out.shape[2] == 2);

    // test token plus positional embeddings: every output row must be
    // exactly wte[token] + wpe[position]
    Tensor embed_wte = Tensor::from_data({4, 2}, {
        0, 1,
        10, 11,
        20, 21,
        30, 31
    });
    Tensor embed_wpe = Tensor::from_data({3, 2}, {
        100, 200,
        300, 400,
        500, 600
    });
    Tensor embed_ids = Tensor::from_data({2, 2}, {
        2, 0,
        3, 3
    });

    Tensor embed_out = embed(embed_wte, embed_wpe, embed_ids);

    assert(embed_out.ndim == 3);
    assert(embed_out.shape[0] == 2 && embed_out.shape[1] == 2 && embed_out.shape[2] == 2);

    assert(embed_out.at({0, 0, 0}) == 120 && embed_out.at({0, 0, 1}) == 221);
    assert(embed_out.at({0, 1, 0}) == 300 && embed_out.at({0, 1, 1}) == 401);
    assert(embed_out.at({1, 0, 0}) == 130 && embed_out.at({1, 0, 1}) == 231);
    assert(embed_out.at({1, 1, 0}) == 330 && embed_out.at({1, 1, 1}) == 431);

    // test the model forward pass: with no-op blocks the logits must be
    // exactly layernorm(embeddings) times the tied unembedding
    ModelWeights tiny_model;
    tiny_model.config = {4, 3, 2, 1, 2};
    tiny_model.wte = embed_wte;
    tiny_model.wpe = embed_wpe;
    tiny_model.blocks = std::vector<TransformerBlockWeights>(2, noop_block);
    tiny_model.final_gamma = Tensor::ones({2});
    tiny_model.final_beta = Tensor::zeros({2});
    tiny_model.lm_head = embed_wte.transpose(0, 1);

    Tensor logits = model_forward(tiny_model, embed_ids);

    assert(logits.ndim == 3);
    assert(logits.shape[0] == 2 && logits.shape[1] == 2 && logits.shape[2] == 4);

    Tensor logits_expected = embed(embed_wte, embed_wpe, embed_ids)
        .layernorm(tiny_model.final_gamma, tiny_model.final_beta)
        .reshape({4, 2}).matmul(tiny_model.lm_head).reshape({2, 2, 4});

    for (size_t i = 0; i < logits.numel(); i++) {
        assert(logits.data[i] == logits_expected.data[i]);
    }

    // live blocks just need the right logits shape end to end
    tiny_model.blocks = std::vector<TransformerBlockWeights>(2, live_block);
    Tensor logits_live = model_forward(tiny_model, embed_ids);

    assert(logits_live.shape[0] == 2 && logits_live.shape[1] == 2 && logits_live.shape[2] == 4);

    // test the weight file loader: write a small file in the documented
    // format by hand and read it back
    const char* weight_path = "bin/test_weights.bin";
    std::FILE* wf = std::fopen(weight_path, "wb");
    assert(wf != nullptr);

    std::fwrite("LLMW", 1, 4, wf);
    write_u32(wf, 1);  // version
    write_u32(wf, 2);  // tensor count

    write_u32(wf, 5);
    std::fwrite("alpha", 1, 5, wf);
    write_u32(wf, 2);
    write_u32(wf, 2);
    write_u32(wf, 3);
    float alpha_values[6] = {1, 2, 3, 4, 5, 6.5f};
    std::fwrite(alpha_values, sizeof(float), 6, wf);

    write_u32(wf, 4);
    std::fwrite("beta", 1, 4, wf);
    write_u32(wf, 1);
    write_u32(wf, 4);
    float beta_values[4] = {-1, 0, 0.25f, 8};
    std::fwrite(beta_values, sizeof(float), 4, wf);

    std::fclose(wf);

    Fp32FileLoader loader(weight_path);

    assert(loader.has("alpha"));
    assert(loader.has("beta"));
    assert(!loader.has("gamma"));

    Tensor alpha = loader.load("alpha");
    assert(alpha.ndim == 2 && alpha.shape[0] == 2 && alpha.shape[1] == 3);
    for (int i = 0; i < 6; i++) {
        assert(alpha.data[i] == alpha_values[i]);
    }

    Tensor beta = loader.load("beta");
    assert(beta.ndim == 1 && beta.shape[0] == 4);
    for (int i = 0; i < 4; i++) {
        assert(beta.data[i] == beta_values[i]);
    }

    // load hands out copies, so scribbling on one does not touch the loader
    alpha.data[0] = 99;
    assert(loader.load("alpha").data[0] == 1);

    std::remove(weight_path);

    // test load_model: dump the tiny no-op model to a file under the
    // dump script's naming scheme and check the loaded forward pass
    // reproduces the in-memory one exactly
    const char* model_path = "bin/test_model.bin";
    std::FILE* mf = std::fopen(model_path, "wb");
    assert(mf != nullptr);

    std::fwrite("LLMW", 1, 4, mf);
    write_u32(mf, 1);
    write_u32(mf, 5 + 2 * 16);

    write_tensor(mf, "config", Tensor::from_data({5}, {4, 3, 2, 1, 2}));
    write_tensor(mf, "wte", embed_wte);
    write_tensor(mf, "wpe", embed_wpe);

    for (int i = 0; i < 2; i++) {
        std::string prefix = "h" + std::to_string(i) + ".";
        write_tensor(mf, prefix + "attn_gamma", noop_block.attn_gamma);
        write_tensor(mf, prefix + "attn_beta", noop_block.attn_beta);
        write_tensor(mf, prefix + "w_q", noop_block.w_q);
        write_tensor(mf, prefix + "b_q", noop_block.b_q);
        write_tensor(mf, prefix + "w_k", noop_block.w_k);
        write_tensor(mf, prefix + "b_k", noop_block.b_k);
        write_tensor(mf, prefix + "w_v", noop_block.w_v);
        write_tensor(mf, prefix + "b_v", noop_block.b_v);
        write_tensor(mf, prefix + "w_o", noop_block.w_o);
        write_tensor(mf, prefix + "b_o", noop_block.b_o);
        write_tensor(mf, prefix + "mlp_gamma", noop_block.mlp_gamma);
        write_tensor(mf, prefix + "mlp_beta", noop_block.mlp_beta);
        write_tensor(mf, prefix + "w_fc", noop_block.w_fc);
        write_tensor(mf, prefix + "b_fc", noop_block.b_fc);
        write_tensor(mf, prefix + "w_proj", noop_block.w_proj);
        write_tensor(mf, prefix + "b_proj", noop_block.b_proj);
    }

    write_tensor(mf, "final_gamma", tiny_model.final_gamma);
    write_tensor(mf, "final_beta", tiny_model.final_beta);
    std::fclose(mf);

    Fp32FileLoader model_loader(model_path);
    ModelWeights loaded_model = load_model(model_loader);

    assert(loaded_model.config.vocab_size == 4);
    assert(loaded_model.config.max_seq == 3);
    assert(loaded_model.config.hidden == 2);
    assert(loaded_model.config.num_heads == 1);
    assert(loaded_model.config.num_layers == 2);
    assert(loaded_model.lm_head.ndim == 2);
    assert(loaded_model.lm_head.shape[0] == 2 && loaded_model.lm_head.shape[1] == 4);

    Tensor loaded_logits = model_forward(loaded_model, embed_ids);

    assert(loaded_logits.numel() == logits.numel());
    for (size_t i = 0; i < logits.numel(); i++) {
        assert(loaded_logits.data[i] == logits.data[i]);
    }

    std::remove(model_path);

    // test greedy generation: keeps the prompt, stays in vocab, and the
    // first new token matches a manual argmax over the forward pass
    std::vector<int> gen_prompt = {2};
    std::vector<int> gen_out = generate(tiny_model, gen_prompt, 2);

    assert(gen_out.size() == 3);
    assert(gen_out[0] == 2);
    for (int t : gen_out) {
        assert(t >= 0 && t < 4);
    }

    Tensor gen_ids = Tensor::from_data({1, 1}, {2});
    Tensor gen_logits = model_forward(tiny_model, gen_ids);

    int expected_next = 0;
    for (int t = 1; t < 4; t++) {
        if (gen_logits.data[t] > gen_logits.data[expected_next]) expected_next = t;
    }
    assert(gen_out[1] == expected_next);

    std::cout << "All tensor tests passed" << std::endl;

    return 0;
}