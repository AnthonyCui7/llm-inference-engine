#include <utility>
#include <cassert>
#include <cstring>
#include <iostream>

#include "tensor.hpp"
#include "attention.hpp"
#include "transformer.hpp"
#include "model.hpp"

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

    std::cout << "All tensor tests passed" << std::endl;

    return 0;
}