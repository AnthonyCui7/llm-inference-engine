"""
Dumps a huggingface GPT-2 checkpoint into the weight file format described
in src/weights.hpp. The fused c_attn projection is split into separate
q/k/v weights, and a small "config" tensor carries the model dimensions.

Usage: python3 scripts/dump_gpt2.py [model] [out.bin]
"""

import struct
import sys

import torch
from transformers import GPT2LMHeadModel


def write_tensor(f, name, tensor):
    data = tensor.detach().to(torch.float32).contiguous()
    name_bytes = name.encode()
    f.write(struct.pack("<I", len(name_bytes)))
    f.write(name_bytes)
    f.write(struct.pack("<I", data.dim()))
    for size in data.shape:
        f.write(struct.pack("<I", size))
    f.write(data.numpy().tobytes())


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    out_path = sys.argv[2] if len(sys.argv) > 2 else model_name + ".bin"

    model = GPT2LMHeadModel.from_pretrained(model_name)
    config = model.config
    state = model.transformer.state_dict()
    hidden = config.n_embd

    tensors = [
        ("config", torch.tensor([config.vocab_size, config.n_positions,
                                 config.n_embd, config.n_head, config.n_layer],
                                dtype=torch.float32)),
        ("wte", state["wte.weight"]),
        ("wpe", state["wpe.weight"]),
    ]

    for i in range(config.n_layer):
        hf = f"h.{i}."
        out = f"h{i}."
        tensors.append((out + "attn_gamma", state[hf + "ln_1.weight"]))
        tensors.append((out + "attn_beta", state[hf + "ln_1.bias"]))

        # c_attn is [hidden, 3 * hidden] with q, k, v side by side
        w_attn = state[hf + "attn.c_attn.weight"]
        b_attn = state[hf + "attn.c_attn.bias"]
        for j, proj in enumerate(("q", "k", "v")):
            tensors.append((out + "w_" + proj, w_attn[:, j * hidden:(j + 1) * hidden]))
            tensors.append((out + "b_" + proj, b_attn[j * hidden:(j + 1) * hidden]))

        tensors.append((out + "w_o", state[hf + "attn.c_proj.weight"]))
        tensors.append((out + "b_o", state[hf + "attn.c_proj.bias"]))
        tensors.append((out + "mlp_gamma", state[hf + "ln_2.weight"]))
        tensors.append((out + "mlp_beta", state[hf + "ln_2.bias"]))
        tensors.append((out + "w_fc", state[hf + "mlp.c_fc.weight"]))
        tensors.append((out + "b_fc", state[hf + "mlp.c_fc.bias"]))
        tensors.append((out + "w_proj", state[hf + "mlp.c_proj.weight"]))
        tensors.append((out + "b_proj", state[hf + "mlp.c_proj.bias"]))

    tensors.append(("final_gamma", state["ln_f.weight"]))
    tensors.append(("final_beta", state["ln_f.bias"]))

    with open(out_path, "wb") as f:
        f.write(b"LLMW")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<I", len(tensors)))
        for name, tensor in tensors:
            write_tensor(f, name, tensor)

    total = sum(t.numel() for _, t in tensors)
    print(f"wrote {len(tensors)} tensors, {total} values to {out_path}")


if __name__ == "__main__":
    main()
