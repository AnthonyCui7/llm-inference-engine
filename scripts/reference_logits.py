"""
Dumps huggingface GPT-2 logits for a fixed prompt, for checking the C++
forward pass end to end. Writes the prompt ids and the [seq, vocab]
logits into the usual weight file format.

Usage: python3 scripts/reference_logits.py [model] [out.bin] [prompt]
"""

import struct
import sys

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from dump_gpt2 import write_tensor


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "reference.bin"
    prompt = sys.argv[3] if len(sys.argv) > 3 else "The quick brown fox jumps over the lazy dog"

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # float64: torch's fp32 cpu blas on apple silicon returns NaNs for the
    # larger gpt2 head projections even with finite inputs
    model = GPT2LMHeadModel.from_pretrained(model_name).double()
    model.eval()

    ids = tokenizer.encode(prompt)
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0]

    with open(out_path, "wb") as f:
        f.write(b"LLMW")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<I", 2))
        write_tensor(f, "token_ids", torch.tensor(ids, dtype=torch.float32))
        write_tensor(f, "logits", logits)

    print(f"prompt {prompt!r}, {len(ids)} tokens -> {out_path}")


if __name__ == "__main__":
    main()
