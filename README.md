# llm-inference-engine

From scratch C++17 tensor library and transformer inference engine, no
dependencies beyond the STL. A custom `Tensor` struct with manual strides,
a persistent thread pool and SIMD matmul kernels underneath, and a GPT-2
forward pass on top that runs the real pretrained weights.

## Usage

Dump GPT-2 small into the engine's weight format, then generate:

```
python3 scripts/dump_gpt2.py gpt2 gpt2.bin
make generate
./bin/generate gpt2.bin 10 464 3139 286 4881 318
```

The driver works on raw token ids; `scripts/tokenizer.py encode`/`decode`
converts to and from text.

Check the forward pass against huggingface logits on a fixed prompt:

```
python3 scripts/reference_logits.py
make validate
```

`make run-test` runs the unit tests, `make run-bench` the matmul benchmark
(numbers in `benchmarks/matmul.md`).

## Notes

- Positional embeddings are GPT-2's learned kind, so they are just one
  more embedding lookup added to the token embeddings.
- Tokenization stays in python; a C++ BPE tokenizer was not worth it for
  getting a real model running.
- Weights are fp32 and load through the `WeightLoader` interface in
  `src/weights.hpp`. Quantized formats would be added as another loader
  behind the same interface.
- CPU only for now; a GPU backend would be a separate compute path rather
  than a change to the tensor layout.
