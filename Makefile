CC := clang++
CFLAGS := -std=c++17 -Wall -Wextra -Wpedantic -Isrc
THREAD_FLAGS := -pthread

DEBUG_FLAGS := $(CFLAGS) -O0 -g
RELEASE_FLAGS := $(CFLAGS) -O3 -march=native -DNDEBUG
FAST_MATH_FLAGS := $(RELEASE_FLAGS) -ffast-math

# everything except the driver's main
CORE_SRC := $(filter-out src/main.cpp, $(wildcard src/*.cpp))

.PHONY: test run-test bench run-bench bench-fast-math bench-generate run-bench-generate generate validate clean

test:
	$(CC) $(DEBUG_FLAGS) $(THREAD_FLAGS) $(CORE_SRC) tests/test_tensor.cpp -o bin/test_tensor

run-test: test
	./bin/test_tensor

bench:
	$(CC) $(RELEASE_FLAGS) $(THREAD_FLAGS) $(CORE_SRC) benchmarks/bench_matmul.cpp -o bin/bench_matmul

run-bench: bench
	./bin/bench_matmul

bench-fast-math:
	$(CC) $(FAST_MATH_FLAGS) $(THREAD_FLAGS) $(CORE_SRC) benchmarks/bench_matmul.cpp -o bin/bench_matmul_fast_math

# needs gpt2.bin from scripts/dump_gpt2.py
bench-generate:
	$(CC) $(RELEASE_FLAGS) $(THREAD_FLAGS) $(CORE_SRC) benchmarks/bench_generate.cpp -o bin/bench_generate

run-bench-generate: bench-generate
	./bin/bench_generate

generate:
	$(CC) $(RELEASE_FLAGS) $(THREAD_FLAGS) $(CORE_SRC) src/main.cpp -o bin/generate

# needs gpt2.bin from scripts/dump_gpt2.py and reference.bin from
# scripts/reference_logits.py
validate:
	$(CC) $(RELEASE_FLAGS) $(THREAD_FLAGS) $(CORE_SRC) tests/test_gpt2_logits.cpp -o bin/test_gpt2_logits
	./bin/test_gpt2_logits

clean:
	rm -rf bin/*
