CC := clang++
CFLAGS := -std=c++17 -Wall -Wextra -Wpedantic -Isrc
DEBUG_FLAGS := $(CFLAGS) -O0 -g
RELEASE_FLAGS := $(CFLAGS) -O3 -march=native -DNDEBUG

.PHONY: test run-test bench run-bench clean

test:
	$(CC) $(DEBUG_FLAGS) src/tensor.cpp tests/test_tensor.cpp -o bin/test_tensor

run-test: test
	./bin/test_tensor

bench:
	$(CC) $(RELEASE_FLAGS) src/tensor.cpp benchmarks/bench_matmul.cpp -o bin/bench_matmul

run-bench: bench
	./bin/bench_matmul

clean:
	rm -rf bin