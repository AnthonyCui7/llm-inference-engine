/*
Binary weight file format and loaders.

File layout, all little endian:
  char[4]  magic "LLMW"
  u32      version, currently 1
  u32      tensor count
then per tensor, back to back:
  u32      name length
  char[]   name
  u32      ndim
  u32[]    shape
  f32[]    data, row major
*/
#pragma once

#include <map>
#include <string>

#include "tensor.hpp"

// hands out named weight tensors; model loading goes through this so a
// quantized loader can slot in later without touching model code
struct WeightLoader {
    virtual ~WeightLoader() {}
    virtual bool has(const std::string& name) const = 0;
    virtual Tensor load(const std::string& name) const = 0;
};

// reads a whole fp32 weight file into memory up front
struct Fp32FileLoader : WeightLoader {
    explicit Fp32FileLoader(const std::string& path);

    bool has(const std::string& name) const override;
    Tensor load(const std::string& name) const override;

    std::map<std::string, Tensor> tensors;
};
