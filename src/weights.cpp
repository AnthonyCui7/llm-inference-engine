#include <cstdint>
#include <cstdio>
#include <cstring>

#include "weights.hpp"

static uint32_t read_u32(std::FILE* f) {
    uint32_t value = 0;
    size_t got = std::fread(&value, sizeof(value), 1, f);
    assert(got == 1);
    (void)got;
    return value;
}

Fp32FileLoader::Fp32FileLoader(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    assert(f != nullptr && "could not open weight file");

    char magic[4];
    size_t got = std::fread(magic, 1, 4, f);
    assert(got == 4);
    assert(std::memcmp(magic, "LLMW", 4) == 0);

    uint32_t version = read_u32(f);
    assert(version == 1);
    (void)version;

    uint32_t count = read_u32(f);
    for (uint32_t t = 0; t < count; t++) {
        uint32_t name_len = read_u32(f);
        assert(name_len > 0 && name_len < 256);

        std::string name(name_len, '\0');
        got = std::fread(&name[0], 1, name_len, f);
        assert(got == name_len);

        uint32_t ndim = read_u32(f);
        assert(ndim >= 1 && ndim <= 8);

        int shape[8] = {0};
        for (uint32_t d = 0; d < ndim; d++) {
            shape[d] = static_cast<int>(read_u32(f));
            assert(shape[d] > 0);
        }

        Tensor tensor(shape, static_cast<int>(ndim), name.c_str());
        got = std::fread(tensor.data, sizeof(float), tensor.numel(), f);
        assert(got == tensor.numel());
        (void)got;

        tensors[name] = std::move(tensor);
    }

    std::fclose(f);
}

bool Fp32FileLoader::has(const std::string& name) const {
    return tensors.find(name) != tensors.end();
}

Tensor Fp32FileLoader::load(const std::string& name) const {
    auto it = tensors.find(name);
    assert(it != tensors.end() && "unknown weight name");
    return it->second;
}
