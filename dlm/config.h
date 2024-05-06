#pragma once

#include <cstdint>
#include <string>

struct config
{
    std::string bind_address; // can be any zmq endpoint, e.g. inproc://llm_1

    std::string model_path;
    uint32_t n_batch;
    uint32_t n_ctx;
    uint32_t n_threads;
    uint32_t n_gpu_layers;
};

