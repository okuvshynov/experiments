#pragma once

#include <string>
#include <vector>

#include <common.h>
#include <llama.h>
#include <nlohmann/json.hpp>

using llama_tokens = std::vector<llama_token>;

struct query
{
    std::string prompt;
    size_t      n_predict; // new tokens to produce
};

// mutable part of context which might be accessed by 
// multiple threads. Protected by single mutex for simplicify.
struct spec_context
{
    llama_tokens speculation;
    bool done = false;
    std::mutex mtx;
};

struct output_context
{
    bool done = false;
    std::string             output;
    std::mutex              mtx;
};

struct query_context
{
    query           q;
    llama_context * llama_ctx;
    llama_batch     batch;
    spec_context    spec_ctx;
    size_t          n_len;
    output_context  out_ctx;
};
