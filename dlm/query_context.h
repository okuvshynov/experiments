#pragma once

#include <string>
#include <vector>

#include <common.h>
#include <llama.h>
#include <nlohmann/json.hpp>
#include <zmq.hpp>

using llama_tokens = std::vector<llama_token>;

struct query
{
    std::string prompt;
    size_t      n_predict; // new tokens to produce
};

struct spec_context
{
    llama_tokens speculation;
    std::mutex mtx;
    bool done = false;
};

struct query_context
{
    query           q;
    llama_context * llama_ctx;
    zmq::socket_t * client; // where to call 'expert'
    llama_batch     batch;
    spec_context    spec_ctx;
    size_t          n_len;
};

