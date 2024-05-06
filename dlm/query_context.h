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
    std::string expert; // speculator will communicate with expert
};

struct query_context
{
    query           q;
    llama_context * llama_ctx;
    zmq::socket_t * client; // where to call 'expert'
    llama_batch     batch;
};

