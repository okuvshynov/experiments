#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>

#include <common.h>
#include <llama.h>
#include <nlohmann/json.hpp>
#include <zmq.hpp>

#include <zmq.h>

#include "config.h"
#include "utils.h"

namespace
{

using json = nlohmann::json;

using llama_tokens = std::vector<llama_token>;

// returns true if main model completed the generation
// any call might reset the state to new query (e.g. change n_matched to 0)
bool call(zmq::socket_t * client, llama_tokens & curr, /* out */ size_t & n_matched, /* out */ size_t & n_len)
{
    json req_j;
    req_j["spec"] = curr;

    auto req_z = json_to_zmsg(req_j);
    client->send(req_z, zmq::send_flags::none);

    zmq::message_t res_z;
    client->recv(res_z, zmq::recv_flags::none);

    auto res_j = json_from_zmsg(res_z);

    bool done = res_j["done"].get<bool>();
    if (done)
    {
        return true;
    }
    n_matched = res_j["n_matched"].get<size_t>();
    n_matched = res_j["n_len"].get<size_t>();
    curr      = res_j["spec"].get<llama_tokens>();
    return false;
}

int loop(config conf)
{
    using namespace std::chrono_literals;

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers       = conf.n_gpu_layers;

    llama_model * model = llama_load_model_from_file(conf.model_path.c_str(), model_params);
    if (model == nullptr)
    {
        return 1;
    }

    zmq::context_t zmq_ctx(1);
    zmq::socket_t socket(zmq_ctx, ZMQ_REQ);
    try
    {
        socket.set(zmq::sockopt::sndtimeo, 1000);
        socket.set(zmq::sockopt::rcvtimeo, 1000);
        socket.connect(conf.attach_to);
    }
    catch (const zmq::error_t& e)
    {
        fprintf(stderr, "zeromq error: %s\n", e.what());
        llama_free_model(model);
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_batch   = conf.n_batch;
    ctx_params.n_ctx     = conf.n_ctx;
    ctx_params.n_threads = conf.n_threads;

    llama_context * llama_ctx = llama_new_context_with_model(model, ctx_params);
    llama_batch         batch = llama_batch_init(conf.n_batch, 0, 1);
    
    llama_tokens curr; // empty 
    size_t n_matched = 0;
    size_t n_len     = 0; // we'll populate this from the server

    while (true)
    {
        // get work/reconcile
        if (call(&socket, curr, n_matched, n_len))
        {
            fprintf(stderr, "done.\n");
            n_matched = 0;
            curr.clear();

            std::this_thread::sleep_for(100ms);
            continue;
        }

        if (curr.size() == 0 || curr.size() >= n_len)
        {
            std::this_thread::sleep_for(100ms);
            continue;
        }

        llama_kv_cache_seq_rm(llama_ctx, 0, n_matched, -1);
        if (n_matched == curr.size())
        {
            n_matched -= 1;
        }

        llama_batch_clear(batch);
        for (size_t i = n_matched; i < curr.size(); i++)
        {
            llama_batch_add(batch, curr[i], i, { 0 }, false);
        }
        batch.logits[batch.n_tokens - 1] = true;

        if (llama_decode(llama_ctx, batch) != 0)
        {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            continue;
        }
        auto next_tokens = greedy_tokens(model, llama_ctx, batch.n_tokens - 1, batch.n_tokens);
        if (next_tokens.size() != 1)
        {
            fprintf(stderr, "invalid next tokens size\n");
            continue;
        }

        curr.push_back(next_tokens[0]);
    }

    llama_batch_free(batch);
    llama_free(llama_ctx);
    socket.close();
    llama_free_model(model);
    return 0;
}

}

int main(int argc, char ** argv)
{
    int res = 0;
    llama_backend_init();
    config conf =
    {
        /* bind_address = */ "",
        /* attach_to    = */ "",

        /* model_path   = */ argv[1],
        /* n_batch      = */ 512,
        /* n_ctx        = */ 4096,
        /* n_threads    = */ 16,
        /* n_gpu_layers = */ 0
    };
    parser p;
    p.parse_options(argc, argv, conf);

    res = loop(conf);

    llama_backend_free();

    return res;
}
