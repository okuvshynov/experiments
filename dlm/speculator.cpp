#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>

#include <common.h>
#include <llama.h>
#include <nlohmann/json.hpp>
#include <httplib.h>

#include "config.h"
#include "utils.h"

namespace
{

struct config
{
    std::string host;
    int32_t     port;

    std::string model_path;
    uint32_t n_batch;
    uint32_t n_ctx;
    uint32_t n_threads;
    uint32_t n_gpu_layers;
};

config gen_config(int argc, char ** argv)
{
    config res = 
    {
        /* host = */ "localhost",
        /* port = */ 5555,

        /* model_path   = */ "",
        /* n_batch      = */ 512,
        /* n_ctx        = */ 4096,
        /* n_threads    = */ 16,
        /* n_gpu_layers = */ 0
    };
    parser<config> p;
    // main server endpoint to connect to
    p.add_option({"--host", "-h"},                             &config::host);
    p.add_option({"--port", "-p"},                             &config::port);

    // llama options
    p.add_option({"--model", "-m"},                            &config::model_path);
    p.add_option({"--batch_size", "--batch-size", "-b"},       &config::n_batch);
    p.add_option({"--n_ctx", "--n-ctx", "-c"},                 &config::n_ctx);
    p.add_option({"--threads", "-t"},                          &config::n_threads);
    p.add_option({"--n_gpu_layers", "--n-gpu-layers", "-ngl"}, &config::n_gpu_layers);
    
    p.parse_options(argc, argv, res);

    return res;
}

using json = nlohmann::json;

using llama_tokens = std::vector<llama_token>;

// returns true if main model completed the generation
// any call might reset the state to new query (e.g. change n_matched to 0)
bool call(httplib::Client * client, llama_tokens & curr, /* out */ size_t & n_matched, /* out */ size_t & n_len)
{
    try
    {
        json req_j;
        req_j["spec"] = curr;
        auto res = client->Post("/hint", req_j.dump(), "application/json");
        if (res)
        {
            json res_j = json::parse(res->body);
            bool done = res_j["done"].get<bool>();
            if (done)
            {
                return true;
            }
            n_matched = res_j["n_matched"].get<size_t>();
            n_len     = res_j["n_len"].get<size_t>();
            curr      = res_j["spec"].get<llama_tokens>();
            return false;
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
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

    httplib::Client http_client(conf.host, conf.port);
    http_client.set_keep_alive(true);

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
        if (call(&http_client, curr, n_matched, n_len))
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
    llama_free_model(model);
    return 0;
}

}

int main(int argc, char ** argv)
{
    int res = 0;
    llama_backend_init();
    config conf = gen_config(argc, argv);

    res = loop(conf);

    llama_backend_free();

    return res;
}
