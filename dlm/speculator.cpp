#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>

#include <common.h>
#include <llama.h>
#include <nlohmann/json.hpp>
#include <zmq.hpp>

#include "config.h"
#include "query_context.h"
#include "utils.h"

namespace
{

using json = nlohmann::json;

// instead of receiving a copy of request it should 'attach' itself to the main node?
class speculator
{
  public:
    static std::unique_ptr<speculator> create(config conf);
    virtual ~speculator();
    int loop();

  private:
    explicit speculator(config conf);

    bool call(llama_tokens & curr, size_t & n_matched);

    llama_model   * model_;
    const config    conf_;

    // current context, as we operate on one query at a time
    query_context query_ctx_;
};

std::unique_ptr<speculator> speculator::create(config conf)
{
    auto self = std::unique_ptr<speculator>(new speculator(conf));

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers       = conf.n_gpu_layers;

    self->model_ = llama_load_model_from_file(conf.model_path.c_str(), model_params);

    if (self->model_ == nullptr)
    {
        fprintf(stderr, "Model %s load failed.\n", conf.model_path.c_str());
        return nullptr;
    }

    return self;
}

speculator::speculator(config conf): conf_(conf)
{
}

speculator::~speculator()
{
    if (model_ != nullptr)
    {
        llama_free_model(model_);
    }
}

// returns true if main model completed the generation
bool speculator::call(llama_tokens & curr, /* OUT */ size_t & n_matched)
{
    json req_j;
    req_j["spec"] = curr;

    auto req_z = json_to_zmsg(req_j);
    query_ctx_.client->send(req_z, zmq::send_flags::none);

    zmq::message_t res_z;
    query_ctx_.client->recv(res_z, zmq::recv_flags::none);

    auto res_j = json_from_zmsg(res_z);

    bool done = res_j["done"].get<bool>();
    if (done)
    {
        return true;
    }
    n_matched = res_j["n_matched"].get<size_t>();
    curr      = res_j["spec"].get<llama_tokens>();
    return false;
}

int speculator::loop()
{
    using namespace std::chrono_literals;

    zmq::context_t zmq_ctx(1);
    zmq::socket_t socket(zmq_ctx, ZMQ_REQ);
    socket.connect(conf_.attach_to);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_batch   = conf_.n_batch;
    ctx_params.n_ctx     = conf_.n_ctx;
    ctx_params.n_threads = conf_.n_threads;

    llama_context * ctx = llama_new_context_with_model(model_, ctx_params);
    llama_batch   batch = llama_batch_init(conf_.n_batch, 0, 1);
    llama_tokens   curr; // empty 
    size_t n_matched = 0;

    while (true)
    {
        // get work/reconcile
        if (call(curr, n_matched))
        {
            fprintf(stderr, "done.\n");
            n_matched = 0;
            curr.clear();

            std::this_thread::sleep_for(100ms);
            continue;
        }

        if (curr.size() == 0)
        {
            std::this_thread::sleep_for(100ms);
            continue;
        }

        if (n_matched == curr.size())
        {
            // TODO: double-check if this can ever happen
            // at least one input
            n_matched -= 1;
        }

        llama_kv_cache_seq_rm(ctx, 0, n_matched, -1);
        llama_batch_clear(batch);
        for (size_t i = n_matched; i < curr.size(); i++)
        {
            llama_batch_add(batch, curr[i], i, { 0 }, true);
        }

        if (llama_decode(ctx, batch) != 0)
        {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            continue;
        }
        auto next_tokens = greedy_tokens(model_, ctx, batch.n_tokens - 1, batch.n_tokens);
        if (next_tokens.size() != 1)
        {
            fprintf(stderr, "invalid next tokens size\n");
            continue;
        }

        curr.push_back(next_tokens[0]);
    }

    llama_batch_free(query_ctx_.batch);
    llama_free(query_ctx_.llama_ctx);
    socket.close();
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

    auto sp = speculator::create(conf);
    if (sp == nullptr)
    {
        fprintf(stderr, "Unable to create speculator\n");
        res = 1;
    }
    else
    {
        res = sp->loop();
    }

    llama_backend_free();

    return res;
}
