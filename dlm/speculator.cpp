// ZeroMQ
#include <zmq.hpp>
#include <zmq.h>

// json
#include <nlohmann/json.hpp>

// llama.cpp
#include <llama.h>
#include <common.h>

// std
#include <memory>
#include <string>
#include <vector>

#include "utils.h"

namespace {

using json = nlohmann::json;

struct config
{
    std::string bind_address; // can be any zmq endpoint, e.g. inproc://llm_1

    std::string model_path;
    uint32_t n_threads;
    uint32_t n_gpu_layers;
};

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
};

class speculator
{
  public:
    static std::unique_ptr<speculator> create(config conf);
    ~speculator();
    int serve();

  private:
    explicit speculator(config conf);
    json handle_request(const json & j);
    void eval_loop();
    int  speculate(const std::vector<llama_token> & tokens_list);
    bool merge_speculation(
            std::vector<llama_token> & local_spec,
            size_t                   & match_len);

    zmq::context_t zmq_context_;
    void * zmq_ctx_;
    mt_queue<query> queue_;
    llama_model * model_;
    const config conf_;

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

    self->zmq_ctx_ = zmq_ctx_new();
    if (self->zmq_ctx_ == nullptr)
    {
        fprintf(stderr, "zmq_ctx_new failed.\n");
        return nullptr;
    }
    return self;
}

speculator::speculator(config conf): zmq_context_(1), conf_(conf)
{
}

speculator::~speculator()
{
    if (model_ != nullptr)
    {
        llama_free_model(model_);
    }
    if (zmq_ctx_ != nullptr)
    {
        if (zmq_ctx_destroy(zmq_ctx_) != 0)
        {
            fprintf(stderr, "zmq_ctx_destroy failed\n");
        }
    }
}

int speculator::serve()
{
    zmq::socket_t socket(zmq_context_, ZMQ_REP);
    socket.bind(conf_.bind_address);

    std::thread eval_thread([this]() { eval_loop(); });

    while (true)
    {
        zmq::message_t req_z;
        socket.recv(req_z, zmq::recv_flags::none);

        auto req_j = json_from_zmsg(req_z);
        auto res_j = handle_request(req_j);
        auto res_z = json_to_zmsg(res_j);
        socket.send(res_z, zmq::send_flags::none);
    }

    eval_thread.join();
    return 0;
}

json speculator::handle_request(const json & j)
{
    json res;
    res["status"] = "ok";
    if (j.contains("prompt"))
    {
        query q = 
        {
            j["prompt"],
            j["expert"],
        };
        queue_.push(q);
        return res;
    }

    return res;
}

bool speculator::merge_speculation(
    std::vector<llama_token> & local_spec,
    size_t                   & match_len)
{
    json req_j;
    req_j["spec"] = local_spec;

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
    match_len  = res_j["match_len"].get<size_t>();
    local_spec = res_j["spec"].get<std::vector<llama_token>>();
    return false;
}

// Continuous speculation on single prompt
// TODO: if running indefinitely, this will get out of bounds
int speculator::speculate(const std::vector<llama_token> & tokens_list)
{
    llama_context * ctx = query_ctx_.llama_ctx;

    // TODO: this gets leaked in case of error
    llama_batch batch = llama_batch_init(512, 0, 1);

    for (size_t i = 0; i < tokens_list.size(); i++)
    {
        llama_batch_add(batch, tokens_list[i], i, { 0 }, false);
    }

    // llama_decode will output logits only for the last token of the prompt
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0)
    {
        fprintf(stderr, "%s: llama_decode() failed\n", __func__);
        return 1;
    }

    int logit_idx = batch.n_tokens - 1;
    std::vector<llama_token> local_spec = tokens_list;
    size_t match_len = 0;

    while (true)
    {
        auto next_tokens = greedy_tokens(model_, ctx, logit_idx, logit_idx + 1);
        if (next_tokens.size() != 1)
        {
            fprintf(stderr, "invalid next tokens\n");
            return 1;
        }

        local_spec.push_back(next_tokens[0]);

        // TODO: this is doing a query every time. 
        // TODO: pass delta only
        if (merge_speculation(local_spec, match_len))
        {
            break;
        }
        llama_kv_cache_seq_rm(ctx, 0, match_len, -1);

        llama_batch_clear(batch);
        for (size_t i = match_len; i < local_spec.size(); i++)
        {
            llama_batch_add(batch, local_spec[i], i, { 0 }, true);
        }

        logit_idx = batch.n_tokens - 1;

        if (llama_decode(ctx, batch))
        {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }
    }

    llama_batch_free(batch);
    return 0;
}

void speculator::eval_loop()
{
    while (true)
    {
        query_ctx_.q = queue_.pop();

        const auto t_start = ggml_time_us();
        llama_context_params ctx_params = llama_context_default_params();

        // TODO: configure these as well
        // ctx_params.seed  = 1234;
        // ctx_params.n_threads_batch = 16;

        ctx_params.n_ctx = 2048;
        ctx_params.n_threads = conf_.n_threads;
        query_ctx_.llama_ctx = llama_new_context_with_model(model_, ctx_params);

        auto tokens_list = llama_tokenize(query_ctx_.llama_ctx, query_ctx_.q.prompt, true);

        // connection to expert
        zmq::socket_t socket(zmq_context_, ZMQ_REQ);
        socket.connect(query_ctx_.q.expert);

        query_ctx_.client = &socket;

        if (speculate(tokens_list) != 0)
        {
            fprintf(stderr, "speculation failed\n");
            // TODO: error
        }

        llama_free(query_ctx_.llama_ctx);
        socket.close();
        const auto t_end = ggml_time_us();
        printf("Processing time: %.3lf\n", (t_end - t_start) / 1000000.0);
    }
}

}

int main(int argc, char ** argv)
{
    int res = 0;
    llama_backend_init();
    config conf =
    {
        /* bind_address = */ "tcp://*:5566",

        /* model_path   = */ argv[1],
        /* n_threads    = */ 16,
        /* n_gpu_layers = */ 0
    };
    auto sp = speculator::create(conf);
    if (sp == nullptr)
    {
        fprintf(stderr, "Unable to create speculator\n");
        res = 1;
    }
    else
    {
        res = sp->serve();
    }

    llama_backend_free();

    return res;
}
