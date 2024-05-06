// ZeroMQ
#include <zmq.hpp>

// json
#include <nlohmann/json.hpp>

// llama.cpp
#include <llama.h>
#include <common.h>

// std
#include <string>
#include <vector>

#include "utils.h"

namespace {

using json = nlohmann::json;

struct config
{
    std::string bind_address; // can be any zmq protocol, e.g. inproc://llm_1

    std::string model_path;
    uint32_t n_threads;
    uint32_t n_gpu_layers;
};

struct query_context
{
    std::string prompt;
    std::string expert; // speculator will communicate with expert

    llama_context * llama_ctx;
    zmq::socket_t * client;
};

class speculator
{
  public:
    // TODO: this should take some config as argument?
    explicit speculator(config conf);
    ~speculator();
    int serve();

  private:
    json handle_request(const json & j);
    void eval_loop();
    int speculate(const std::vector<llama_token> & tokens_list);
    bool merge_speculation(
            std::vector<llama_token>   & local_spec,
            size_t                     & match_len);

    zmq::context_t zmq_context_;
    mt_queue<query_context> queue_;
    llama_model * model_;
    const config conf_;
    query_context * qctx_ = nullptr;
};

speculator::speculator(config conf): zmq_context_(1), conf_(conf)
{
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = conf_.n_gpu_layers;
    model_ = llama_load_model_from_file(conf_.model_path.c_str(), model_params);

    if (model_ == nullptr)
    {
        // TODO: fail
    }
}

speculator::~speculator()
{
    llama_free_model(model_);
    llama_backend_free();
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
        query_context q = 
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
        std::vector<llama_token>   & local_spec,
        size_t                     & match_len)
{
    json req_j;
    req_j["spec"] = local_spec;

    auto req_z = json_to_zmsg(req_j);
    qctx_->client->send(req_z, zmq::send_flags::none);

    zmq::message_t res_z;
    qctx_->client->recv(res_z, zmq::recv_flags::none);

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
    llama_context * ctx = qctx_->llama_ctx;

    // TODO: this gets leaked in case of error
    llama_batch batch = llama_batch_init(512, 0, 1);

    // evaluate the initial prompt
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

        if (merge_speculation(local_spec, match_len)) {
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
        query_context qctx = queue_.pop();

        const auto t_start = ggml_time_us();
        llama_context_params ctx_params = llama_context_default_params();

        // TODO: configure these as well
        // ctx_params.seed  = 1234;
        // ctx_params.n_threads_batch = 16;

        ctx_params.n_ctx = 2048;
        ctx_params.n_threads = conf_.n_threads;
        qctx.llama_ctx = llama_new_context_with_model(model_, ctx_params);

        this->qctx_ = &qctx;

        auto tokens_list = llama_tokenize(qctx.llama_ctx, qctx.prompt, true);

        // connection to expert
        zmq::socket_t socket(zmq_context_, ZMQ_REQ);
        socket.connect(qctx_->expert);

        qctx_->client = &socket;


        if (speculate(tokens_list) != 0)
        {
            // TODO: error
        }

        llama_free(qctx.llama_ctx);
        socket.close();
        const auto t_end = ggml_time_us();
        printf("Processing time: %.3lf\n", (t_end - t_start) / 1000000.0);
    }
}

}

int main(int argc, char ** argv)
{
    config conf =
    {
        /* bind_address = */ "tcp://*:5566",

        /* model_path   = */ argv[1],
        /* n_threads    = */ 16,
        /* n_gpu_layers = */ 0
    };
    speculator spec(conf);
    return spec.serve();
}
