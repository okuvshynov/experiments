// ZeroMQ
#include <zmq.hpp>

// json
#include <nlohmann/json.hpp>

// llama.cpp
#include <llama.h>
#include <common.h>

#include <string>

#include "utils.h"

using json = nlohmann::json;

mt_queue<std::string> prompt_queue;
zmq::context_t context(1);
zmq::socket_t main_socket(context, ZMQ_REQ);

json handle_request(const json & j)
{
    // new prompt
    json res;
    res["status"] = "ok";
    if (j.contains("prompt")) {
        std::string prompt = j["prompt"];
        // enqueue here
        prompt_queue.push(prompt);
    }
    return res;
}

int serve_loop()
{
    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_REP);
    // TODO: configure this
    socket.bind("tcp://*:5566");
    while (true)
    {
        zmq::message_t req_z;
        socket.recv(req_z, zmq::recv_flags::none);

        auto req_j = json_from_zmsg(req_z);
        auto res_j = handle_request(req_j);
        auto res_z = json_to_zmsg(res_j);
        socket.send(res_z, zmq::send_flags::none);
    }
    return 0;
}

// returns true/false if completed
bool merge_speculation(
        llama_context              * ctx,
        std::vector<llama_token>   & local_spec,
        size_t                     & match_len)
{
    json req_j;
    req_j["spec"] = local_spec;

    auto req_z = json_to_zmsg(req_j);
    main_socket.send(req_z, zmq::send_flags::none);

    zmq::message_t res_z;
    main_socket.recv(res_z, zmq::recv_flags::none);

    auto res_j = json_from_zmsg(res_z);

    bool done = res_j["done"].get<bool>();
    if (done)
    {
        return true;
    }
    match_len  = res_j["match_len"].get<size_t>();
    local_spec = res_j["spec"].get<std::vector<llama_token>>();
    llama_kv_cache_seq_rm(ctx, 0, match_len, -1);
    return false;
}

// TODO: if running indefinitely, this will get out of bounds
static int speculate(
        llama_model * model,
        llama_context * ctx,
        std::vector<llama_token> tokens_list /* copy here */)
{
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
        LOG_TEE("%s: llama_decode() failed\n", __func__);
        return 1;
    }

    int logit_idx = batch.n_tokens - 1;
    std::vector<llama_token> local_spec = tokens_list;
    size_t match_len = 0;

    while (true)
    {
        auto next_tokens = greedy_tokens(model, ctx, logit_idx, logit_idx + 1);
        if (next_tokens.size() != 1)
        {
            fprintf(stderr, "invalid next tokens\n");
            return 1;
        }

        local_spec.push_back(next_tokens[0]);

        if (merge_speculation(ctx, local_spec, match_len)) {
            break;
        }

        llama_batch_clear(batch);
        for (size_t i = match_len; i < local_spec.size(); i++)
        {
            llama_batch_add(batch, local_spec[i], i, { 0 }, true);
        }

        logit_idx = batch.n_tokens - 1;

        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch))
        {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }
    }

    llama_batch_free(batch);
    return 0;
}

int eval_loop(llama_model * model)
{
    while (true)
    {
        // blocking wait
        std::string prompt = prompt_queue.pop();

        const auto t_start = ggml_time_us();
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.seed  = 1234;
        ctx_params.n_ctx = 2048;
        ctx_params.n_threads = 16;
        ctx_params.n_threads_batch = 16;
        llama_context * ctx   = llama_new_context_with_model(model, ctx_params);

        dbg_not_matched(prompt, 0);

        auto tokens_list = llama_tokenize(ctx, prompt, true);

        speculate(model, ctx, tokens_list);

        llama_free(ctx);
        const auto t_end = ggml_time_us();
        printf("Processing time: %.3lf\n", (t_end - t_start) / 1000000.0);
    }
}

int main(int argc, char ** argv)
{
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    llama_model * model = llama_load_model_from_file(argv[1], model_params);

    // TODO configure this
    main_socket.connect("tcp://localhost:5555");
    std::thread t_eval(eval_loop, model);
    std::thread t_serve(serve_loop);

    t_eval.join();
    llama_free_model(model);
    llama_backend_free();

    t_serve.join();

    return 0;
}
