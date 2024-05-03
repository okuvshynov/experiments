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

int serve_loop()
{
    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_REP);
    socket.bind("tcp://*:5555");
    while (true)
    {
        zmq::message_t request;

        // Wait for the next request from the client
        socket.recv(request, zmq::recv_flags::none);
        std::string req_str(static_cast<char*>(request.data()), request.size());

        // Deserialize JSON
        json j = json::parse(req_str);

        // Process JSON (e.g., multiply fields by 2)
        if (j.contains("value"))
        {
            j["value"] = j["value"].get<int>() * 2;
        }

        // Serialize JSON
        std::string response_str = j.dump();

        // Send reply back to client
        zmq::message_t reply(response_str.size());
        memcpy(reply.data(), response_str.data(), response_str.size());
        socket.send(reply, zmq::send_flags::none);
    }
    return 0;
}

struct linear_speculative_context
{
    std::vector<llama_token> speculation;
    std::mutex mtx;
    bool done;
};

static int speculation_loop(
        llama_model * model,
        linear_speculative_context * spec_ctx,
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

        {
            std::lock_guard<std::mutex> _lock(spec_ctx->mtx);
            if (spec_ctx->done)
            {
                break;
            } 
            auto& spec = spec_ctx->speculation;
            bool match = true;
            match_len = local_spec.size() - 1;
            for (size_t i = 0; i < std::min(spec.size(), local_spec.size()); i++)
            {
                if (spec[i] != local_spec[i])
                {
                    match = false;
                    match_len = i;
                    llama_kv_cache_seq_rm(ctx, 0, i, -1);
                    break;
                }
            }
            if (match && spec.size() < local_spec.size())
            {
                spec = local_spec;
            }
            else
            {
                local_spec = spec;
            }
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

int main(int argc, char ** argv) {
    gpt_params params;

    llama_backend_init();
    llama_numa_init(params.numa);

    // init context params
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.seed  = 1234;
    ctx_params.n_ctx = 2048;
    ctx_params.n_threads = params.n_threads;
    ctx_params.n_threads_batch = params.n_threads_batch == -1 ? params.n_threads : params.n_threads_batch;

    // Init main model and context
    if (argc >= 2) {
        params.model = argv[1];
    }
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    llama_model   * model = llama_load_model_from_file(params.model.c_str(), model_params);
    llama_context * ctx   = llama_new_context_with_model(model, ctx_params);

    // Print & tokenize prompt
    // tokenizer should be the same and prompt tokens should be the same
    if (argc >= 3) {
      params.prompt = argv[2];
    }
    if (params.prompt.empty()) {
        params.prompt = "What's the difference between instruction cache and data cache?";
    }
    dbg_not_matched(params.prompt, 0);
    std::vector<llama_token> tokens_list = llama_tokenize(ctx, params.prompt, true);

    // Init shared speculative context
    linear_speculative_context spec_ctx;
    spec_ctx.speculation = tokens_list;
    spec_ctx.done = false;

    const auto t_main_start = ggml_time_us();
    std::thread t_eval(speculation_loop, model, &spec_ctx, ctx, tokens_list);
    std::thread t_serve(serve_loop);
    t_eval.join();
    const auto t_main_end = ggml_time_us();

    printf("Total time: %.3lf\n", (t_main_end - t_main_start) / 1000000.0);

    llama_free_model(model);
    llama_free(ctx);
    llama_backend_free();

    t_serve.join();

    return 0;
}
