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

struct linear_speculative_context
{
    std::vector<llama_token> speculation;
    std::mutex mtx;
    bool done;
};

linear_speculative_context spec_ctx;

std::string parse_request(const zmq::message_t& request)
{
    const std::string req_str(static_cast<const char*>(request.data()), request.size());

    // TODO: this might throw
    json j = json::parse(req_str);

    // new prompt
    if (j.contains("prompt")) {
        std::string prompt = j["prompt"];
        // enqueue here
        prompt_queue.push(prompt);
        return req_str;
    }
    if (j.contains("spec")) {
        std::vector<llama_token> local_spec = j["spec"];
        // process speculation and return speculation result
        // this should be equivalent to what was done in spec thread
        {
            json res;
            std::lock_guard<std::mutex> _lock(spec_ctx.mtx);
            if (spec_ctx.done)
            {
                res["done"] = true;
            } 
            else
            {
                res["done"] = false;
                auto& spec = spec_ctx.speculation;
                bool match = true;
                size_t match_len = local_spec.size() - 1;
                for (size_t i = 0; i < std::min(spec.size(), local_spec.size()); i++)
                {
                    if (spec[i] != local_spec[i])
                    {
                        match = false;
                        match_len = i;
                        // llama_kv_cache_seq_rm(ctx, 0, i, -1);
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
                res["spec"] = local_spec;
                res["match_len"] = match_len;
            }
            std::string res_str = res.dump();

            return res_str;
        }
    }
    return req_str;
}

int serve_loop()
{
    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_REP);
    // TODO: configure this
    socket.bind("tcp://*:5555");
    while (true)
    {
        zmq::message_t request;

        // Wait for the next request from the client
        socket.recv(request, zmq::recv_flags::none);

        auto res_str = parse_request(request);
        zmq::message_t response(res_str.size());
        memcpy(response.data(), res_str.data(), res_str.size());

        // sending back same thing
        socket.send(response, zmq::send_flags::none);
    }
    return 0;
}

int eval_prompt(
        llama_model                    * model,
        llama_context                  * ctx,
        const std::vector<llama_token> & tokens_list)
{
    const int n_len = 1024;

    llama_batch batch = llama_batch_init(1024, 0, 1);

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

    // how many tokens are currently accepted
    int n_cur  = batch.n_tokens;

    std::vector<llama_token> input_seq, next_tokens;
    input_seq.push_back(tokens_list.back());

    int logits_from = n_cur - 1;
    int logits_to   = n_cur;
    size_t bg_index = 0;

    while (n_cur <= n_len)
    {
        bg_index++;
        next_tokens = greedy_tokens(model, ctx, logits_from, logits_to);
        if (next_tokens.size() != input_seq.size())
        {
            fprintf(stderr, "invalid next tokens\n");
            return 1;
        }

        // this is where next_tokens start
        int next_tokens_pos = n_cur;
        // we always accept at least one new token
        n_cur += 1;
        for (size_t i = 0; i + 1 < input_seq.size(); i++)
        {
            if (next_tokens[i] == input_seq[i + 1])
            {
                n_cur += 1;
            }
            else
            {
                // reject. next_tokens[i] is the last 'correct' one.
                next_tokens.erase(next_tokens.begin() + i + 1, next_tokens.end());
                break;
            }
        }
        // empty the main model cache
        llama_kv_cache_seq_rm(ctx, 0, n_cur - 1, -1);

        bool done = false;
        for (llama_token new_token_id: next_tokens)
        {
            if (new_token_id == llama_token_eos(model))
            {
                done = true;
            }
        }
        if (n_cur >= n_len || done)
        {
            break;
        }

        // CRITICAL SECTION -- reconcile main and speculative
        {
            std::lock_guard<std::mutex> _lock(spec_ctx.mtx);
            auto & spec = spec_ctx.speculation;
            size_t n_match = 0;
            for (size_t i = 0; i < next_tokens.size() && i + next_tokens_pos < spec.size(); i++)
            {
                if (next_tokens[i] == spec[i + next_tokens_pos])
                {
                    n_match++;
                }
                else
                {
                    break;
                }
            }

            std::string accepted = "";
            // Write accepted/rejected/not matched
            // this is slow and inefficient but for short strings doesn't matter 
            for (size_t i = next_tokens_pos; i < next_tokens_pos + n_match; i++)
            {
                accepted += llama_token_to_piece(ctx, spec[i]);
            }
            dbg_accepted(accepted, bg_index);
            if (n_match != next_tokens.size())
            {
                std::string rejected = "";
                for (size_t i = next_tokens_pos + n_match; i < spec.size(); i++)
                {
                    rejected += llama_token_to_piece(ctx, spec[i]);
                }
                dbg_rejected(rejected, bg_index);
                // need to modify speculation
                spec.erase(spec.begin() + next_tokens_pos, spec.end());
                for (const auto tok: next_tokens)
                {
                    spec.push_back(tok);
                }
                std::string not_matched = "";
                for (size_t i = n_match; i < next_tokens.size(); i++)
                {
                    not_matched += llama_token_to_piece(ctx, next_tokens[i]);
                }
                dbg_not_matched(not_matched, bg_index);
            }

            input_seq.assign(spec.begin() + n_cur - 1, spec.end());
        }

        llama_batch_clear(batch);
        for (size_t i = 0; i < input_seq.size(); i++)
        {
            llama_batch_add(batch, input_seq[i], n_cur - 1 + i, { 0 }, true);
        }
        if (llama_decode(ctx, batch))
        {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }
        logits_from = 0;
        logits_to = input_seq.size();
    }

    for (size_t i = 0; i < next_tokens.size(); i++)
    {
        dbg_not_matched(llama_token_to_piece(ctx, next_tokens[i]), bg_index);
    }
    std::cout << std::endl << std::endl;
    {
        std::lock_guard<std::mutex> _lock(spec_ctx.mtx);
        spec_ctx.done = true;
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

        // Init shared speculative context
        spec_ctx.speculation = tokens_list;
        spec_ctx.done = false;

        eval_prompt(model, ctx, tokens_list);

        llama_free(ctx);
        const auto t_end = ggml_time_us();
        printf("Processing time: %.3lf\n", (t_end - t_start) / 1000000.0);
    }
}

int main(int argc, char ** argv)
{
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;
    llama_model * model = llama_load_model_from_file(argv[1], model_params);

    std::thread t_eval(eval_loop, model);
    std::thread t_serve(serve_loop);

    t_eval.join();
    llama_free_model(model);
    llama_backend_free();

    t_serve.join();

    return 0;
}
