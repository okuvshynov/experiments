#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>

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

class llama_node
{
  public:
    static std::unique_ptr<llama_node> create(config conf);
    virtual ~llama_node();
    int serve();

  private:
    explicit llama_node(config conf);

    json handle_request(const json & j);
    void eval_loop(zmq::context_t & zmq_ctx);
    int  generate(const llama_tokens & tokens_list);

    mt_queue<query> queue_;
    llama_model   * model_;
    const config    conf_;

    // current context, as we operate on one query at a time
    query_context query_ctx_;
};

std::unique_ptr<llama_node> llama_node::create(config conf)
{
    auto self = std::unique_ptr<llama_node>(new llama_node(conf));

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

llama_node::llama_node(config conf): conf_(conf)
{
}

llama_node::~llama_node()
{
    if (model_ != nullptr)
    {
        llama_free_model(model_);
    }
}

int llama_node::serve()
{
    zmq::context_t zmq_ctx;
    zmq::socket_t socket(zmq_ctx, ZMQ_REP);
    socket.bind(conf_.bind_address);

    std::thread eval_thread([this, &zmq_ctx]() { this->eval_loop(zmq_ctx); });

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

json llama_node::handle_request(const json & j)
{
    json res;
    res["status"] = "ok";
    if (j.contains("prompt"))
    {
        query q =
        {
            j["prompt"],
            j.value("expert", "")
        };
        queue_.push(q);
        return res;
    }

    // we received evaluation request from 'child' node 
    if (j.contains("spec"))
    {
        llama_tokens local_spec = j["spec"];
        {
            std::lock_guard<std::mutex> _lock(query_ctx_.spec_ctx.mtx);
            if (query_ctx_.spec_ctx.done)
            {
                res["done"] = true;
            } 
            else
            {
                res["done"] = false;
                auto& spec = query_ctx_.spec_ctx.speculation;
                bool match = true;
                size_t n_matched = local_spec.size() - 1;
                for (size_t i = 0; i < std::min(spec.size(), local_spec.size()); i++)
                {
                    if (spec[i] != local_spec[i])
                    {
                        match = false;
                        n_matched = i;
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
                res["n_matched"] = n_matched;
            }
            return res;
        }
    }
    return res;
}

int llama_node::generate(const llama_tokens & tokens_list)
{
    llama_context * ctx   = query_ctx_.llama_ctx;
    llama_batch   & batch = query_ctx_.batch; 

    llama_model * model = model_;

    // TODO configure this one as well
    const int n_len = 1024;

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

    // how many tokens are currently accepted
    int n_cur  = batch.n_tokens;

    llama_tokens input_seq, next_tokens;
    input_seq.push_back(tokens_list.back());

    int logits_from = n_cur - 1;
    int logits_to   = n_cur;
    size_t bg_index = 0;

    const auto t_start = ggml_time_us();
    while (n_cur < n_len)
    {
        bg_index += 1;
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
        // empty the non-match portion of kv cache
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
            std::lock_guard<std::mutex> _lock(query_ctx_.spec_ctx.mtx);
            auto & spec = query_ctx_.spec_ctx.speculation;
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

            // Write accepted/rejected/not matched
            // this is slow and inefficient but for short strings doesn't matter 
            std::string accepted = "";
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
                std::string not_matched = "";
                for (size_t i = n_match; i < next_tokens.size(); i++)
                {
                    not_matched += llama_token_to_piece(ctx, next_tokens[i]);
                }
                dbg_not_matched(not_matched, bg_index);
            }

            // remove non-matched tokens
            if (n_match != next_tokens.size())
            {
                spec.erase(spec.begin() + next_tokens_pos, spec.end());
                for (const auto tok: next_tokens)
                {
                    spec.push_back(tok);
                }
            }
            input_seq.assign(spec.begin() + n_cur - 1, spec.end());
        }

        llama_batch_clear(batch);
        if (input_seq.size() + n_cur > n_len) {
            input_seq.resize(n_len - n_cur);
        }
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
        std::lock_guard<std::mutex> _lock(query_ctx_.spec_ctx.mtx);
        query_ctx_.spec_ctx.done = true;
    }
    size_t n_gen = n_cur - tokens_list.size();
    const auto duration_us = ggml_time_us() - t_start;
    double gen_tps = n_gen * 1000000.0 / duration_us;
    printf("Generation tps: %.3lf\n", gen_tps);

    return 0;
}

void llama_node::eval_loop(zmq::context_t & zmq_ctx)
{
    while (true)
    {
        query_ctx_.q = queue_.pop();

        const auto t_start = ggml_time_us();

        llama_context_params ctx_params = llama_context_default_params();
        // TODO: configure these as well
        // ctx_params.seed  = 1234;
        // ctx_params.n_threads_batch = 16;
        ctx_params.n_batch   = conf_.n_batch;
        ctx_params.n_ctx     = conf_.n_ctx;
        ctx_params.n_threads = conf_.n_threads;
        query_ctx_.llama_ctx = llama_new_context_with_model(model_, ctx_params);

        // TODO: optional printing?
        dbg_not_matched(query_ctx_.q.prompt, 0);

        auto prompt = llama_tokenize(query_ctx_.llama_ctx, query_ctx_.q.prompt, true);

        query_ctx_.batch = llama_batch_init(conf_.n_batch, 0, 1);

        // Init shared speculative context
        query_ctx_.spec_ctx.speculation = prompt;
        query_ctx_.spec_ctx.done = false;

        if (generate(prompt) != 0)
        {
            fprintf(stderr, "generation failed\n");
        }

        llama_batch_free(query_ctx_.batch);
        llama_free(query_ctx_.llama_ctx);
        const auto t_end = ggml_time_us();
        printf("Processing time: %.3lf\n", (t_end - t_start) / 1000000.0);
    }
}

}

int main(int argc, char ** argv)
{
    llama_backend_init();

    config conf =
    {
        /* bind_address = */ "tcp://*:5555",

        /* model_path   = */ argv[1],
        /* n_batch      = */ 512,
        /* n_ctx        = */ 4096,
        /* n_threads    = */ 16,
        /* n_gpu_layers = */ 99
    };

    auto node = llama_node::create(conf);
    int res = node->serve();

    llama_backend_free();

    return 0;
}
