#include <cstdint>
#include <cstdio>
#include <memory>
#include <sstream>
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

using llama_tokens = std::vector<llama_token>;

// mutable part of context which might be accessed by 
// multiple threads. Protected by single mutex for simplicify.
struct spec_context
{
    llama_tokens speculation;
    bool done = false;
    std::mutex mtx;
};

struct output_context
{
    bool done = false;
    std::string             output;
    std::mutex              mtx;
};

struct query_context
{
    std::string     prompt;
    size_t          n_predict; // new tokens to produce
    llama_context * llama_ctx;
    llama_batch     batch;
    spec_context    spec_ctx;
    size_t          n_len;
    output_context  out_ctx;
};

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
        /* host = */ "0.0.0.0",
        /* port = */ 5555,

        /* model_path   = */ "",
        /* n_batch      = */ 512,
        /* n_ctx        = */ 4096,
        /* n_threads    = */ 16,
        /* n_gpu_layers = */ 0
    };
    parser<config> p;
    p.add_option({"--host", "-h"},                             &config::host);
    p.add_option({"--port", "-p"},                             &config::port);

    // llama options
    p.add_option({"--model", "-m"},                            &config::model_path);
    p.add_option({"--batch_size", "--batch-size", "-b"},       &config::n_batch);
    p.add_option({"--n_ctx", "--n-ctx", "-c"},                 &config::n_ctx);
    p.add_option({"--threads", "-t"},                          &config::n_threads);
    p.add_option({"--n_gpu_layers", "--n-gpu-layers", "-ngl"}, &config::n_gpu_layers);

    if (0 != p.parse_options(argc, argv, res))
    {
        exit(-1);
    }

    return res;
}

using json = nlohmann::json;

std::string llama3_strip_eot(const std::string& str)
{
    const std::string tag = "<|eot_id|>";
    size_t len = str.size();
    while (len >= tag.size() && str.substr(len - tag.size(), tag.size()) == tag)
    {
        len -= tag.size();
    }
    return str.substr(0, len);
}

std::string llama3_instruct_fmt_msg(const json& j)
{
    std::ostringstream oss;
    oss << "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n";
    oss << j.value("system", "") << "<|eot_id|>\n";

    for (const auto& msg: j["messages"])
    {
        oss << "<|start_header_id|>" << msg["role"].get<std::string>() << "<|end_header_id|>\n\n";
        oss << msg["content"].get<std::string>() << "<|eot_id|>";
    }

    oss << "<|start_header_id|>assistant<|end_header_id|>";
    return oss.str();
}

class llama_node
{
  public:
    static std::unique_ptr<llama_node> create(config conf);
    virtual ~llama_node();
    void serve();

  private:
    explicit llama_node(config conf);

    int  generate(const llama_tokens & tokens_list);
    void setup_http();

    llama_model   * model_;
    const config    conf_;

    // current context, as we operate on one query at a time
    query_context   query_ctx_;
    httplib::Server http_serv_;
};

std::unique_ptr<llama_node> llama_node::create(config conf)
{
    auto self = std::unique_ptr<llama_node>(new llama_node(conf));

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers       = conf.n_gpu_layers;

    self->model_ = llama_load_model_from_file(conf.model_path.c_str(), model_params);

    if (self->model_ == nullptr)
    {
        fprintf(stderr, "Unable to load model from %s\n", conf.model_path.c_str());
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

void llama_node::setup_http()
{
    http_serv_.Post("/hint", [this](const httplib::Request & req, httplib::Response & res)
    {
        try
        {
            json res_j;
            auto req_j = json::parse(req.body);

            llama_tokens remote_spec = req_j["spec"];
            {
                std::lock_guard<std::mutex> _lock(query_ctx_.spec_ctx.mtx);
                if (query_ctx_.spec_ctx.done)
                {
                    res_j["done"] = true;
                    query_ctx_.spec_ctx.speculation.clear();
                } 
                else
                {
                    auto& spec = query_ctx_.spec_ctx.speculation;
                    bool match = true;
                    size_t n_matched = remote_spec.size();
                    for (size_t i = 0; i < std::min(spec.size(), remote_spec.size()); i++)
                    {
                        if (spec[i] != remote_spec[i])
                        {
                            match = false;
                            n_matched = i;
                            break;
                        }
                    }
                    if (match && spec.size() < remote_spec.size())
                    {
                        spec = remote_spec;
                    }
                    else
                    {
                        remote_spec = spec;
                    }
                    res_j["done"]      = false;
                    res_j["spec"]      = remote_spec;
                    res_j["n_matched"] = n_matched;
                    res_j["n_len"]     = query_ctx_.n_len;
                }
            }
            res.set_content(res_j.dump(), "application/json");
        }
        catch(const std::exception & e)
        {
            std::cerr << e.what() << '\n';
        }
    });
    http_serv_.Post("/messages", [this](const httplib::Request & req, httplib::Response & res)
    {
        // we process one message at a time anyway.
        std::lock_guard<std::mutex> _lock(query_ctx_.out_ctx.mtx);
        try
        {
            auto req_j = json::parse(req.body);

            query_ctx_.prompt    = llama3_instruct_fmt_msg(req_j);
            query_ctx_.n_predict = static_cast<size_t>(req_j.value("max_tokens", 1024));
            const auto t_start = ggml_time_us();

            llama_context_params ctx_params = llama_context_default_params();
            ctx_params.n_batch   = conf_.n_batch;
            ctx_params.n_ctx     = conf_.n_ctx;
            ctx_params.n_threads = conf_.n_threads;
            query_ctx_.llama_ctx = llama_new_context_with_model(model_, ctx_params);

            dbg_not_matched(query_ctx_.prompt);

            auto prompt = llama_tokenize(query_ctx_.llama_ctx, query_ctx_.prompt, true);

            query_ctx_.batch = llama_batch_init(conf_.n_batch, 0, 1);
            query_ctx_.n_len = query_ctx_.n_predict + prompt.size();
            query_ctx_.out_ctx.output = "";
            query_ctx_.out_ctx.done   = false;

            // Init speculation context
            {
                std::lock_guard<std::mutex> _lock(query_ctx_.spec_ctx.mtx);
                query_ctx_.spec_ctx.speculation = prompt;
                query_ctx_.spec_ctx.done        = false;
            }

            if (generate(prompt) != 0)
            {
                fprintf(stderr, "generation failed\n");
            }

            const auto t_end = ggml_time_us();
            printf("Processing time: %.3lf\n", (t_end - t_start) / 1000000.0);
            auto output = llama3_strip_eot(query_ctx_.out_ctx.output);
            
            json res_j = { {"content",  {{"text", output}}} };
            res.set_content(res_j.dump(), "application/json");

            llama_batch_free(query_ctx_.batch);
            llama_free(query_ctx_.llama_ctx);
        }
        catch(const std::exception & e)
        {
            std::cerr << e.what() << '\n';
        }
    });
} 

void llama_node::serve()
{
    setup_http();
    http_serv_.listen(conf_.host, conf_.port);
}

int llama_node::generate(const llama_tokens & tokens_list)
{
    llama_context * ctx   = query_ctx_.llama_ctx;
    llama_batch   & batch = query_ctx_.batch; 

    // evaluate the initial prompt
    auto bsz = conf_.n_batch;
    for (size_t i = 0; i < tokens_list.size();)
    {
        llama_batch_clear(batch);
        size_t j;
        for (j = 0; j < bsz && i + j < tokens_list.size(); j++)
        {
            llama_batch_add(batch, tokens_list[i + j], i + j, { 0 }, false);
        }
        if (i + j == tokens_list.size())
        {
            batch.logits[batch.n_tokens - 1] = true;
        }
        if (llama_decode(ctx, batch) != 0)
        {
            fprintf(stderr, "%s: llama_decode() failed\n", __func__);
            return 1;
        }
        i += j;
    }

    // how many tokens are currently accepted
    int n_cur  = tokens_list.size();

    llama_tokens input_seq, next_tokens;
    input_seq.push_back(tokens_list.back());

    int logits_from = batch.n_tokens - 1;
    int logits_to   = batch.n_tokens;
    const auto t_start = ggml_time_us();
    while (n_cur < query_ctx_.n_len)
    {
        next_tokens = greedy_tokens(model_, ctx, logits_from, logits_to);
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
        // append next_tokens to the output
        std::string next_str;
        for (llama_token tok : next_tokens)
        {
            next_str += llama_token_to_piece(ctx, tok);
        }
        query_ctx_.out_ctx.output += next_str;


        // empty the non-match portion of kv cache
        llama_kv_cache_seq_rm(ctx, 0, n_cur - 1, -1);

        bool done = false;
        for (size_t i = 0; i < next_tokens.size(); i++)
        {
            // TODO: what should we do here
            if (next_tokens[i] == llama_token_eos(model_) || next_tokens[i] == 128009)
            {
                done = true;
                next_tokens.erase(next_tokens.begin() + i, next_tokens.end());
                break;
            }
        }
        if (n_cur >= query_ctx_.n_len || done)
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
            dbg_accepted(accepted);
            if (n_match != next_tokens.size())
            {
                std::string rejected = "";
                for (size_t i = next_tokens_pos + n_match; i < spec.size(); i++)
                {
                    rejected += llama_token_to_piece(ctx, spec[i]);
                }
                dbg_rejected(rejected);
                std::string not_matched = "";
                for (size_t i = n_match; i < next_tokens.size(); i++)
                {
                    not_matched += llama_token_to_piece(ctx, next_tokens[i]);
                }
                dbg_not_matched(not_matched);
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
        if (input_seq.size() + n_cur > query_ctx_.n_len)
        {
            input_seq.resize(query_ctx_.n_len - n_cur);
        }
        // TODO: in some cases this might be not the most efficient thing to do
        // for correctness just make the input size <= batch size
        if (input_seq.size() > bsz)
        {
            fprintf(stderr, "warn: trimming speculation to fit in batch size\n");
            input_seq.resize(bsz);
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

    std::string next_str = "";
    for (size_t i = 0; i < next_tokens.size(); i++)
    {
        auto sp = llama_token_to_piece(ctx, next_tokens[i]);
        dbg_not_matched(sp);
        // not very efficient
        next_str += sp;
    }
    query_ctx_.out_ctx.output += next_str;
    query_ctx_.out_ctx.done = true;
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

}

int main(int argc, char ** argv)
{
    llama_backend_init();

    config conf = gen_config(argc, argv);

    auto node = llama_node::create(conf);
    if (node != nullptr)
    {
        node->serve();
    }

    llama_backend_free();

    return 0;
}
