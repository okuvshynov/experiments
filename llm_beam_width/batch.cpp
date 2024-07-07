#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// llama.cpp
#include <common.h>
#include <llama.h>

#include "shared.h"

const int32_t batch_size = 10;
const int32_t seq_size   = 10;

int run(llama_model * model, llama_context * ctx, size_t idx, size_t n_predict)
{
    std::string question, prompt_template;
    if (   0 != r_file(q_filename(idx), &question)
        || 0 != r_file(p_filename(idx), &prompt_template))
    {
        return 1;
    }

    const auto n_vocab = llama_n_vocab(model);

    std::string prompt = replace(prompt_template, question);
    std::vector<llama_token> in_tokens;
    in_tokens = ::llama_tokenize(ctx, prompt, true);

    for (auto id : in_tokens)
    {
        fprintf(stderr, "%s", llama_token_to_piece(ctx, id).c_str());
    }
    llama_batch batch = llama_batch_init(512 * batch_size, 0, batch_size);

    for (size_t i = 0; i < in_tokens.size(); i++)
    {
        bool do_logits = (i + 1 == in_tokens.size());
        for (int32_t j = 0; j < batch_size; j++)
        {
            llama_batch_add(batch, in_tokens[i], i, { j }, do_logits);
        }
    }

    if (llama_decode(ctx, batch) != 0)
    {
        fprintf(stderr, "%s: llama_decode() failed\n", __func__);
        return 1;
    }

    size_t n_cur  = in_tokens.size();
    int logit_idx = batch.n_tokens - 1;

    auto started = ggml_time_us();
    while (n_cur <= n_predict)
    {
        auto logits = llama_get_logits_ith(ctx, logit_idx);
        llama_token id = greedy(logits, n_vocab);
        if (llama_token_is_eog(model, id) || n_cur == n_predict)
        {
            fprintf(stderr, "\n");
            break;
        }

        std::string id_str = llama_token_to_piece(ctx, id);

        fprintf(stderr, "%s", id_str.c_str());
        fflush(stderr);

        llama_batch_clear(batch);
        for (int32_t i = 0; i < batch_size; i++)
        {
            for (int32_t j = 0; j < seq_size; j++)
            {
                llama_batch_add(batch, id, n_cur + j, { i }, true);
            }
        }

        n_cur += 1;

        if (llama_decode(ctx, batch))
        {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }

        // clear the cache for fake entries in all sequences
        llama_kv_cache_seq_rm(ctx, -1, n_cur, -1);
        logit_idx = 0;
    }
    double dur_s = (ggml_time_us() - started) * 1.0e-6;

    double evps = (n_cur - in_tokens.size()) / dur_s;
    fprintf(stderr, "batch_sz = %d, seq_sz = %d, evps = %.3lf\n", batch_size, seq_size, evps);
    fflush(stderr);

    return 0;
}

int main(int argc, char ** argv)
{
    gpt_params params;

    if (gpt_params_parse(argc, argv, params) == false)
    {
        return 1;
    }

    if (params.seed == LLAMA_DEFAULT_SEED)
    {
        params.seed = time(NULL);
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // main model and context
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    std::tie(model, ctx) = llama_init_from_gpt_params(params);

    size_t input_index = 0;
    while (true)
    {
        fprintf(stderr, "#####\nprocessing %zu\n#####\n", input_index);
        llama_kv_cache_clear(ctx);
        if (0 != run(model, ctx, input_index++, params.n_predict))
        {
            break;
        }
    }

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}
