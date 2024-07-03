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

int run(llama_model * model, llama_context * ctx, llama_sampling_context * ctx_sampling, size_t idx, size_t n_predict)
{
    std::string question, prompt_template;
    if (   0 != r_file(q_filename(idx), &question)
        || 0 != r_file(p_filename(idx), &prompt_template))
    {
        return 1;
    }

    std::string prompt = replace(prompt_template, question);
    std::vector<llama_token> in_tokens, out_tokens;
    in_tokens = ::llama_tokenize(ctx, prompt, true);

    for (auto id : in_tokens)
    {
        fprintf(stderr, "%s", llama_token_to_piece(ctx, id).c_str());
    }
    llama_batch batch = llama_batch_init(512, 0, 1);

    for (size_t i = 0; i < in_tokens.size(); i++)
    {
        llama_batch_add(batch, in_tokens[i], i, { 0 }, false);
    }

    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0)
    {
        fprintf(stderr, "%s: llama_decode() failed\n", __func__);
        return 1;
    }

    size_t n_cur = batch.n_tokens;

    while (n_cur <= n_predict)
    {
        llama_token id = llama_sampling_sample(ctx_sampling, ctx, nullptr, batch.n_tokens - 1);
        llama_sampling_accept(ctx_sampling, ctx, id, true);
        if (llama_token_is_eog(model, id) || n_cur == n_predict)
        {
            fprintf(stderr, "\n");
            break;
        }

        std::string id_str = llama_token_to_piece(ctx, id);
        out_tokens.push_back(id);

        fprintf(stderr, "%s", id_str.c_str());
        fflush(stderr);

        llama_batch_clear(batch);
        llama_batch_add(batch, id, n_cur, { 0 }, true);

        n_cur += 1;

        if (llama_decode(ctx, batch))
        {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }
    }
    fflush(stderr);

    if (0 != w_file(a_filename(idx), out_tokens))
    {
        return 1;
    }

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
        llama_sampling_context * ctx_sampling = llama_sampling_init(params.sparams);
        if (0 != run(model, ctx, ctx_sampling, input_index++, params.n_predict))
        {
            break;
        }

        llama_sampling_free(ctx_sampling);
    }

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}
