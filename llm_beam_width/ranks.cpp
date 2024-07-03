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

int run(llama_model * model, llama_context * ctx, size_t idx)
{
    std::string question, prompt_template;
    std::vector<llama_token> main_tokens;
    if (   0 != r_file(q_filename(idx), &question)
        || 0 != r_file(p_filename(idx), &prompt_template)
        || 0 != r_tokens(a_filename(idx), &main_tokens))
    {
        return 1;
    }

    std::string prompt = replace(prompt_template, question);
    std::vector<llama_token> in_tokens, out_tokens;
    in_tokens = ::llama_tokenize(ctx, prompt, true);
    const auto n_vocab = llama_n_vocab(model);

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

    std::map<size_t, uint64_t> ranks;

    for (llama_token true_id : main_tokens)
    {
        auto * logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
        size_t rank = 0;
        for (llama_token token_id = 0; token_id < n_vocab; token_id++)
        {
            if (logits[token_id] > logits[true_id])
            {
                rank++;
            }
        }
        ranks[rank]++;
        fprintf(stderr, ".");
        fflush(stderr);

        llama_batch_clear(batch);
        llama_batch_add(batch, true_id, n_cur, { 0 }, true);
        // no need to do that, we always evaluate correct ones
        // llama_kv_cache_seq_rm(ctx, 0, n_cur, -1);

        n_cur += 1;

        if (llama_decode(ctx, batch))
        {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }
    }

    std::cout << std::endl << "ranks of true tokens: " << std::endl;
    for (auto it : ranks)
    {
        std::cout << "  n_rank[" << it.first << "] = " << it.second << std::endl;
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
        llama_kv_cache_clear(ctx);
        if (0 != run(model, ctx, input_index++))
        {
            break;
        }
    }

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}
