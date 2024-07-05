#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <queue>
#include <vector>

// llama.cpp
#include <common.h>
#include <llama.h>

#include "shared.h"

using llama_tokens = std::vector<llama_token>;

struct seq_context
{
    llama_tokens local_tokens;
    double       p = 1.0;
    float      * logits;
};

struct candidate
{
    size_t seq_index;
    llama_token token;
    double p;

    bool operator<(const candidate& other) const
    {
        return this->p < other.p;
    }
};

llama_token greedy(float * logits, llama_token n_vocab)
{
    llama_token res = 0;
    for (llama_token tok = 1; tok < n_vocab; tok++)
    {
        if (logits[tok] > logits[res])
        {
            res = tok;
        }
    }
    return res;
}

// beam might be more narrow originally - we expand it to beam_width here
void beam_advance(
        std::vector<seq_context>& beam,
        llama_context * ctx,
        size_t beam_width,
        llama_token n_vocab)
{
    std::vector<candidate> candidates;
    std::set<size_t> vacant, used;
    for (size_t bi = 0; bi < beam.size(); ++bi)
    {
        //std::cout << llama_token_to_piece(ctx, greedy(beam[bi].logits, n_vocab)) << " " << beam[bi].p << std::endl;
        // all slots are vacant
        vacant.insert(bi);
        double d = 0.0;
        for (llama_token token_id = 0; token_id < n_vocab; token_id++)
        {
            d += exp(double(beam[bi].logits[token_id]));
        }
        for (llama_token token_id = 0; token_id < n_vocab; token_id++)
        {
            double p = exp(double(beam[bi].logits[token_id])) / d;
            candidate cc =
            {
                bi,
                token_id,
                // TODO: this needs to be changed according to actual cost function
                beam[bi].p * p
            };

            candidates.push_back(cc);
        }
    }

    // create empty placeholders for missing sequences
    while (beam.size() < beam_width)
    {
        vacant.insert(beam.size());
        beam.push_back(seq_context{});
    }

    // not very efficient for finding top K
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(candidates.begin(), candidates.end() - beam_width);

    for (const auto& cc : candidates)
    {
        vacant.erase(cc.seq_index);
    }

    for (const auto& cc : candidates)
    {
        //std::cout << "Candidate " << llama_token_to_piece(ctx, cc.token) << " " << cc.p << " " << cc.token << std::endl;
        auto & curr = beam[cc.seq_index];
        if (used.count(cc.seq_index) == 0)
        {
            // we just continue with this sequence
            used.insert(cc.seq_index);

            curr.p = cc.p;
            curr.local_tokens.push_back(cc.token);
            continue;
        }
        // we need to copy over to one of the vacant slots
        if (vacant.size() == 0)
        {
            // sanity check, this should never happen
            fprintf(stderr, "F: vacant mismatch\n");
            exit(0);
        }
        size_t index_to = * (vacant.begin());
        vacant.erase(vacant.begin());

        // now we need to copy tokens, probability and cache
        beam[index_to].local_tokens.assign(curr.local_tokens.begin(), curr.local_tokens.end() - 1);
        beam[index_to].local_tokens.push_back(cc.token);
        beam[index_to].p = cc.p;

        // copy entire cache?
        llama_kv_cache_seq_cp(ctx, cc.seq_index, index_to, 0, -1);
    }

    // bump probabilities here, so they are not too small?
}

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

    const size_t beam_width = 2;

    llama_batch batch = llama_batch_init(512, 0, beam_width);

    std::vector<seq_context> beam;
    seq_context be;
    be.p = 1.0;
    beam.push_back(be);

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
    for (auto& be : beam)
    {
        be.logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
    }

    for (llama_token true_id : main_tokens)
    {
        beam_advance(beam, ctx, beam_width, n_vocab);

        bool has_match = false;

        for (size_t bi = 0; bi < beam_width; bi++)
        {
            //std::cout << " >> " << bi << " | ";
            //for (auto tok : beam[bi].local_tokens)
            //{
            //    std::cout << llama_token_to_piece(ctx, tok);
            //}
            //std::cout << std::endl;
            has_match = has_match || std::equal(beam[bi].local_tokens.begin(), beam[bi].local_tokens.end(), main_tokens.begin());
        }

        std::cout << n_cur  - in_tokens.size() << " true = " << llama_token_to_piece(ctx, true_id) << " has_match = " << has_match << std::endl;

        llama_batch_clear(batch);
        for (int bi = 0; bi < beam_width; bi++)
        {
            llama_batch_add(batch, beam[bi].local_tokens.back(), n_cur, { bi }, true);
        }

        n_cur += 1;

        if (llama_decode(ctx, batch))
        {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }
        // fill in logits for next iteration
        for (int bi = 0; bi < beam_width; bi++)
        {
            beam[bi].logits = llama_get_logits_ith(ctx, bi);
        }
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
