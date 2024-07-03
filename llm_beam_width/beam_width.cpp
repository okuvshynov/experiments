#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <common.h>
#include <llama.h>

/*
 * First, we need to generate in/out pairs.
 * 1. Prepare template (as we are going to use instruct)
 * 2. Prepare prompts (e.g. 100-1000 prompts?)
 * 3. Merge them into in[id].txt
 * 4. For each in[id].txt write corresponding out[id].tokens and out[id].txt. Tokens would be a list of ints
 *
 * 5. Then, run smaller model on same inputs. While doing that, measure the beam width effect on predicting main model output.
 */


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

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}
