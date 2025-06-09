# Copyright © 2023-2024 Apple Inc.

import argparse
import contextlib
import functools
import json
import sys
import time
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_reduce
from transformers import PreTrainedTokenizer

from mlx_lm.models import cache
from mlx_lm.models.cache import QuantizedKVCache
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import load


def prepare_prompt(args, tokenizer) -> List[int]:
    """
    Prepare the prompt tokens based on command line arguments.
    
    Args:
        args: Command line arguments
        tokenizer: The tokenizer to use for encoding
        
    Returns:
        List[int]: The prepared prompt tokens
    """
    # Get the base prompt
    base_prompt = args.prompt.replace("\\n", "\n").replace("\\t", "\t")
    base_prompt = sys.stdin.read() if base_prompt == "-" else base_prompt
    
    # Tokenize the base prompt
    base_tokens = tokenizer.encode(base_prompt)
    
    # Repeat tokens to reach desired length
    target_length = args.prompt_length
    if len(base_tokens) > 0:
        repeats = (target_length + len(base_tokens) - 1) // len(base_tokens)
        prompt = base_tokens * repeats
        prompt = prompt[:target_length]
    else:
        prompt = []
    
    return prompt


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description="LLM inference script")
    parser.add_argument(
        "--model",
        type=str,
        help="The path to the local model directory or Hugging Face repo.",
        default="mlx-community/Llama-3.2-3B-Instruct-4bit",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        default="The quick brown fox jumps over the lazy dog. ",
        help="Base prompt text to repeat ('-' reads from stdin)",
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=1000,
        help="Target number of tokens for the prompt",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        help="Set the maximum key-value cache size",
        default=None,
    )
    parser.add_argument(
        "--kv-bits",
        type=int,
        help="Number of bits for KV cache quantization. "
        "Defaults to no quantization.",
        default=None,
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        help="Group size for KV cache quantization.",
        default=64,
    )
    parser.add_argument(
        "--quantized-kv-start",
        help="When --kv-bits is set, start quantizing the KV cache "
        "from this step onwards.",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--prefill-step-size",
        help="Number of tokens to process at once during prompt prefill",
        type=int,
        default=2048,
    )
    return parser


# A stream on the default device just for generation
generation_stream = mx.new_stream(mx.default_device())


@contextlib.contextmanager
def wired_limit(model: nn.Module, streams: Optional[List[mx.Stream]] = None):
    """
    A context manager to temporarily change the wired limit.

    Note, the wired limit should not be changed during an async eval.  If an
    async eval could be running pass in the streams to synchronize with prior
    to exiting the context manager.
    """
    model_bytes = tree_reduce(
        lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, model, 0
    )
    max_rec_size = mx.metal.device_info()["max_recommended_working_set_size"]
    if model_bytes > 0.9 * max_rec_size:
        model_mb = model_bytes // 2**20
        max_rec_mb = max_rec_size // 2**20
        print(
            f"[WARNING] Generating with a model that requires {model_mb} MB "
            f"which is close to the maximum recommended size of {max_rec_mb} "
            "MB. This can be slow. See the documentation for possible work-arounds: "
            "https://github.com/ml-explore/mlx-lm/tree/main#large-models"
        )
    old_limit = mx.set_wired_limit(max_rec_size)
    try:
        yield None
    finally:
        if streams is not None:
            for s in streams:
                mx.synchronize(s)
        else:
            mx.synchronize()
        mx.set_wired_limit(old_limit)


@dataclass
class GenerationResponse:
    """
    The output of :func:`stream_generate`.

    Args:
        text (str): The next segment of decoded text. This can be an empty string.
        token (int): The next token.
        prompt_tokens (int): The number of tokens in the prompt.
        prompt_tps (float): The prompt processing tokens-per-second.
        generation_tokens (int): The number of generated tokens.
        generation_tps (float): The tokens-per-second for generation.
        peak_memory (float): The peak memory used so far in GB.
    """

    text: str
    token: int
    prompt_tokens: int
    prompt_tps: float
    generation_tokens: int
    generation_tps: float
    peak_memory: float


def maybe_quantize_kv_cache(prompt_cache, quantized_kv_start, kv_group_size, kv_bits):
    if (
        kv_bits is not None
        and not isinstance(prompt_cache[0], cache.QuantizedKVCache)
        and prompt_cache[0].offset > quantized_kv_start
    ):
        for i in range(len(prompt_cache)):
            if isinstance(prompt_cache[i], cache.KVCache):
                prompt_cache[i] = prompt_cache[i].to_quantized(
                    group_size=kv_group_size, bits=kv_bits
                )


def generate_step(
    prompt: mx.array,
    model: nn.Module,
    *,
    max_tokens: int = 256,
    sampler: Optional[Callable[mx.array, mx.array]] = None,
    max_kv_size: Optional[int] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 2048,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
) -> Generator[mx.array, None, None]:
    """
    A generator producing token ids based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        max_tokens (int): The maximum number of tokens. Use``-1`` for an infinite
          generator. Default: ``256``.
        sampler (Callable[mx.array, mx.array], optional): A sampler for sampling a
          token from a vector of log probabilities. Default: ``None``.
        max_kv_size (int, optional): Maximum size of the key-value cache. Old
          entries (except the first 4 tokens) will be overwritten.
        prompt_cache (List[Any], optional): A pre-computed prompt cache. Note, if
          provided, the cache will be updated in place.
        prefill_step_size (int): Step size for processing the prompt.
        kv_bits (int, optional): Number of bits to use for KV cache quantization.
          None implies no cache quantization. Default: ``None``.
        kv_group_size (int): Group size for KV cache quantization. Default: ``64``.
        quantized_kv_start (int): Step to begin using a quantized KV cache.
           when ``kv_bits`` is non-None. Default: ``0``.

    Yields:
        mx.array: One token.
    """

    # Create the KV cache for generation
    if prompt_cache is None:
        prompt_cache = cache.make_prompt_cache(
            model,
            max_kv_size=max_kv_size,
        )
    elif len(prompt_cache) != len(model.layers):
        raise ValueError("Wrong number of layers in the prompt cache.")


    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    sampler = sampler or make_sampler(0.0)

    def _model_call(y):
        return model(y, cache=prompt_cache)

    def _step(y):
        with mx.stream(generation_stream):
            logits = _model_call(y[None])
            logits = logits[:, -1, :]
            quantize_cache_fn(prompt_cache)
            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            y = sampler(logprobs)
            return y

    y = prompt
    with mx.stream(generation_stream):
        total_prompt_tokens = y.shape[0]
        prompt_processed_tokens = 0
        while y.shape[0] > prefill_step_size:
            _model_call(y[:prefill_step_size][None])
            quantize_cache_fn(prompt_cache)
            mx.eval([c.state for c in prompt_cache])
            prompt_processed_tokens += prefill_step_size
            y = y[prefill_step_size:]
            mx.clear_cache()

        y = _step(y)

    mx.async_eval(y)
    n = 0
    while True:
        if n != max_tokens:
            next_y = _step(y)
            mx.async_eval(next_y)
        if n == 0:
            mx.eval(y)
        if n == max_tokens:
            break
        yield y.item()
        if n % 256 == 0:
            mx.clear_cache()
        y = next_y
        n += 1


def stream_generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: List[int],
    **kwargs,
) -> Generator[GenerationResponse, None, None]:
    """
    A generator producing text based on the given prompt from the model.

    Args:
        model (nn.Module): The model to use for generation.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        prompt (List[int]): The input prompt as integer tokens.
        kwargs: The remaining options get passed to :func:`generate_step`.
          See :func:`generate_step` for more details.

    Yields:
        GenerationResponse: An instance containing the generated text segment and
            associated metadata. See :class:`GenerationResponse` for details.
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    prompt = mx.array(prompt)

    detokenizer = tokenizer.detokenizer

    token_generator = generate_step(prompt, model, **kwargs)
    with wired_limit(model, [generation_stream]):
        detokenizer.reset()
        tic = time.perf_counter()
        for n, token in enumerate(token_generator):
            if n == 0:
                prompt_time = time.perf_counter() - tic
                prompt_tps = prompt.size / prompt_time
                tic = time.perf_counter()

            detokenizer.add_token(token)

            yield GenerationResponse(
                text=detokenizer.last_segment,
                token=token,
                prompt_tokens=prompt.size,
                prompt_tps=prompt_tps,
                generation_tokens=n + 1,
                generation_tps=(n + 1) / (time.perf_counter() - tic),
                peak_memory=mx.get_peak_memory() / 1e9,
            )

        detokenizer.finalize()
        yield GenerationResponse(
            text=detokenizer.last_segment,
            token=token,
            prompt_tokens=prompt.size,
            prompt_tps=prompt_tps,
            generation_tokens=n + 1,
            generation_tps=(n + 1) / (time.perf_counter() - tic),
            peak_memory=mx.get_peak_memory() / 1e9,
        )


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    tokenizer_config = {"trust_remote_code" : True}

    model, tokenizer = load(
        args.model,
        tokenizer_config=tokenizer_config,
    )

    prompt = prepare_prompt(args, tokenizer)

    print("=" * 10)
    text = ""
    for response in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=args.max_tokens,
        max_kv_size=args.max_kv_size,
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        quantized_kv_start=args.quantized_kv_start,
        prefill_step_size=args.prefill_step_size,
    ):
        print(response.text, end="", flush=True)
        text += response.text
    
    print()
    print("=" * 10)
    if len(text) == 0:
        print("No text generated for this prompt")
    else:
        print(
            f"Prompt: {response.prompt_tokens} tokens, "
            f"{response.prompt_tps:.3f} tokens-per-sec"
        )
        print(
            f"Generation: {response.generation_tokens} tokens, "
            f"{response.generation_tps:.3f} tokens-per-sec"
        )
        print(f"Peak memory: {response.peak_memory:.3f} GB")


if __name__ == "__main__":
    main()
