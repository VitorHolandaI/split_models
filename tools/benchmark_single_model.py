"""Benchmark one full Hugging Face causal LM on one machine.

Usage example:
    uv run python tools/benchmark_single_model.py --model HuggingFaceTB/SmolLM2-360M-Instruct --max-new-tokens 100
"""

from __future__ import annotations

import argparse
import resource
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPES: dict[str, object] = {
    "auto": "auto",
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=DTYPES.keys(), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--prompt", default="Write a poem about distributed computers.")
    parser.add_argument("--ignore-eos", action="store_true", help="force generation to use the full token budget")
    return parser.parse_args()


def prompt_ids(tokenizer, prompt: str, device: str) -> torch.Tensor:
    """Return input ids using the model's chat template when available.

    Usage example:
        ids = prompt_ids(tokenizer, "Write a poem", "cpu")
    """
    if getattr(tokenizer, "chat_template", None):
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        return encoded["input_ids"].to(device)
    return tokenizer(prompt, return_tensors="pt").input_ids.to(device)


def load_model(model_name: str, device: str, dtype_name: str):
    """Load tokenizer and model, returning load time in seconds.

    Usage example:
        tokenizer, model, seconds = load_model("gpt2", "cpu", "auto")
    """
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=DTYPES[dtype_name]).to(device).eval()
    return tokenizer, model, time.perf_counter() - start


def max_rss_mb() -> float:
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss_kb / 1024.0


def main() -> None:
    args = parse_args()
    tokenizer, model, load_seconds = load_model(args.model, args.device, args.dtype)
    inputs = prompt_ids(tokenizer, args.prompt, args.device)
    eos_token_id = None if args.ignore_eos else tokenizer.eos_token_id

    start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    generate_seconds = time.perf_counter() - start

    generated = output[0, inputs.shape[1]:]
    token_count = int(generated.numel())
    tok_s = token_count / generate_seconds if generate_seconds else 0.0
    ms_tok = generate_seconds / token_count * 1000.0 if token_count else 0.0

    print(tokenizer.decode(generated, skip_special_tokens=True))
    print(f"MODEL {args.model}")
    print(f"DEVICE {args.device} dtype={args.dtype}")
    print(f"PROMPT tokens={inputs.shape[1]} generated={token_count} max_new_tokens={args.max_new_tokens}")
    print(f"LOAD {load_seconds:.3f}s peak_rss={max_rss_mb():.1f} MiB")
    print(f"GENERATE {generate_seconds:.3f}s tok/s={tok_s:.2f} ms/token={ms_tok:.1f}")


if __name__ == "__main__":
    main()
