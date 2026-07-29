"""Benchmark one full Hugging Face causal LM on one machine.

Usage example:
    uv run python tools/benchmark_single_model.py --model HuggingFaceTB/SmolLM2-360M-Instruct --max-new-tokens 100
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from collections.abc import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from split_lm.context_benchmark import (
    build_context_ids, parse_context_sizes, write_context_outputs)
from split_lm.metrics import (
    GenerationMetrics, append_metrics, build_generation_metrics, format_metrics)


DTYPES: dict[str, object] = {
    "auto": "auto",
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


class TokenTimingStreamer:
    """Capture one timestamp for each token emitted by `generate`."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter):
        self.clock = clock
        self.received_prompt = False
        self.token_times: list[float] = []

    def put(self, value: torch.Tensor) -> None:
        if not self.received_prompt:
            self.received_prompt = True
            return
        self.token_times.append(self.clock())

    def end(self) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=DTYPES.keys(), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--prompt", default="Write a poem about distributed computers.")
    parser.add_argument("--ignore-eos", action="store_true", help="force generation to use the full token budget")
    parser.add_argument("--metrics-file", default="benchmark-results/generation.jsonl")
    parser.add_argument("--context-sizes", default=None,
                        help="exact increasing contexts, ex 128,256,512")
    parser.add_argument("--benchmark-output", default="benchmark-results/context-sweep-local")
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


def _measure_generation(model, tokenizer, inputs: torch.Tensor,
                        args: argparse.Namespace) -> tuple[torch.Tensor, GenerationMetrics]:
    streamer = TokenTimingStreamer()
    start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=None if args.ignore_eos or args.context_sizes else tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )
    finished = time.perf_counter()
    generated = output[0, inputs.shape[1]:]
    metrics = build_generation_metrics(
        "local", args.model, int(inputs.shape[1]), start, finished,
        streamer.token_times, workers=0)
    append_metrics(args.metrics_file, metrics)
    return generated, metrics


def _print_measurement(generated: torch.Tensor, tokenizer,
                       metrics: GenerationMetrics) -> None:
    print(tokenizer.decode(generated, skip_special_tokens=True))
    print(format_metrics(metrics, color=sys.stdout.isatty()))


def _run_context_sweep(args: argparse.Namespace, tokenizer, model) -> None:
    maximum = int(model.config.max_position_embeddings) - args.max_new_tokens
    assert isinstance(args.context_sizes, str)
    contexts = parse_context_sizes(args.context_sizes, maximum)
    results: list[GenerationMetrics] = []
    csv_path = svg_path = None
    for context in contexts:
        print(f"\nBENCHMARK LOCAL context={context} generated_max={args.max_new_tokens}")
        inputs = build_context_ids(tokenizer, context, args.device)
        generated, metrics = _measure_generation(model, tokenizer, inputs, args)
        _print_measurement(generated, tokenizer, metrics)
        results.append(metrics)
        csv_path, svg_path = write_context_outputs(results, args.benchmark_output)
    assert csv_path is not None and svg_path is not None
    print(f"BENCHMARK outputs: {csv_path} {svg_path}")


def main() -> None:
    args = parse_args()
    tokenizer, model, load_seconds = load_model(args.model, args.device, args.dtype)
    print(f"MODEL {args.model}")
    print(f"DEVICE {args.device} dtype={args.dtype}")
    print(f"LOAD {load_seconds:.3f}s peak_rss={max_rss_mb():.1f} MiB")
    if args.context_sizes:
        _run_context_sweep(args, tokenizer, model)
        return
    inputs = prompt_ids(tokenizer, args.prompt, args.device)
    generated, metrics = _measure_generation(model, tokenizer, inputs, args)

    _print_measurement(generated, tokenizer, metrics)


if __name__ == "__main__":
    main()
