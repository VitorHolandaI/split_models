"""Build `llama-bench` invocations for the questions this project asks.

llama-bench already sweeps: every test parameter accepts a comma-separated
list, it repeats each point, warms up first, and prints CSV. Writing a sweep
here would mean reimplementing that worse -- an earlier attempt did exactly
that, parsing llama-cli output with a regex.

What is left to own is which grid to sweep and why, which is the part specific
to this project.
"""

from __future__ import annotations

from collections.abc import Sequence

LLAMA_BENCH = "llama-bench"

# llama-bench separa os dispositivos de uma razao com barra e as razoes entre si
# com virgula: "100/0,90/10". Diferente do llama-cli, que usa virgula dentro da
# razao. Trocar os dois faz o benchmark medir uma configuracao que nao e a
# pedida, sem erro nenhum.
DEFAULT_RATIOS = ("100/0", "90/10", "75/25", "50/50")

# Profundidade e o KV cache ja preenchido antes de medir. E o eixo em que a
# frota soma capacidade: os pesos sao fixos, o cache cresce com o contexto.
DEFAULT_DEPTHS = (0, 4096, 16384)


def _joined(values: Sequence[object]) -> str:
    return ",".join(str(value) for value in values)


def bench_command(model_path: str, endpoints: str = "",
                  ratios: Sequence[str] = (),
                  depths: Sequence[int] = DEFAULT_DEPTHS,
                  generated: int = 64, prompt: int = 512,
                  repetitions: int = 3) -> list[str]:
    """Build one llama-bench sweep over ratios and context depths.

    Ratios only make sense with remote workers, so they are dropped when there
    are none; llama-bench would otherwise measure a split that cannot happen.

    Usage example:
        `bench_command("m.gguf", "10.0.0.2:50052", ("100/0", "90/10"))`.
    """
    remote = ["--rpc", endpoints] if endpoints else []
    split = ["-ts", _joined(ratios)] if endpoints and ratios else []
    return [LLAMA_BENCH, "-m", model_path, *remote, *split,
            "-sm", "layer", "-ngl", "99",
            "-d", _joined(depths), "-n", str(generated), "-p", str(prompt),
            "-r", str(repetitions), "-o", "csv"]


def device_speed_command(model_path: str, endpoints: str = "") -> list[str]:
    """Build the shortest run that ranks a machine, for picking the master.

    The master samples every token and therefore needs the logits locally, so
    the fastest machine should hold that role; anything else pays a round trip
    per token. One depth, few repetitions, small generation -- enough to rank,
    cheap enough to run at startup.

    Usage example: `device_speed_command("m.gguf")`.
    """
    return bench_command(model_path, endpoints, ratios=(), depths=(0,),
                         generated=32, prompt=128, repetitions=2)
