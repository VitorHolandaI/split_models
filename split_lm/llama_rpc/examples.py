"""Runnable `split-llama` commands, printed by `split-llama exemplos`.

Unlike the PyTorch path, which is pinned to the few models the split planner
understands, llama.cpp reads any GGUF. The point of this list is to make that
reach obvious without sending the operator to guess repository names: a name
that does not exist answers 401 "Invalid username or password", which reads
like an auth failure and is not one.

Every repository below was confirmed to resolve on huggingface.co.
"""

from __future__ import annotations

from dataclasses import dataclass

from .master import DEFAULT_GGUF_REPO


@dataclass(frozen=True)
class GgufModel:
    """One GGUF repository worth trying, with the reason it is listed."""

    repo: str
    note: str


# Sizes chosen to bracket the fleet: one that fits anywhere, one that needs the
# split to be worth running, and one that will not fit on a single 8 GB node.
KNOWN_MODELS = (
    GgufModel(DEFAULT_GGUF_REPO, "padrao da appliance; cabe em qualquer no"),
    GgufModel("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "ainda cabe sozinho; util para comparar"),
    GgufModel("Qwen/Qwen2.5-3B-Instruct-GGUF", "mesmo modelo dos benchmarks PyTorch"),
    GgufModel("Qwen/Qwen2.5-7B-Instruct-GGUF", "exige o split; nao cabe em 8 GB sozinho"),
    GgufModel("Qwen/Qwen3-4B-GGUF", "geracao mais nova, mesmo caminho de execucao"),
    GgufModel("google/gemma-3-1b-it-qat-q4_0-gguf", "outra familia, para provar que nao e so Qwen"),
)


def _general_examples() -> list[str]:
    return [
        "COMANDOS GERAIS",
        "  split-llama worker                    sobe este no como worker RPC",
        "  split-llama nodes                     lista os workers da LAN",
        "  /root/split-llama-master.sh           promove este no a master",
        "  llama-cli --list-devices              dispositivos locais",
        "",
        "MODO DE EXECUCAO",
        "  No console o llama-cli abre em conversa: /exit, /regen, /clear.",
        "  Sem terminal (ssh sem tty, script) roda um turno so e sai.",
        "  Forcar um turno so no console:",
        "    /root/split-llama-master.sh -- -st -p 'Explique memoria RAM.' -n 40",
    ]


def _model_examples() -> list[str]:
    lines = ["", "MODELOS GGUF", "",
             "Qualquer repositorio GGUF serve. Nome inexistente responde 401",
             "'Invalid username or password', que parece erro de login e nao e.",
             "Confira em huggingface.co antes de culpar a rede."]
    for model in KNOWN_MODELS:
        lines.extend([
            f"\n{model.repo}",
            f"  {model.note}",
            f"  SPLIT_LLAMA_MODEL={model.repo} /root/split-llama-master.sh",
        ])
    return lines


def _quant_examples() -> list[str]:
    return [
        "", "ESCOLHER A QUANTIZACAO",
        "  Sufixo :quant depois do repositorio. Menor quantizacao, menos RAM e",
        "  menos banda de memoria por token, ao custo de qualidade.",
        "    SPLIT_LLAMA_MODEL=Qwen/Qwen2.5-7B-Instruct-GGUF:q4_k_m \\",
        "      /root/split-llama-master.sh",
        "  Sem sufixo o llama.cpp escolhe sozinho.",
    ]


def format_llama_examples() -> str:
    """Render runnable worker, master, and model commands.

    Usage example: `print(format_llama_examples())`.
    """
    return "\n".join([*_general_examples(), *_model_examples(), *_quant_examples()])
