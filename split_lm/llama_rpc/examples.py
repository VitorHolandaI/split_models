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
    """One GGUF repository worth trying, with its download size and reason."""

    repo: str
    quant: str
    gigabytes: float
    note: str

    @property
    def needs_split(self) -> bool:
        """Whether the weights alone leave no room on one 8 GB node.

        The threshold is deliberately below 8: the OS, the RPC server and the
        KV cache all take from the same budget.
        """
        return self.gigabytes > 6.0

    def fits_master(self, usable_gigabytes: float) -> bool:
        """Whether the master can hold this file, which caps the whole fleet.

        Splitting does not lift this. The master downloads the entire GGUF and
        mmaps it, and on a stateless appliance that cache is tmpfs, so the file
        sits in the master's RAM for as long as the model is loaded. Workers
        receive only their layers over RPC and are never the binding limit.

        Usage example: `model.fits_master(12.0)`.
        """
        return self.gigabytes <= usable_gigabytes - _MASTER_OVERHEAD_GB


# KV cache, runtime do llama.cpp e o proprio sistema saem do mesmo orcamento.
_MASTER_OVERHEAD_GB = 2.0


# Sizes read from the HuggingFace API, not estimated, and every
# repository confirmed ungated -- these download with no account and no token.
# The list brackets the fleet on purpose: models that fit one node, models that
# barely do, and models that cannot, which is the whole argument for the split.
KNOWN_MODELS = (
    GgufModel(DEFAULT_GGUF_REPO, "Q4_K_M", 0.53,
              "perfil small oficial da appliance"),
    GgufModel("unsloth/Qwen3.5-4B-GGUF", "Q4_K_M", 2.74,
              "geracao atual; cabe folgado em um no de 8 GB"),
    GgufModel("ggml-org/gemma-4-E4B-it-GGUF", "Q4_0", 4.59,
              "familia Gemma, conversao oficial do ggml; ainda cabe em 8 GB"),
    GgufModel("unsloth/Qwen3.5-9B-GGUF", "Q6_K", 7.46,
              "teto seguro num master de 12 GB; Q6 e indistinguivel do Q8"),
    GgufModel("unsloth/Ornith-1.0-9B-GGUF", "Q6_K", 7.48,
              "outra familia no mesmo teto, para comparar qualidade"),
    GgufModel("unsloth/gemma-4-12b-it-GGUF", "Q5_K_M", 8.41,
              "modelo maior no mesmo orcamento, trocando quantizacao por parametros"),
    GgufModel("ggml-org/gpt-oss-20b-GGUF", "MXFP4", 12.11,
              "passa de um master de 12 GB; serve para ver o limite estourar"),
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
             "Qualquer repositorio GGUF serve, de qualquer familia: Qwen, Gemma,",
             "Ornith, gpt-oss. Os abaixo foram conferidos e baixam sem conta e",
             "sem token. Tamanho e do arquivo de fato, lido da API em 2026-08-01,",
             "somando os shards quando o modelo vem partido.",
             "",
             "Nome de repositorio inexistente responde 401 'Invalid username or",
             "password': parece erro de login e nao e. Confira o nome em",
             "huggingface.co antes de culpar a rede."]
    for model in KNOWN_MODELS:
        fits = "PRECISA DE MAIS DE UM NO" if model.needs_split else "cabe em um no"
        lines.extend([
            f"\n{model.repo}  [{model.gigabytes:.2f} GB {model.quant}] {fits}",
            f"  {model.note}",
            f"  SPLIT_LLAMA_MODEL={model.repo} /root/split-llama-master.sh",
        ])
    return lines


def _master_limit_examples() -> list[str]:
    return [
        "", "O QUE LIMITA O MODELO MAXIMO",
        "  O master baixa o GGUF inteiro e da mmap nele. A appliance e",
        "  stateless, entao esse cache e tmpfs: o arquivo fica na RAM do master",
        "  enquanto o modelo estiver carregado. Somar nos NAO levanta esse teto,",
        "  porque os workers so recebem as layers deles pela rede.",
        "",
        "  teto do modelo = RAM utilizavel do master - 2 GB (KV cache, runtime, OS)",
        "",
        "  Num master com 12 GB utilizaveis, o maior arquivo e ~10 GB.",
    ]


def _quant_examples() -> list[str]:
    return [
        "", "ESCOLHER A QUANTIZACAO",
        "  Sufixo :quant depois do repositorio. Menor quantizacao, menos RAM e",
        "  menos banda de memoria por token, ao custo de qualidade. Como cada",
        "  token rele todos os pesos, o tamanho do arquivo e o teto de tok/s.",
        "    SPLIT_LLAMA_MODEL=unsloth/Qwen3.5-9B-GGUF:Q4_K_M \\",
        "      /root/split-llama-master.sh",
        "  Sem sufixo o llama.cpp escolhe sozinho.",
    ]


def format_llama_examples() -> str:
    """Render runnable worker, master, and model commands.

    Usage example: `print(format_llama_examples())`.
    """
    return "\n".join([*_general_examples(), *_model_examples(),
                      *_master_limit_examples(), *_quant_examples()])
