"""Build the llama-cli invocation that spans the discovered RPC workers."""

from __future__ import annotations

from collections.abc import Sequence

LLAMA_CLI = "llama-cli"
# Official smoke profile for the appliance. The quant suffix avoids upstream
# selection changing when a repository adds another GGUF.
DEFAULT_GGUF_REPO = "unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M"


def list_devices_command(endpoints: str) -> list[str]:
    """Build the command that shows local and remote devices before inference.

    Worth running first on a new fleet: it is the only cheap way to see that a
    worker contributed the accelerator you expected instead of falling back to
    its CPU.

    Usage example: `command = list_devices_command("192.168.0.11:50052")`.
    """
    return [LLAMA_CLI, "--rpc", endpoints, "--list-devices"]


def llama_cli_command(model_repo: str, endpoints: str,
                      extra_args: Sequence[str] = (),
                      interactive: bool = True) -> list[str]:
    """Build the master command, with layer splitting across the fleet.

    `--split-mode layer` gives each device a contiguous range of layers, which
    is the mode that matches this project's pipeline split. `-ngl auto` and
    `--fit on` let llama.cpp size the offload against the memory each device
    actually reports, instead of a number guessed here.

    `--tensor-split` is intentionally left unset. The target GPUs share system
    memory with the host, so a static ratio would be wrong on any machine but
    the one it was measured on.

    `interactive` False adds `-st`, which runs one turn and exits. llama-cli
    opens a chat UI by default, so a run with no terminal attached waits on
    stdin forever -- measured spinning a core at 93% once RPC was involved.
    The caller decides from whether it has a terminal, so the physical console
    still gets a chat and a scripted or SSH run still terminates.

    Usage example:
        `command = llama_cli_command(DEFAULT_GGUF_REPO, "10.0.0.2:50052")`.
    """
    single_turn = [] if interactive else ["-st"]
    return [LLAMA_CLI, "-hf", model_repo, "--rpc", endpoints,
            "--split-mode", "layer", "-ngl", "auto", "--fit", "on",
            *single_turn, *extra_args]
