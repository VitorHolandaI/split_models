"""Parse synthetic hardware scores and real small-model llama.cpp throughput."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

from .master import DEFAULT_GGUF_REPO
from .mode_store import RUNTIME_DIR

_VKPEAK_VALUE = re.compile(r"^(?P<label>[a-z0-9-]+)\s*=\s*(?P<value>[0-9.]+)",
                           re.MULTILINE)
_DEVICE = re.compile(r"^device\s*=\s*(?P<device>.+)$", re.MULTILINE)
_SEVEN_ZIP_TOTAL = re.compile(r"^Tot:\s+\d+\s+\d+\s+(?P<mips>\d+)\s*$",
                              re.MULTILINE)

log = logging.getLogger("split_lm")


class HardwareBenchmarkError(ValueError):
    """A benchmark output omitted a metric required for honest reporting."""


@dataclass(frozen=True)
class HardwareScore:
    """CPU, iGPU, and effective llama.cpp measurements for one machine."""

    device: str
    fp16_tflops: float
    int8_tops: float
    cpu_mips: int
    cpu_tokens_per_second: float
    vulkan_tokens_per_second: float

    def to_record(self) -> dict[str, str | float | int]:
        return {"device": self.device, "fp16_tflops": self.fp16_tflops,
                "int8_tops": self.int8_tops, "cpu_mips": self.cpu_mips,
                "cpu_tokens_per_second": self.cpu_tokens_per_second,
                "vulkan_tokens_per_second": self.vulkan_tokens_per_second}

    def to_json(self) -> str:
        return json.dumps(self.to_record())

    @classmethod
    def from_json(cls, text: str) -> HardwareScore:
        try:
            record = json.loads(text)
            return cls(str(record["device"]), float(record["fp16_tflops"]),
                       float(record["int8_tops"]), int(record["cpu_mips"]),
                       float(record["cpu_tokens_per_second"]),
                       float(record["vulkan_tokens_per_second"]))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise HardwareBenchmarkError(
                f"hardware-bench={text[-500:]!r}; esperado JSON de HardwareScore") from exc


def format_hardware_score(score: HardwareScore) -> str:
    """Render synthetic units honestly beside effective llama.cpp throughput."""
    return "\n".join((
        f"iGPU: {score.device}",
        f"iGPU FP16: {score.fp16_tflops:.2f} TFLOPS",
        f"iGPU INT8: {score.int8_tops:.2f} TOPS",
        f"CPU 7-Zip: {score.cpu_mips} MIPS",
        f"llama.cpp CPU: {score.cpu_tokens_per_second:.2f} tok/s",
        f"llama.cpp Vulkan: {score.vulkan_tokens_per_second:.2f} tok/s"))


def run_hardware_benchmark(*,
                           cache_dir: Path = RUNTIME_DIR / "hardware-bench-cache",
                           run: Callable[..., subprocess.CompletedProcess[str]]
                           = subprocess.run) -> HardwareScore:
    """Run vkpeak, 7-Zip, and the official small llama-bench on CPU and Vulkan.

    Machines without Vulkan or without int8-matrix support still get an honest
    CPU-only score: the missing measurements degrade to zero with a warning.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "LLAMA_CACHE": str(cache_dir)}
    try:
        seven_zip = _checked(run, ["7z", "b"])
        cpu = _checked(run, _llama_speed_command(0), environment)
        vkpeak = _optional(run, ["vkpeak", "0", "fp16-matrix,int8-matrix"])
        device, fp16, int8 = _vkpeak_or_cpu_only(vkpeak)
        vulkan_tokens = _vulkan_tokens_per_second(device, run, environment)
        return HardwareScore(device, fp16, int8, parse_7zip_mips(seven_zip),
                             parse_llama_tokens_per_second(cpu), vulkan_tokens)
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def _vulkan_tokens_per_second(
        device: str, run: Callable[..., subprocess.CompletedProcess[str]],
        environment: dict[str, str]) -> float:
    if device == "sem-vulkan":
        log.warning("backend efetivo CPU; tok/s Vulkan fica em zero")
        return 0.0
    vulkan = _optional(run, _llama_speed_command(99), environment)
    if vulkan is not None:
        return parse_llama_tokens_per_second(vulkan)
    log.warning("llama-bench Vulkan indisponivel: zerando tok/s Vulkan")
    return 0.0


def _vkpeak_or_cpu_only(vkpeak: str | None) -> tuple[str, float, float]:
    if vkpeak is None:
        log.warning("vkpeak indisponivel: medindo so CPU (sem iGPU)")
        return "sem-vulkan", 0.0, 0.0
    try:
        return parse_vkpeak(vkpeak)
    except HardwareBenchmarkError:
        log.warning("iGPU sem fp16/int8-matrix: zerando TOPS e TFLOPS")
        return "sem-vulkan", 0.0, 0.0


def _optional(run: Callable[..., subprocess.CompletedProcess[str]],
              command: list[str],
              environment: dict[str, str] | None = None) -> str | None:
    result = run(command, check=False, capture_output=True, text=True,
                 env=environment)
    return result.stdout if result.returncode == 0 else None


def _llama_speed_command(gpu_layers: int) -> list[str]:
    return ["llama-bench", "-hf", DEFAULT_GGUF_REPO,
            "-p", "128", "-n", "32", "-r", "2",
            "-ngl", str(gpu_layers), "-o", "json"]


def _checked(run: Callable[..., subprocess.CompletedProcess[str]],
             command: list[str], environment: dict[str, str] | None = None) -> str:
    result = run(command, check=False, capture_output=True, text=True,
                 env=environment)
    if result.returncode != 0:
        raise HardwareBenchmarkError(
            f"comando={' '.join(command)!r} saiu {result.returncode}; "
            f"esperado 0. stderr={result.stderr[-500:]!r}")
    return result.stdout


def parse_vkpeak(output: str) -> tuple[str, float, float]:
    """Parse vkpeak matrix values and convert giga-operations to tera-operations."""
    device_match = _DEVICE.search(output)
    metrics = {match.group("label"): float(match.group("value"))
               for match in _VKPEAK_VALUE.finditer(output)}
    required = ("fp16-matrix", "int8-matrix")
    if device_match is None or any(label not in metrics for label in required):
        raise HardwareBenchmarkError(
            f"vkpeak={output[-500:]!r}; esperado device, fp16-matrix e int8-matrix")
    return (device_match.group("device").strip(),
            metrics["fp16-matrix"] / 1000,
            metrics["int8-matrix"] / 1000)


def parse_7zip_mips(output: str) -> int:
    """Return the final 7-Zip CPU rating in its native MIPS unit."""
    match = _SEVEN_ZIP_TOTAL.search(output)
    if match is None:
        raise HardwareBenchmarkError(
            f"7z={output[-500:]!r}; esperado linha Tot com Rating MIPS")
    return int(match.group("mips"))


def parse_llama_tokens_per_second(output: str) -> float:
    """Return avg_ts from llama-bench's generation row."""
    try:
        rows = json.loads(output)
    except json.JSONDecodeError as exc:
        raise HardwareBenchmarkError(
            f"llama-bench={output[-500:]!r}; esperado array JSON") from exc
    if not isinstance(rows, list):
        raise HardwareBenchmarkError(
            f"llama-bench={rows!r}; esperado array JSON")
    for row in rows:
        if isinstance(row, dict) and _positive_int(row.get("n_gen")):
            return _positive_float(row.get("avg_ts"), output)
    raise HardwareBenchmarkError(
        f"llama-bench={output[-500:]!r}; esperado linha n_gen > 0 com avg_ts")


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _positive_float(value: object, output: str) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    raise HardwareBenchmarkError(
        f"avg_ts={value!r} em {output[-500:]!r}; esperado numero > 0")
