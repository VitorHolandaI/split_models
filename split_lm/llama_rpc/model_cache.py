"""Keep the stateless llama.cpp cache limited to one selected GGUF model."""

from __future__ import annotations

import shutil
from pathlib import Path

_SELECTED_MODEL = ".selected-model"
_STORAGE_RESERVE_GB = 0.5


class ModelCacheError(ValueError):
    """The selected cache cannot safely hold the requested GGUF."""


def ensure_model_fits_storage(free_bytes: int, model_gigabytes: float) -> None:
    """Refuse a download unless the measured free space includes a reserve."""
    free_gigabytes = free_bytes / 1_000_000_000
    required = model_gigabytes + _STORAGE_RESERVE_GB
    if free_gigabytes < required:
        raise ModelCacheError(
            f"modelo={model_gigabytes:.2f} GB e reserva={_STORAGE_RESERVE_GB:.2f} "
            f"GB; cache tem somente {free_gigabytes:.2f} GB livres")


def select_cached_model(cache_dir: Path, model: str,
                        model_gigabytes: float | None = None) -> bool:
    """Preserve `model` when already selected, otherwise remove old downloads.

    The free-space check runs before anything is erased, so a refused download
    never destroys the previous GGUF.

    Usage example: `select_cached_model(Path("/run/models"), "org/model:Q4")`.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    marker = cache_dir / _SELECTED_MODEL
    if marker.exists() and marker.read_text(encoding="utf-8") == model:
        return True
    if model_gigabytes is not None:
        ensure_model_fits_storage(shutil.disk_usage(cache_dir).free,
                                  model_gigabytes)
    _clear_cache(cache_dir)
    marker.write_text(model, encoding="utf-8")
    return False


def record_cached_model(cache_dir: Path, model: str) -> None:
    """Record the model that a supervised server loaded successfully.

    Usage example: `record_cached_model(Path("/run/models"), "org/model:Q4")`.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / _SELECTED_MODEL).write_text(model, encoding="utf-8")


def _clear_cache(cache_dir: Path) -> None:
    for child in cache_dir.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
