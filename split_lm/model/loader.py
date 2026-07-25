import json
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM

from ..constants import SUPPORTED_MODEL_TYPES


def _check_supported_config(config):
    model_type = getattr(config, "model_type", None)
    if model_type in SUPPORTED_MODEL_TYPES:
        return
    expected = ", ".join(sorted(SUPPORTED_MODEL_TYPES))
    raise ValueError(f"modelo {model_type!r} nao suportado; esperado um de: {expected}")


def _load_weights(model_name, layer_start, layer_end, n_total, config):
    shard_paths = _weight_paths(model_name, layer_start, layer_end, n_total, config)
    is_head = layer_start == 0
    is_last = layer_end == n_total
    tied_embeddings = bool(getattr(config, "tie_word_embeddings", False))
    w = {}
    for path in shard_paths:
        with safe_open(path, framework="pt", device="cpu") as sd:
            for key in sd.keys():
                if key == "model.embed_tokens.weight":
                    weight = sd.get_tensor(key)
                    if is_head:
                        w[key] = weight
                    if is_last and tied_embeddings:
                        w["lm_head.weight"] = weight
                if key == "lm_head.weight" and is_last:
                    w[key] = sd.get_tensor(key)
                if key == "model.norm.weight" and is_last:
                    w[key] = sd.get_tensor(key)
                if key == "model.rotary_emb.inv_freq" and is_head:
                    w[key] = sd.get_tensor(key)
                if key.startswith("model.layers."):
                    lidx = int(key.split(".")[2])
                    if layer_start <= lidx < layer_end:
                        nk = key.replace(f".{lidx}.", f".{lidx - layer_start}.")
                        w[nk] = sd.get_tensor(key)
    return w


def _wanted_weight(key, layer_start, layer_end, n_total, config):
    tied_embeddings = bool(getattr(config, "tie_word_embeddings", False))
    if key == "model.embed_tokens.weight":
        return layer_start == 0 or (layer_end == n_total and tied_embeddings)
    if key == "lm_head.weight": return layer_end == n_total
    if key == "model.norm.weight": return layer_end == n_total
    if key == "model.rotary_emb.inv_freq": return layer_start == 0
    if not key.startswith("model.layers."): return False
    lidx = int(key.split(".")[2])
    return layer_start <= lidx < layer_end


def _weight_paths(model_name, layer_start, layer_end, n_total, config):
    try:
        return [hf_hub_download(repo_id=model_name, filename="model.safetensors")]
    except Exception:
        index_path = hf_hub_download(repo_id=model_name, filename="model.safetensors.index.json")
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    names = {name for key, name in index["weight_map"].items()
             if _wanted_weight(key, layer_start, layer_end, n_total, config)}
    return [hf_hub_download(repo_id=model_name, filename=name) for name in sorted(names)]


def _validate_loaded_weights(model, weights, model_name, layer_start, layer_end,
                             n_total):
    required = {name for name, _ in model.named_parameters()
                if name.startswith("model.layers.")}
    if layer_start == 0:
        required.add("model.embed_tokens.weight")
    if layer_end == n_total:
        required.update({"model.norm.weight", "lm_head.weight"})
    missing = sorted(required - weights.keys())
    if missing:
        raise ValueError(
            f"pesos ausentes em {model_name!r} para layers {layer_start}..{layer_end}: {missing}"
        )


def _drop_unused_modules(model, is_head, is_last):
    if not is_head:
        model.model.embed_tokens = None
    if not is_last:
        model.model.norm = None
        model.lm_head = None


def _build(model_name, layer_start, layer_end, n_total, device):
    cfg = AutoConfig.from_pretrained(model_name)
    _check_supported_config(cfg)
    cfg.num_hidden_layers = layer_end - layer_start
    model = AutoModelForCausalLM.from_config(cfg)
    _drop_unused_modules(model, layer_start == 0, layer_end == n_total)
    w = _load_weights(model_name, layer_start, layer_end, n_total, cfg)
    _validate_loaded_weights(model, w, model_name, layer_start, layer_end,
                             n_total)
    incompatible = model.load_state_dict(w, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            f"pesos inesperados em {model_name!r}: {sorted(incompatible.unexpected_keys)}"
        )
    model = model.to(device).eval()
    return model
