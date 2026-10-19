import torch
from transformers.cache_utils import DynamicCache


def _causal_mask(seq_len, dev, dtype, key_len=None, pos_start=0):
    key_len = key_len or seq_len
    query_pos = torch.arange(pos_start, pos_start + seq_len, device=dev).unsqueeze(1)
    key_pos = torch.arange(key_len, device=dev).unsqueeze(0)
    blocked = key_pos > query_pos
    mask = torch.zeros((seq_len, key_len), device=dev, dtype=dtype)
    return mask.masked_fill(blocked, float("-inf")).unsqueeze(0).unsqueeze(0)


def _new_cache_state():
    return {"cache": DynamicCache(), "pos": 0}


def _final_logits(model, hidden):
    normalized = model.model.norm(hidden)
    return model.lm_head(normalized[:, -1:, :])


def _forward(model, hidden, dev, is_head, is_last, input_ids=None,
             cache_state=None):
    if cache_state is None:
        raise ValueError("cache_state é obrigatório para evitar recomputação da sequência")
    with torch.inference_mode():
        if is_head:
            hidden = model.model.embed_tokens(input_ids)
            seq_len = input_ids.shape[1]
        else:
            seq_len = hidden.shape[1]
        pos_start = cache_state["pos"]
        key_len = pos_start + seq_len
        pos = torch.arange(pos_start, key_len, device=dev).unsqueeze(0)
        cos, sin = model.model.rotary_emb(hidden, pos)
        pe = (cos, sin)
        mask = _causal_mask(seq_len, dev, hidden.dtype, key_len, pos_start)
        cache_pos = torch.arange(pos_start, key_len, device=dev)
        for layer in model.model.layers:
            out = layer(hidden, position_embeddings=pe, attention_mask=mask,
                        past_key_values=cache_state["cache"],
                        use_cache=True,
                        cache_position=cache_pos)
            hidden = out[0] if isinstance(out, (tuple, list)) else out
        if is_last:
            return _final_logits(model, hidden)
    return hidden
