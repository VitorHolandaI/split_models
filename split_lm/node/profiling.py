import sys


def _stat_add(stats, key, seconds):
    if stats is None:
        return
    stats[key] = stats.get(key, 0.0) + seconds


def _stat_inc(stats, key, count=1):
    if stats is None:
        return
    stats[key] = stats.get(key, 0) + count


def _format_bytes(value):
    value = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} GiB"


def _profile_report(node, stats, net_stats=None):
    if not stats:
        return
    total = stats.get("total", 0.0)
    tokens = int(stats.get("tokens", 0))
    tok_s = tokens / total if total else 0.0
    ms_tok = total / tokens * 1000.0 if tokens else 0.0
    print(f"PROFILE {node}: generated {tokens} tokens in {total:.3f}s ({tok_s:.2f} tok/s, {ms_tok:.1f} ms/token)", file=sys.stderr, flush=True)
    for key in sorted(k for k in stats if k not in {"tokens", "total"}):
        _profile_step_report(key, stats[key], total)
    if net_stats is not None:
        _profile_network_report(net_stats)


def _profile_step_report(key, seconds, total):
    names = {
        "forward": "ran this node's model layers",
        "recv": "waited for hidden states from the previous node",
        "recv_token": "waited for the final token from the tail node",
        "send": "sent hidden states/input to the next node",
        "send_downstream": "sent hidden states to the next worker",
        "send_token": "sent the selected token back to the head",
    }
    pct = (seconds / total * 100.0) if total else 0.0
    label = names.get(key, key.replace("_", " "))
    print(f"PROFILE step: {label}: {seconds:.3f}s ({pct:.1f}% of total)", file=sys.stderr, flush=True)


def _profile_network_report(net_stats):
    sent_avg = net_stats.sent_bytes / net_stats.sent_messages if net_stats.sent_messages else 0.0
    received_avg = net_stats.received_bytes / net_stats.received_messages if net_stats.received_messages else 0.0
    min_sent = net_stats.min_sent or 0
    min_received = net_stats.min_received or 0
    print(f"PROFILE network: sent {_format_bytes(net_stats.sent_bytes)} in {net_stats.sent_messages} messages", file=sys.stderr, flush=True)
    print(f"PROFILE network: received {_format_bytes(net_stats.received_bytes)} in {net_stats.received_messages} messages", file=sys.stderr, flush=True)
    print(f"PROFILE network: sent frame min/max/avg: {_format_bytes(min_sent)} / {_format_bytes(net_stats.max_sent)} / {_format_bytes(sent_avg)}", file=sys.stderr, flush=True)
    print(f"PROFILE network: received frame min/max/avg: {_format_bytes(min_received)} / {_format_bytes(net_stats.max_received)} / {_format_bytes(received_avg)}", file=sys.stderr, flush=True)
