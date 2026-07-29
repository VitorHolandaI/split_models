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


def _color(text: str, code: str, enabled: bool) -> str:
    return f"\033[{code}m{text}\033[0m" if enabled else text


def _profile_value(label: str, value: str, code: str, color: bool) -> str:
    return f"{label:<24}{_color(value, f'1;{code}', color)}"


def _profile_step_line(key: str, seconds: float, total: float,
                       color: bool) -> str:
    pct = (seconds / total * 100.0) if total else 0.0
    value = f"{seconds * 1000:.1f} ms ({pct:.1f}%)"
    return _profile_value(key, value, "34", color)


def _profile_lines(node: str, stats: dict[str, float | int],
                   color: bool = False) -> list[str]:
    total = float(stats.get("total", 0.0))
    tokens = int(stats.get("tokens", 0))
    rate = tokens / total if total else 0.0
    average = total / tokens * 1000 if tokens else 0.0
    lines = [
        _color(f"=== PROFILE | {node.upper()} ===", "1;36", color), "",
        _profile_value("generated_tokens", str(tokens), "34", color),
        _profile_value("total_latency", f"{total * 1000:.1f} ms", "35", color),
        _profile_value("average_per_token", f"{average:.1f} ms", "33", color),
        _profile_value("tokens_per_second", f"{rate:.2f} tok/s", "32", color), "",
    ]
    for key in sorted(name for name in stats if name not in {"tokens", "total"}):
        lines.append(_profile_step_line(key, float(stats[key]), total, color))
    return lines


def _profile_report(node, stats, net_stats=None):
    if not stats:
        return
    print("\n".join(_profile_lines(node, stats, sys.stderr.isatty())),
          file=sys.stderr, flush=True)
    if net_stats is not None:
        _profile_network_report(net_stats)


def _profile_network_report(net_stats):
    sent_avg = net_stats.sent_bytes / net_stats.sent_messages if net_stats.sent_messages else 0.0
    received_avg = net_stats.received_bytes / net_stats.received_messages if net_stats.received_messages else 0.0
    min_sent = net_stats.min_sent or 0
    min_received = net_stats.min_received or 0
    print(f"PROFILE network: sent {_format_bytes(net_stats.sent_bytes)} in {net_stats.sent_messages} messages", file=sys.stderr, flush=True)
    print(f"PROFILE network: received {_format_bytes(net_stats.received_bytes)} in {net_stats.received_messages} messages", file=sys.stderr, flush=True)
    print(f"PROFILE network: sent frame min/max/avg: {_format_bytes(min_sent)} / {_format_bytes(net_stats.max_sent)} / {_format_bytes(sent_avg)}", file=sys.stderr, flush=True)
    print(f"PROFILE network: received frame min/max/avg: {_format_bytes(min_received)} / {_format_bytes(net_stats.max_received)} / {_format_bytes(received_avg)}", file=sys.stderr, flush=True)
