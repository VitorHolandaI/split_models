"""Configuracao HAProxy para o cenario replicado, gerada pelo coordenador.

O HAProxy roda fora dos nos de inferencia. A checagem ativa exige 200 em
`/health` (o llama-server responde 503 enquanto carrega o modelo); o `timeout
tunnel` protege o streaming SSE de ser cortado no meio de uma resposta.
"""

from __future__ import annotations

import re

_SERVER_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]")


def _server_name(node_id: str) -> str:
    return _SERVER_NAME_PATTERN.sub("_", node_id) or "node"


def haproxy_config(servers: tuple[tuple[str, str], ...],
                   listen_port: int = 8080) -> str:
    """Render the HAProxy file pointing at every replica.

    `servers` is a tuple of `(node_id, address)` where address is already
    `host:port` of the llama-server HTTP endpoint.

    Usage example: `config = haproxy_config((("node-a", "10.0.0.2:8080"),))`.
    """
    if not servers:
        raise ValueError(
            f"servers={servers!r}; esperado ao menos um par (node_id, host:porta)")
    if not 1 <= listen_port <= 65535:
        raise ValueError(
            f"listen_port={listen_port}; esperado inteiro entre 1 e 65535")
    server_lines = "\n".join(
        f"    server {_server_name(node_id)} {address} check inter 5s "
        f"fall 3 rise 2"
        for node_id, address in servers)
    return (
        "# Gerado por `split-llama fleet replicated`; nao edite a mao.\n"
        "# Checagem ativa exige 200 em /health: o llama-server responde 503\n"
        "# enquanto carrega o modelo. Timeout tunnel\n"
        "# protege o streaming de expirar no meio.\n"
        "\n"
        "global\n"
        "    maxconn 256\n"
        "    log stdout format raw local0\n"
        "\n"
        "defaults\n"
        "    mode http\n"
        "    timeout connect 5s\n"
        "    timeout client 300s\n"
        "    timeout server 300s\n"
        "    timeout tunnel 3600s\n"
        "    timeout http-keep-alive 10s\n"
        "\n"
        "frontend split-llama-http\n"
        f"    bind *:{listen_port}\n"
        "    default_backend split-llama-replicas\n"
        "\n"
        "backend split-llama-replicas\n"
        "    balance roundrobin\n"
        "    option httpchk GET /health\n"
        "    http-check expect status 200\n"
        f"{server_lines}\n")
