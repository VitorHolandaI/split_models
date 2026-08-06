"""Inventario exato de um no da frota, coletado por SSH em uma unica rodada.

O comando remoto emite secoes rotuladas (`SECTION <nome>`); o parser e estrito
por design: falta de secao ou formato inesperado viram InventoryError com o
host e o formato esperado, porque um inventario pela metade vira uma decisao
errada de promocao mais tarde. Valor ausente legitimo (build-info.txt fora da
imagem, nenhum dispositivo listado) e registrado no JSON, nao silenciado.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# Estados reportados ficam como texto: o no remoto pode rodar outra versao e
# o coordenador nao deve morrer nem mentir por causa de um enum inexistente.
_INVENTORY_FIELDS = ("host", "node_id", "hostname", "ip", "mem_total_kb",
                     "devices", "build_commit", "mode")
_SECTIONS = ("hostname", "ip", "memtotal", "status", "build", "devices")
_MEMTOTAL_PATTERN = re.compile(r"MemTotal:\s+(\d+)\s+kB")
_BUILD_INFO = "/usr/share/doc/split-llama-cpp/build-info.txt"


class InventoryError(ValueError):
    """A coleta ou a leitura de um inventario falhou; o host e o esperado vao na mensagem."""


def inventory_sections_command() -> list[str]:
    """Build the one SSH round trip that captures every inventory fact.

    Seccoes opcionais (build, devices) terminam em `|| true` para nunca matar
    a coleta: a ausencia e registrada como `unknown`/vazio no parser. O restante
    e obrigatorio e a falta vira InventoryError.

    Usage example: `result = ssh.run(host, inventory_sections_command())`.
    """
    script = (
        "printf 'SECTION hostname\\n'; hostname\n"
        "printf 'SECTION ip\\n'; hostname -I\n"
        "printf 'SECTION memtotal\\n'; grep '^MemTotal:' /proc/meminfo\n"
        "printf 'SECTION status\\n'; /usr/local/bin/split-llama mode status --brief\n"
        f"printf 'SECTION build\\n'; sed -n 's/^commit: //p' {_BUILD_INFO} 2>/dev/null || true\n"
        "printf 'SECTION devices\\n'; llama-cli --list-devices 2>&1 || true\n"
    )
    return ["bash", "-c", script]


def parse_inventory_sections(stdout: str) -> dict[str, str]:
    """Split the labeled `SECTION x` payload into one text per section."""
    sections: dict[str, str] = {}
    current: str | None = None
    lines: list[str] = []
    for line in stdout.splitlines():
        if line.startswith("SECTION "):
            if current is not None:
                sections[current] = "\n".join(lines)
            current = line[len("SECTION "):]
            lines = []
        else:
            lines.append(line)
    if current is not None:
        sections[current] = "\n".join(lines)
    return sections


def parse_node_inventory(stdout: str, host: str) -> NodeInventory:
    """Parse one node's section payload into the exact typed inventory.

    Usage example: `node = parse_node_inventory(result.stdout, "node-a")`.
    """
    sections = parse_inventory_sections(stdout)
    missing = [name for name in _SECTIONS if name not in sections]
    if missing:
        raise InventoryError(
            f"inventario de {host} sem secoes {missing}; esperado "
            f"todas de {list(_SECTIONS)} no stdout")
    hostname = sections["hostname"].strip()
    status_ip, node_id, mode = _parse_status(sections["status"], host)
    ip = status_ip or _first_token(sections["ip"])
    return NodeInventory(
        host, node_id, hostname, ip or "unknown",
        _parse_memtotal(sections["memtotal"], host),
        _parse_devices(sections["devices"]),
        sections["build"].strip() or "unknown", mode)


def _parse_memtotal(text: str, host: str) -> int:
    match = _MEMTOTAL_PATTERN.search(text)
    if match is None:
        raise InventoryError(
            f"memtotal={text.strip()!r} do no {host}; esperado "
            "'MemTotal: N kB' em /proc/meminfo")
    return int(match.group(1))


def _parse_status(text: str, host: str) -> tuple[str, str, str]:
    fields: dict[str, str] = {}
    for part in text.strip().split("|"):
        part = part.strip()
        if "=" in part:
            key, _, value = part.partition("=")
            fields[key.strip()] = value.strip()
    if "role" not in fields or "node" not in fields:
        raise InventoryError(
            f"status={text.strip()!r} do no {host}; esperado "
            "'role=<modo> | node=node-<hostname> | ip=<ip>' do "
            "`split-llama mode status --brief`")
    return fields.get("ip", ""), fields["node"], fields["role"]


def _parse_devices(text: str) -> tuple[str, ...]:
    return tuple(line.strip() for line in text.splitlines() if line.strip())


def _first_token(text: str) -> str:
    return text.strip().split()[0] if text.strip() else ""


@dataclass(frozen=True)
class NodeInventory:
    """Every measured fact about one node of the fleet, in one record.

    `host` is the SSH host as the operator typed it; `ip` is the LAN address
    the node itself chose (status) with `hostname -I` as fallback.
    """

    host: str
    node_id: str
    hostname: str
    ip: str
    mem_total_kb: int
    devices: tuple[str, ...]
    build_commit: str
    mode: str

    def to_record(self) -> dict[str, object]:
        """Serialize for the per-run inventory JSON.

        Usage example: `record = node.to_record()`.
        """
        return {"host": self.host, "node_id": self.node_id,
                "hostname": self.hostname, "ip": self.ip,
                "mem_total_kb": self.mem_total_kb,
                "devices": list(self.devices),
                "build_commit": self.build_commit, "mode": self.mode}

    @classmethod
    def from_record(cls, record: object) -> NodeInventory:
        """Rebuild a node, refusing anything but the exact recorded shape.

        Usage example: `node = NodeInventory.from_record(json.loads(line))`.
        """
        if not isinstance(record, dict) or set(record) != set(_INVENTORY_FIELDS):
            raise ValueError(
                f"registro={record!r}; esperado objeto com campos "
                f"{sorted(_INVENTORY_FIELDS)}")
        devices = record["devices"]
        if not isinstance(devices, list) or not all(
                isinstance(item, str) for item in devices):
            raise ValueError(
                f"devices={devices!r}; esperado lista de textos")
        mem = record["mem_total_kb"]
        if not isinstance(mem, int) or isinstance(mem, bool):
            raise ValueError(
                f"mem_total_kb={mem!r}; esperado inteiro de KiB")
        return cls(
            _text(record, "host"), _text(record, "node_id"),
            _text(record, "hostname"), _text(record, "ip"), mem,
            tuple(devices), _text(record, "build_commit"),
            _text(record, "mode"))


def _text(record: dict[str, object], label: str) -> str:
    value = record[label]
    if not isinstance(value, str):
        raise ValueError(f"{label}={value!r}; esperado texto")
    return value


@dataclass(frozen=True)
class FleetInventory:
    """One collection run: timestamp plus the exact nodes in input order."""

    timestamp: str
    nodes: tuple[NodeInventory, ...]

    def to_json(self) -> str:
        """Serialize the run; preserved verbatim per scenario run."""
        return json.dumps({"timestamp": self.timestamp,
                           "nodes": [node.to_record() for node in self.nodes]},
                          indent=2)

    @classmethod
    def from_json(cls, text: str) -> FleetInventory:
        """Rebuild the run, validating both timestamp and every node record."""
        try:
            record = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"inventario={text!r}; esperado JSON com timestamp e nodes") from exc
        if not isinstance(record, dict) or set(record) != {"timestamp", "nodes"}:
            raise ValueError(
                f"inventario={record!r}; esperado objeto com campos "
                "'timestamp' e 'nodes'")
        nodes = record["nodes"]
        if not isinstance(nodes, list):
            raise ValueError(f"nodes={nodes!r}; esperado lista de registros")
        return cls(_text(record, "timestamp"),
                   tuple(NodeInventory.from_record(item) for item in nodes))

    def save(self, path: Path) -> None:
        """Write the run atomically under the run directory.

        Usage example: `inventory.save(Path("runs/inventory-1.json"))`.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
