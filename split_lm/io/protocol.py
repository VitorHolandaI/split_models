import json, math, socket, struct
import torch


_TENSOR_HEADER = struct.Struct("!2sBBBBI")
_TENSOR_MAGIC = b"SL"
_TENSOR_VERSION = 1
_DTYPE_TO_CODE = {
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.float32: 3,
    torch.float64: 4,
    torch.int8: 5,
    torch.uint8: 6,
    torch.int16: 7,
    torch.int32: 8,
    torch.int64: 9,
    torch.bool: 10,
}
_CODE_TO_DTYPE = {code: dtype for dtype, code in _DTYPE_TO_CODE.items()}


class SocketClosedError(ConnectionError):
    pass


class NetworkStats:
    def __init__(self):
        self.sent_bytes = 0
        self.received_bytes = 0
        self.sent_messages = 0
        self.received_messages = 0
        self.min_sent = None
        self.max_sent = 0
        self.min_received = None
        self.max_received = 0

    def record_sent(self, payload_size):
        self.sent_messages += 1
        self.sent_bytes += payload_size
        self.min_sent = payload_size if self.min_sent is None else min(self.min_sent, payload_size)
        self.max_sent = max(self.max_sent, payload_size)

    def record_received(self, payload_size):
        self.received_messages += 1
        self.received_bytes += payload_size
        self.min_received = payload_size if self.min_received is None else min(self.min_received, payload_size)
        self.max_received = max(self.max_received, payload_size)


def _recvall(s, n):
    result = bytearray(n)
    view = memoryview(result)
    received = 0
    while received < n:
        count = s.recv_into(view[received:], n - received)
        if not count:
            raise SocketClosedError(
                f"socket fechou após {received} de {n} bytes; esperado frame completo"
            )
        received += count
    return result


def _record_sent(stats, payload_size):
    if stats is not None:
        stats.record_sent(payload_size)


def _record_received(stats, payload_size):
    if stats is not None:
        stats.record_received(payload_size)


def _send(s, t, stats=None, reset_cache=False):
    tensor = t.detach().to("cpu").contiguous()
    dtype_code = _DTYPE_TO_CODE.get(tensor.dtype)
    if dtype_code is None:
        raise ValueError(f"dtype {tensor.dtype} não suportado; esperado {_DTYPE_TO_CODE.keys()}")
    shape = tuple(tensor.shape)
    if len(shape) > 255:
        raise ValueError(f"tensor tem {len(shape)} dimensões; esperado no máximo 255")
    payload = memoryview(tensor.view(torch.uint8).reshape(-1).numpy())
    header = _TENSOR_HEADER.pack(
        _TENSOR_MAGIC, _TENSOR_VERSION, int(reset_cache), dtype_code,
        len(shape), payload.nbytes,
    )
    dimensions = struct.pack(f"!{len(shape)}I", *shape) if shape else b""
    s.sendall(header + dimensions); s.sendall(payload)
    _record_sent(stats, len(header) + len(dimensions) + payload.nbytes)


def _recv(s, dev, stats=None):
    header = _recvall(s, _TENSOR_HEADER.size)
    magic, version, flags, dtype_code, ndim, payload_size = _TENSOR_HEADER.unpack(header)
    if magic != _TENSOR_MAGIC or version != _TENSOR_VERSION:
        raise ValueError(f"frame tensor inválido: magic={magic!r}, version={version}")
    if flags not in (0, 1):
        raise ValueError(f"flags tensor {flags}; esperado 0 ou 1")
    dtype = _CODE_TO_DTYPE.get(dtype_code)
    if dtype is None:
        raise ValueError(f"código dtype {dtype_code}; esperado {_CODE_TO_DTYPE.keys()}")
    dimensions = _recvall(s, ndim * 4)
    shape = struct.unpack(f"!{ndim}I", dimensions) if ndim else ()
    expected_size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if payload_size != expected_size:
        raise ValueError(
            f"payload tensor tem {payload_size} bytes; esperado {expected_size} para shape {shape}"
        )
    payload = _recvall(s, payload_size)
    _record_received(stats, _TENSOR_HEADER.size + len(dimensions) + payload_size)
    tensor = torch.frombuffer(payload, dtype=dtype).reshape(shape)
    return tensor.to(dev), bool(flags)


def _send_msg(s, msg):
    d = json.dumps(msg).encode("utf-8")
    s.sendall(struct.pack("!I", len(d))); s.sendall(d)


def _recv_msg(s):
    n, = struct.unpack("!I", _recvall(s, 4))
    return json.loads(_recvall(s, n).decode("utf-8"))
