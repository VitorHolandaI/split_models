import json
import socket
import struct


class SocketClosedError(ConnectionError):
    pass


def _recvall(connection: socket.socket, size: int) -> bytearray:
    result = bytearray(size)
    view = memoryview(result)
    received = 0
    while received < size:
        count = connection.recv_into(view[received:], size - received)
        if not count:
            raise SocketClosedError(
                f"socket fechou apos {received} de {size} bytes; esperado frame completo"
            )
        received += count
    return result


def _send_msg(connection: socket.socket, message: dict[str, object]) -> None:
    payload = json.dumps(message).encode("utf-8")
    connection.sendall(struct.pack("!I", len(payload)))
    connection.sendall(payload)


def _recv_msg(connection: socket.socket) -> dict[str, object]:
    size, = struct.unpack("!I", _recvall(connection, 4))
    message = json.loads(_recvall(connection, size).decode("utf-8"))
    if not isinstance(message, dict):
        raise ValueError(f"mensagem={message!r}; esperado objeto JSON")
    return message
