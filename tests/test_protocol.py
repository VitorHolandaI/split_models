import socket
import unittest

import torch

from split_lm.io.network import _configure_connected_socket, _connect
from split_lm.io.protocol import NetworkStats, SocketClosedError, _recv, _send


class ProtocolTest(unittest.TestCase):
    def test_tensor_round_trip_tracks_frame_and_reset(self):
        sender, receiver = socket.socketpair()
        sent_stats = NetworkStats()
        received_stats = NetworkStats()
        expected = torch.arange(8, dtype=torch.float32)
        try:
            _send(sender, expected, sent_stats, reset_cache=True)
            actual, reset_cache = _recv(receiver, "cpu", received_stats)
        finally:
            sender.close()
            receiver.close()

        self.assertTrue(reset_cache)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(sent_stats.sent_bytes, received_stats.received_bytes)
        self.assertEqual(sent_stats.sent_messages, 1)
        self.assertEqual(received_stats.received_messages, 1)

    def test_int64_token_uses_compact_binary_frame(self):
        sender, receiver = socket.socketpair()
        stats = NetworkStats()
        try:
            _send(sender, torch.tensor([42], dtype=torch.int64), stats)
            _recv(receiver, "cpu")
        finally:
            sender.close()
            receiver.close()

        self.assertEqual(stats.sent_bytes, 22)

    def test_bfloat16_hidden_state_round_trip(self):
        sender, receiver = socket.socketpair()
        expected = torch.arange(12, dtype=torch.bfloat16).reshape(1, 3, 4)
        try:
            _send(sender, expected)
            actual, reset_cache = _recv(receiver, "cpu")
        finally:
            sender.close()
            receiver.close()

        self.assertFalse(reset_cache)
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(actual, expected))

    def test_closed_socket_has_distinct_error(self):
        sender, receiver = socket.socketpair()
        sender.close()
        try:
            with self.assertRaises(SocketClosedError):
                _recv(receiver, "cpu")
        finally:
            receiver.close()

    def test_tcp_connections_disable_nagle(self):
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        client = _connect(server.getsockname(), "test", attempts=1)
        accepted = _configure_connected_socket(server.accept()[0])
        try:
            option = (socket.IPPROTO_TCP, socket.TCP_NODELAY)
            self.assertEqual(client.getsockopt(*option), 1)
            self.assertEqual(accepted.getsockopt(*option), 1)
        finally:
            client.close()
            accepted.close()
            server.close()


if __name__ == "__main__":
    unittest.main()
