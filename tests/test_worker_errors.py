import socket
import unittest
from typing import cast

from split_lm.io.framing import _recv_msg
from split_lm.node.worker import _send_error


class WorkerErrorTest(unittest.TestCase):
    def test_worker_sends_classified_runtime_error(self):
        sender, receiver = socket.socketpair()
        try:
            _send_error(sender, "node-7", "worker.forward", RuntimeError("tensor failed"))
            payload = _recv_msg(receiver)
        finally:
            sender.close()
            receiver.close()
        machine = cast(dict[str, object], payload["machine"])
        cause = cast(dict[str, object], payload["cause"])

        self.assertEqual(payload["origin"], "worker")
        self.assertEqual(machine["node_id"], "node-7")
        self.assertEqual(payload["stage"], "worker.forward")
        self.assertEqual(cause["type"], "RuntimeError")
        self.assertEqual(cause["message"], "tensor failed")


if __name__ == "__main__":
    unittest.main()
