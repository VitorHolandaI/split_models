import errno
import threading
import unittest

from split_lm.errors import ClusterError
from split_lm.cluster.cli import main as cluster_main
from split_lm.cli import main as split_main
from split_lm.io.network import _wait_ready


class ClusterErrorTest(unittest.TestCase):
    def test_os_error_round_trip_preserves_machine_stage_and_cause(self):
        source = OSError(errno.EADDRINUSE, "Address already in use")
        error = ClusterError.from_exception(
            origin="worker", node_id="node-2", hostname="worker-2",
            address="10.66.66.8", stage="worker.bind_inference", exc=source)

        restored = ClusterError.from_payload(error.to_payload())

        self.assertEqual(restored.origin, "worker")
        self.assertEqual(restored.node_id, "node-2")
        self.assertEqual(restored.stage, "worker.bind_inference")
        self.assertEqual(restored.code, "address_in_use")
        self.assertEqual(restored.cause_type, "OSError")
        self.assertEqual(restored.errno, errno.EADDRINUSE)
        self.assertEqual(
            str(restored),
            "ERROR origin=worker machine=node-2/worker-2@10.66.66.8 "
            "stage=worker.bind_inference cause=address_in_use/OSError errno=98: "
            "[Errno 98] Address already in use")

    def test_head_raises_remote_error_instead_of_dictionary_text(self):
        remote = ClusterError.from_exception(
            "worker", "node-3", "worker-3", "10.66.66.9",
            "worker.run", RuntimeError("forward failed"))

        with self.assertRaisesRegex(ClusterError, "machine=node-3/worker-3@10.66.66.9"):
            _wait_ready(1, set(), [remote.to_payload()], threading.Lock(), 1)

    def test_cluster_cli_classifies_local_master_failure(self):
        with self.assertRaisesRegex(
                ClusterError, "origin=master machine=local/.*@local stage=master.cli"):
            cluster_main(["master"])

    def test_split_cli_classifies_local_probe_failure(self):
        with self.assertRaisesRegex(
                ClusterError, "origin=probe machine=local/.*@local stage=probe.cli"):
            split_main(["probe"])


if __name__ == "__main__":
    unittest.main()
