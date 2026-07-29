import socket
import threading
import unittest
from collections.abc import Sequence

from split_lm.cluster.control import (
    JobValidationError, VerticalJob, request_agent, serve_control,
    worker_command)


class FakeWorkerProcess:
    def __init__(self, pid: int):
        self.pid = pid
        self.terminated = False

    def poll(self) -> int | None:
        return 0 if self.terminated else None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        self.terminated = True


class FakeWorkerProcessFactory:
    def __init__(self):
        self.commands: list[tuple[str, ...]] = []
        self.processes: list[FakeWorkerProcess] = []

    def __call__(self, command: Sequence[str]) -> FakeWorkerProcess:
        process = FakeWorkerProcess(4000 + len(self.processes))
        self.commands.append(tuple(command))
        self.processes.append(process)
        return process


def sample_job() -> VerticalJob:
    return VerticalJob(
        model="Qwen/Qwen2.5-0.5B", layer_start=12, layer_end=24,
        total_layers=24, listen_port=9999,
        control_to="10.66.66.14:10001", node_id="node-2",
        connect_to=None, token_back_to="10.66.66.14:9999",
        max_new_tokens=50,
    )


class WorkerCommandTest(unittest.TestCase):
    def test_command_is_built_from_validated_vertical_fields(self):
        command = worker_command(sample_job(), executable="/venv/python")

        self.assertEqual(command[:3], ["/venv/python", "-m", "split_lm"])
        self.assertIn("worker", command)
        self.assertIn("--token-back-to", command)
        self.assertNotIn("sh", command)

    def test_job_rejects_unknown_shell_command_field(self):
        payload = sample_job().to_payload() | {"command": "touch /tmp/owned"}

        with self.assertRaisesRegex(JobValidationError, "command"):
            VerticalJob.from_payload(payload)

    def test_job_rejects_invalid_layer_range(self):
        payload = sample_job().to_payload() | {"layer_start": 24}

        with self.assertRaisesRegex(JobValidationError, "layer_start=24"):
            VerticalJob.from_payload(payload)


class ControlAgentTest(unittest.TestCase):
    def setUp(self):
        self.factory = FakeWorkerProcessFactory()
        self.stop = threading.Event()
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        self.port = server.getsockname()[1]
        server.close()
        self.thread = threading.Thread(
            target=serve_control,
            args=("node-2", self.port, self.stop, self.factory), daemon=True)
        self.thread.start()

    def tearDown(self):
        self.stop.set()
        self.thread.join(2)

    def test_start_and_stop_manage_one_validated_worker_process(self):
        address = ("127.0.0.1", self.port)

        started = request_agent(address, {"type": "START", "job": sample_job().to_payload()})
        stopped = request_agent(address, {"type": "STOP"})

        self.assertEqual(started, {"type": "STARTED", "node": "node-2", "pid": 4000})
        self.assertEqual(stopped, {"type": "STOPPED", "node": "node-2"})
        self.assertTrue(self.factory.processes[0].terminated)

    def test_second_start_is_rejected_while_worker_is_running(self):
        address = ("127.0.0.1", self.port)
        message = {"type": "START", "job": sample_job().to_payload()}
        request_agent(address, message)

        response = request_agent(address, message)

        self.assertEqual(response["type"], "ERROR")
        self.assertIn("ja executa", str(response["message"]))


if __name__ == "__main__":
    unittest.main()
