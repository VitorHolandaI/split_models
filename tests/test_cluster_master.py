import unittest
from collections.abc import Sequence

from split_lm.cluster.master import head_command, job_for_assignment, run_vertical_cluster
from split_lm.cluster.plan import ClusterNode, WorkerAssignment


class FakeAgentRequester:
    def __init__(self):
        self.requests: list[tuple[tuple[str, int], dict[str, object]]] = []

    def __call__(self, address: tuple[str, int], message: dict[str, object],
                 timeout: float = 5.0) -> dict[str, object]:
        self.requests.append((address, message))
        response_type = "STARTED" if message["type"] == "START" else "STOPPED"
        return {"type": response_type, "node": address[0], "pid": 42}


class FakeHeadRunner:
    def __init__(self):
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: Sequence[str]) -> None:
        self.commands.append(tuple(command))


def sample_assignment() -> WorkerAssignment:
    node = ClusterNode("node-2", "worker-2", "10.66.66.8", 29510, 12, 8_000)
    return WorkerAssignment(
        node=node, layers=(12, 24), control_to="10.66.66.14:10001",
        connect_to=None, token_back_to="10.66.66.14:9999",
    )


class MasterJobTest(unittest.TestCase):
    def test_assignment_becomes_vertical_worker_job(self):
        job = job_for_assignment(
            sample_assignment(), "Qwen/Qwen2.5-0.5B", 24, 50)

        self.assertEqual(job.layer_start, 12)
        self.assertEqual(job.layer_end, 24)
        self.assertEqual(job.node_id, "node-2")
        self.assertEqual(job.token_back_to, "10.66.66.14:9999")
        self.assertEqual(job.precision, "bfloat16")

    def test_head_command_runs_the_first_slice_interactively(self):
        command = head_command(
            "Qwen/Qwen2.5-0.5B", 24, (0, 12), "10.66.66.8:9999", 1,
            50, 120, executable="/venv/python")

        self.assertEqual(command[:3], ["/venv/python", "-m", "split_lm"])
        self.assertIn("head", command)
        self.assertIn("--interactive", command)
        self.assertIn("10.66.66.8:9999", command)

    def test_context_sweep_head_is_non_interactive(self):
        command = head_command(
            "Qwen/Qwen2.5-3B-Instruct", 36, (0, 18), "10.66.66.8:9999", 1,
            16, 300, context_sizes=(128, 256), executable="/venv/python")

        self.assertNotIn("--interactive", command)
        self.assertIn("--context-sizes", command)
        self.assertIn("128,256", command)

    def test_natural_prompt_benchmark_is_non_interactive(self):
        command = head_command(
            "Qwen/Qwen2.5-3B-Instruct", 36, (0, 18), "10.66.66.8:9999", 1,
            128, 300, benchmark_prompt="Write a poem.", executable="/venv/python")

        self.assertNotIn("--interactive", command)
        self.assertIn("--prompt", command)
        self.assertIn("Write a poem.", command)
        self.assertIn("--profiling", command)

    def test_cluster_starts_workers_runs_head_and_stops_workers(self):
        requester = FakeAgentRequester()
        runner = FakeHeadRunner()

        run_vertical_cluster(
            "Qwen/Qwen2.5-0.5B", 24, "10.66.66.14",
            (sample_assignment().node,), 50, 120, requester, runner)

        self.assertEqual([request[1]["type"] for request in requester.requests],
                         ["START", "STOP"])
        self.assertEqual(len(runner.commands), 1)
        self.assertIn("--interactive", runner.commands[0])


if __name__ == "__main__":
    unittest.main()
