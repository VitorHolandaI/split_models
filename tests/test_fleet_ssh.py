import unittest

from split_lm.llama_rpc.fleet_ssh import SshError, SshKeyRunner, SshResult


class FakeProcessResult:
    """Stand-in for subprocess.CompletedProcess, carrying what the runner reads."""

    def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakeSubprocessRun:
    """subprocess.run de mentira: grava argv e responde com saida programada."""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self._returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def __call__(self, argv: list[str], **kwargs: object) -> FakeProcessResult:
        self.calls.append((list(argv), dict(kwargs)))
        return FakeProcessResult(self._returncode, self._stdout, self._stderr)


class SshKeyRunnerTest(unittest.TestCase):
    def test_command_is_ssh_by_key_without_password_prompts(self):
        fake = FakeSubprocessRun()
        runner = SshKeyRunner("/root/.ssh/fleet_ed25519", user="root", run=fake)

        runner.run("node-a", ["split-llama", "mode", "status"])

        command = fake.calls[0][0]
        self.assertEqual(command[0], "ssh")
        self.assertIn("-i", command)
        self.assertEqual(command[command.index("-i") + 1],
                         "/root/.ssh/fleet_ed25519")
        self.assertIn("BatchMode=yes", command)
        self.assertIn("root@node-a", command)
        self.assertEqual(command[-3:], ["split-llama", "mode", "status"])

    def test_success_returns_stdout_and_stderr(self):
        fake = FakeSubprocessRun(stdout="role=worker", stderr="")
        runner = SshKeyRunner("key", run=fake)

        result = runner.run("node-a", ["hostname"])

        self.assertEqual(result, SshResult("role=worker", ""))

    def test_non_zero_exit_names_host_command_and_output(self):
        fake = FakeSubprocessRun(returncode=255, stderr="Permission denied")
        runner = SshKeyRunner("key", run=fake)

        with self.assertRaises(SshError) as failed:
            runner.run("node-a", ["hostname"])

        reported = str(failed.exception)
        self.assertIn("node-a", reported)
        self.assertIn("255", reported)
        self.assertIn("Permission denied", reported)

    def test_missing_ssh_binary_names_the_host_and_key(self):
        def broken(_argv: list[str], **_kwargs: object) -> FakeProcessResult:
            raise FileNotFoundError("ssh")

        runner = SshKeyRunner("/root/.ssh/fleet_ed25519", run=broken)

        with self.assertRaises(SshError) as failed:
            runner.run("node-a", ["hostname"])

        reported = str(failed.exception)
        self.assertIn("node-a", reported)
        self.assertIn("fleet_ed25519", reported)


if __name__ == "__main__":
    unittest.main()
