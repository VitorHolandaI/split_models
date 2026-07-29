"""Master orchestration for discovered vertical workers."""

from __future__ import annotations

import logging
import subprocess
import sys
from collections.abc import Sequence
from typing import Protocol

from .control import VerticalJob, request_agent
from .plan import ClusterNode, WorkerAssignment, VerticalPlan, build_vertical_plan


log = logging.getLogger("split_lm")
INFERENCE_PORT = 9999
HEAD_CONTROL_PORT = 10001


class AgentRequester(Protocol):
    def __call__(self, address: tuple[str, int], message: dict[str, object],
                 timeout: float = 5.0) -> dict[str, object]: ...


class HeadRunner(Protocol):
    def __call__(self, command: Sequence[str]) -> None: ...


def job_for_assignment(assignment: WorkerAssignment, model: str,
                       total_layers: int, max_new_tokens: int) -> VerticalJob:
    """Convert one planned layer range into a validated worker job."""
    start, end = assignment.layers
    return VerticalJob(
        model, start, end, total_layers, INFERENCE_PORT,
        assignment.control_to, assignment.node.node_id,
        assignment.connect_to, assignment.token_back_to, max_new_tokens,
    )


def head_command(model: str, total_layers: int, layers: tuple[int, int],
                 first_worker: str, workers: int, max_new_tokens: int,
                 ready_timeout: float, context_sizes: tuple[int, ...] = (),
                 metrics_file: str = "benchmark-results/generation.jsonl",
                 benchmark_output: str = "benchmark-results/context-sweep",
                 executable: str = sys.executable) -> list[str]:
    """Build the local interactive head command for a vertical plan."""
    start, end = layers
    command = [
        executable, "-m", "split_lm", "--model", model,
        "--total-layers", str(total_layers), "--max-new-tokens", str(max_new_tokens),
        "head", "--layer-start", str(start), "--layer-end", str(end),
        "--connect", first_worker, "--port", str(INFERENCE_PORT),
        "--control-port", str(HEAD_CONTROL_PORT), "--workers", str(workers),
        "--ready-timeout", str(ready_timeout), "--metrics-file", metrics_file, "--verbose",
    ]
    if context_sizes:
        return command + ["--context-sizes", ",".join(map(str, context_sizes)),
                          "--benchmark-output", benchmark_output, "--profiling"]
    return command + ["--interactive"]


def _agent_address(assignment: WorkerAssignment) -> tuple[str, int]:
    return assignment.node.address, assignment.node.control_port


def _start_workers(plan: VerticalPlan, model: str, total_layers: int,
                   max_new_tokens: int, requester: AgentRequester) -> list[WorkerAssignment]:
    started: list[WorkerAssignment] = []
    try:
        for assignment in reversed(plan.workers):
            job = job_for_assignment(assignment, model, total_layers, max_new_tokens)
            message = {"type": "START", "job": job.to_payload()}
            response = requester(_agent_address(assignment), message)
            if response.get("type") != "STARTED":
                raise RuntimeError(f"worker={assignment.node.node_id} recusou START: {response}")
            log.info("worker %s iniciou pid=%s", assignment.node.node_id, response.get("pid"))
            started.append(assignment)
    except Exception:
        _stop_workers(started, requester)
        raise
    return started


def _stop_workers(assignments: list[WorkerAssignment], requester: AgentRequester) -> None:
    for assignment in assignments:
        try:
            requester(_agent_address(assignment), {"type": "STOP"})
        except (ConnectionError, OSError) as exc:
            log.warning("falha parando worker %s: %s", assignment.node.node_id, exc)


def _run_head(command: Sequence[str]) -> None:
    subprocess.run(list(command), check=True)


def run_vertical_cluster(model: str, total_layers: int, head_address: str,
                         workers: tuple[ClusterNode, ...], max_new_tokens: int,
                         ready_timeout: float, requester: AgentRequester = request_agent,
                         runner: HeadRunner = _run_head,
                         context_sizes: tuple[int, ...] = (),
                         metrics_file: str = "benchmark-results/generation.jsonl",
                         benchmark_output: str = "benchmark-results/context-sweep") -> None:
    """Start discovered workers, run interactive head, then stop remote jobs."""
    plan = build_vertical_plan(total_layers, head_address, workers)
    started: list[WorkerAssignment] = []
    try:
        started = _start_workers(plan, model, total_layers, max_new_tokens, requester)
        runner(head_command(model, total_layers, plan.head_layers, plan.first_worker,
                            len(plan.workers), max_new_tokens, ready_timeout,
                            context_sizes, metrics_file, benchmark_output))
    finally:
        _stop_workers(started, requester)
