import unittest

from split_lm.llama_rpc.distributed_planner import (
    CapacityPlanError, RankedNode, allocate_model, build_capacity_plan,
    known_model_gigabytes, known_model_layers)
from split_lm.llama_rpc.fleet_inventory import NodeInventory
from split_lm.llama_rpc.hardware_benchmark import HardwareScore


def inventory(host: str, ram_gib: int) -> NodeInventory:
    return NodeInventory(host, f"node-{host}", host, host,
                         ram_gib * 1024 * 1024, ("Vulkan",), "abc", "worker")


def ranked_node(host: str, ram_gib: int, vulkan_tok_s: float,
                cpu_tok_s: float = 3.0) -> RankedNode:
    return RankedNode(
        inventory(host, ram_gib),
        HardwareScore("GPU", 4.0, 8.0, 5000, cpu_tok_s, vulkan_tok_s))


class DistributedCapacityPlanTest(unittest.TestCase):
    def test_known_gpt_oss_size_matches_the_catalog(self):
        size = known_model_gigabytes("ggml-org/gpt-oss-20b-GGUF:MXFP4")

        self.assertEqual(size, 12.11)

    def test_known_gpt_oss_layers_are_available_for_dry_run(self):
        self.assertEqual(
            known_model_layers("ggml-org/gpt-oss-20b-GGUF:MXFP4"), 24)

    def test_two_14_gib_machines_are_the_safe_minimum_for_12_gb(self):
        candidates = (
            ranked_node("10.0.0.2", 14, 20.0),
            ranked_node("10.0.0.3", 14, 18.0),
            ranked_node("10.0.0.4", 14, 4.0),
        )

        plan = build_capacity_plan(12.11, candidates, "10.0.0.2")

        self.assertEqual(plan.minimum_total, 2)
        self.assertEqual(
            [node.inventory.host for node in plan.select(2)],
            ["10.0.0.2", "10.0.0.3"])

    def test_model_larger_than_safe_aggregate_memory_is_refused(self):
        candidates = (
            ranked_node("10.0.0.2", 8, 20.0),
            ranked_node("10.0.0.3", 8, 18.0),
        )

        with self.assertRaises(CapacityPlanError) as refused:
            build_capacity_plan(12.11, candidates, "10.0.0.2")

        self.assertIn("12.11", str(refused.exception))
        self.assertIn("8.00", str(refused.exception))

    def test_worker_without_vulkan_is_ranked_by_cpu_tokens(self):
        candidates = (
            ranked_node("10.0.0.2", 14, 20.0),
            ranked_node("10.0.0.3", 14, 0.0, cpu_tok_s=9.0),
            ranked_node("10.0.0.4", 14, 5.0),
        )

        plan = build_capacity_plan(12.11, candidates, "10.0.0.2")

        self.assertEqual(plan.select(2)[1].tokens_per_second, 9.0)
        self.assertEqual(
            [node.inventory.host for node in plan.select(3)],
            ["10.0.0.2", "10.0.0.3", "10.0.0.4"])

    def test_allocation_shows_gib_and_contiguous_layer_ranges(self):
        candidates = (
            ranked_node("10.0.0.2", 14, 20.0),
            ranked_node("10.0.0.3", 14, 18.0),
        )
        plan = build_capacity_plan(12.11, candidates, "10.0.0.2")

        allocation = allocate_model(plan, plan.select(2), model_layers=24)

        self.assertEqual([round(item.model_gigabytes, 2) for item in allocation],
                         [7.0, 5.11])
        self.assertEqual([(item.layer_start, item.layer_end) for item in allocation],
                         [(0, 13), (14, 23)])


if __name__ == "__main__":
    unittest.main()
