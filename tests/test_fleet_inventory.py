import json
import unittest

from split_lm.llama_rpc.fleet_inventory import (
    FleetInventory, InventoryError, NodeInventory, inventory_sections_command,
    parse_node_inventory)


def sample_sections(ip: str, hostname: str, role: str = "worker",
                    memtotal_kb: int = 16384000,
                    commit: str = "abc1234def567890") -> str:
    """A stdout de inventario completo e valido para um no."""
    return (
        f"SECTION hostname\n{hostname}\n"
        f"SECTION ip\n{ip}  10.0.0.99\n"
        f"SECTION memtotal\nMemTotal: {memtotal_kb} kB\n"
        f"SECTION status\nrole={role} | node=node-{hostname} | ip={ip}\n"
        f"SECTION build\n{commit}\n"
        "SECTION devices\n"
        "llama_1 device: Vulkan0 (Intel Arc A770)\n"
        "llama_2 device: CPU\n"
    )


class InventorySectionsTest(unittest.TestCase):
    def test_command_is_one_round_trip_per_node(self):
        command = inventory_sections_command()

        self.assertEqual(command[0], "bash")
        self.assertEqual(command[1], "-c")
        script = command[2]
        for section in ("hostname", "ip", "memtotal", "status", "build",
                        "devices"):
            self.assertIn(f"SECTION {section}", script)
        self.assertIn("tr '\\n' ' '; printf '\\n'", script)


class InventoryParseTest(unittest.TestCase):
    def test_exact_fields_are_extracted_from_the_sections(self):
        node = parse_node_inventory(
            sample_sections("10.0.0.2", "node-a-host"), "node-a")

        self.assertEqual(node.host, "node-a")
        self.assertEqual(node.node_id, "node-node-a-host")
        self.assertEqual(node.hostname, "node-a-host")
        self.assertEqual(node.ip, "10.0.0.2")
        self.assertEqual(node.mem_total_kb, 16384000)
        self.assertEqual(node.devices,
                         ("llama_1 device: Vulkan0 (Intel Arc A770)",
                          "llama_2 device: CPU"))
        self.assertEqual(node.build_commit, "abc1234def567890")
        self.assertEqual(node.mode, "worker")

    def test_missing_memtotal_names_the_offending_value_and_shape(self):
        stdout = sample_sections("10.0.0.2", "node-a-host").replace(
            "MemTotal: 16384000 kB", "MemTotal: duvidoso")

        with self.assertRaises(InventoryError) as failed:
            parse_node_inventory(stdout, "node-a")

        reported = str(failed.exception)
        self.assertIn("node-a", reported)
        self.assertIn("duvidoso", reported)
        self.assertIn("MemTotal", reported)

    def test_malformed_status_names_the_expected_brief_shape(self):
        stdout = sample_sections("10.0.0.2", "node-a-host").replace(
            "role=worker | node=node-node-a-host | ip=10.0.0.2",
            "papel=worker")

        with self.assertRaises(InventoryError) as failed:
            parse_node_inventory(stdout, "node-a")

        self.assertIn("papel=worker", str(failed.exception))
        self.assertIn("role=", str(failed.exception))

    def test_missing_section_is_an_explicit_error(self):
        stdout = sample_sections("10.0.0.2", "node-a-host").replace(
            "SECTION memtotal\nMemTotal: 16384000 kB\n", "")

        with self.assertRaises(InventoryError) as failed:
            parse_node_inventory(stdout, "node-a")

        self.assertIn("node-a", str(failed.exception))
        self.assertIn("memtotal", str(failed.exception))

    def test_missing_build_info_and_devices_are_recorded_not_silent(self):
        stdout = (
            "SECTION hostname\nnode-b-host\n"
            "SECTION ip\n10.0.0.3\n"
            "SECTION memtotal\nMemTotal: 8388608 kB\n"
            "SECTION status\nrole=worker | node=node-b-host | ip=10.0.0.3\n"
            "SECTION build\n\n"
            "SECTION devices\n\n"
        )

        node = parse_node_inventory(stdout, "node-b")

        self.assertEqual(node.build_commit, "unknown")
        self.assertEqual(node.devices, ())

    def test_status_ip_wins_over_the_hostname_i_list(self):
        node = parse_node_inventory(
            sample_sections("10.0.0.2", "node-a-host"), "node-a")

        # `hostname -I` lista varias interfaces; o ip da rota de LAN gravado
        # pelo proprio no e o que o master e o HAProxy devem usar.
        self.assertEqual(node.ip, "10.0.0.2")
        self.assertNotEqual(node.ip, "10.0.0.99")


class NodeInventorySerializationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.node = parse_node_inventory(
            sample_sections("10.0.0.2", "node-a-host"), "node-a")

    def test_record_round_trips_exactly(self):
        restored = NodeInventory.from_record(self.node.to_record())

        self.assertEqual(restored, self.node)

    def test_record_with_wrong_shape_is_refused_with_the_expected_fields(self):
        record = dict(self.node.to_record())
        del record["mode"]

        with self.assertRaises(ValueError) as failed:
            NodeInventory.from_record(record)

        self.assertIn("mode", str(failed.exception))


class FleetInventorySerializationTest(unittest.TestCase):
    def test_inventory_survives_a_json_round_trip(self):
        first = parse_node_inventory(
            sample_sections("10.0.0.2", "node-a-host"), "node-a")
        second = parse_node_inventory(
            sample_sections("10.0.0.3", "node-b-host"), "node-b")
        inventory = FleetInventory("2026-08-05T12:00:00+00:00",
                                   (first, second))

        restored = FleetInventory.from_json(inventory.to_json())

        self.assertEqual(restored, inventory)

    def test_inventory_record_carries_hosts_and_exact_fields(self):
        node = parse_node_inventory(
            sample_sections("10.0.0.2", "node-a-host"), "node-a")

        record = json.loads(FleetInventory("ts", (node,)).to_json())

        self.assertEqual(record["nodes"][0]["host"], "node-a")
        self.assertEqual(record["nodes"][0]["mem_total_kb"], 16384000)
        self.assertEqual(record["nodes"][0]["build_commit"],
                         "abc1234def567890")


if __name__ == "__main__":
    unittest.main()
