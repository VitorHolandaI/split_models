import unittest

from split_lm.llama_rpc.haproxy import haproxy_config


class HaProxyConfigTest(unittest.TestCase):
    def test_config_points_at_every_replica_with_health_checks(self):
        config = haproxy_config(
            (("node-node-a", "10.0.0.2:8080"), ("node-node-b", "10.0.0.3:8080")),
            listen_port=8080)

        self.assertIn("bind *:8080", config)
        self.assertIn("server node-node-a 10.0.0.2:8080 check inter 5s fall 3 rise 2",
                      config)
        self.assertIn("server node-node-b 10.0.0.3:8080 check inter 5s fall 3 rise 2",
                      config)

    def test_active_check_targets_health_endpoint_with_200(self):
        config = haproxy_config((("node-node-a", "10.0.0.2:8080"),))

        # O llama-server responde 503 enquanto o modelo carrega: a checagem
        # tem que exigir 200 e so entao mandar trafego para a replica.
        self.assertIn("option httpchk GET /health", config)
        self.assertIn("http-check expect status 200", config)

    def test_streaming_is_protected_by_tunnel_timeout(self):
        config = haproxy_config((("node-node-a", "10.0.0.2:8080"),))

        # Streaming SSE nao pode ser cortado pelo timeout de resposta normal.
        self.assertIn("timeout tunnel", config)
        self.assertNotIn("option passive", config)

    def test_empty_replica_list_is_refused_with_the_expected_shape(self):
        with self.assertRaises(ValueError) as refused:
            haproxy_config(())

        self.assertIn("servers=()", str(refused.exception))
        self.assertIn("host:porta", str(refused.exception))

    def test_out_of_range_listen_port_is_refused(self):
        with self.assertRaises(ValueError) as refused:
            haproxy_config((("node-node-a", "10.0.0.2:8080"),), listen_port=0)

        self.assertIn("listen_port=0", str(refused.exception))

    def test_server_names_are_sanitized_for_haproxy(self):
        config = haproxy_config((("node a/estranho", "10.0.0.2:8080"),))

        # Nomes HAProxy: somente alfanumericos, ponto, hifen e underscore.
        self.assertIn("server node_a_estranho", config)


if __name__ == "__main__":
    unittest.main()
