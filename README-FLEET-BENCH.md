# split-llama fleet bench — Benchmark de capacidade

## Visão geral

Duas máquinas (ou VMs) bootam a ISO e se enxergam na mesma LAN. Uma delas
coordena o benchmark e a outra (ou ambas) vira nó de inferência.

## Pré-requisitos

- ISO construída: `dist/llamacpp-rpc/split-models-llamacpp-rpc-*.iso`
- Ambos os nós na mesma LAN (mesmo switch / segmento L2)
- HAProxy instalado no nó coordenador (`pacman -S haproxy`)

## Fluxo completo (automático)

No nó coordenador, um comando só:

```bash
# Descobre os nós via UDP broadcast, promove a réplica, sobe HAProxy, roda bench
split-llama fleet bench \
  --scenario replicated \
  --model small \
  --profile users-2k \
  --auto-discover \
  --broadcast-address 255.255.255.255 \
  --discovery-timeout 10
```

O que ele faz:
1. Escuta UDP broadcast por 10s → descobre os IPs dos workers
2. SSH em cada nó → promove a `replica` (carrega o modelo)
3. Gera `haproxy.cfg` apontando para todas as réplicas
4. Sobe o HAProxy na porta 8080
5. Aguarda `/health` 200 em todos os nós
6. Executa `bench-capacity` contra o HAProxy
7. Salva resultado em `benchmark-results/capacity-summary.csv`
8. Mata o HAProxy

## Fluxo passo a passo (manual)

Se o `--auto-discover` não funcionar na sua rede:

### 1. Inventário

```bash
split-llama fleet inventory --nodes 192.168.0.10,192.168.0.20
```

### 2. Cenário distribuído (1 llama-server + layers divididas via RPC)

```bash
split-llama fleet distributed \
  --master 192.168.0.10 \
  --workers 192.168.0.20 \
  --model small
```

Bench contra o master:
```bash
split-llama bench-capacity \
  --profile users-2k \
  --scenario distributed \
  --endpoint http://192.168.0.10:8080
```

### 3. Cenário réplica + HAProxy (cada nó serve o modelo completo)

```bash
split-llama fleet replicated \
  --replicas 192.168.0.10,192.168.0.20 \
  --model small \
  --haproxy-file haproxy.cfg

# Sobe o HAProxy
haproxy -f haproxy.cfg

# Bench contra o HAProxy
split-llama bench-capacity \
  --profile users-2k \
  --scenario replicated \
  --endpoint http://127.0.0.1:8080
```

### 4. Reset (volta todos os nós a worker)

```bash
split-llama fleet reset --nodes 192.168.0.10,192.168.0.20
```

## Perfis de benchmark

| Perfil | O que varia | Fixo | Pontos |
|---|---|---|---|
| `users-2k` | usuários (2,4,8,16,32) | contexto 2048 | 5 |
| `context-2users` | contexto (512,1024,2048,4096,8192) | 2 usuários | 5 |
| `nodes-8users-2k` | nós (1,2,4,8) | 8 usuários, ctx 2048 | 4 |
| `capacity-envelope` | usuários (2,4,8,16) × contexto (1024,2048,4096) | — | 12 |

Cada ponto: 1 warmup + 3 medições, 128 tokens gerados.

## Saída

`benchmark-results/capacity-summary.csv` com colunas:
`profile, scenario, users, context, nodes, requests_ok, requests_failed,
tokens_generated, p50_ttft_s, p95_ttft_s, p99_ttft_s, p50_e2e_s, p95_e2e_s,
p99_e2e_s, total_tok_s, busy_slots_peak, saturated`

## Modelos oficiais

| Perfil | Modelo | Tamanho | RAM mínima por nó |
|---|---|---|---|
| `small` | Qwen3.5-0.8B (Q4_K_M) | 0.53 GB | 4 GB |
| `large` | Qwen3.5-4B (Q4_K_M) | 2.74 GB | 8 GB |