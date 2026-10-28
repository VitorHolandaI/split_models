# split-llama fleet bench — Benchmark de capacidade

## Visão geral

Máquinas bootam a ISO, se enxergam na mesma LAN. Uma delas coordena o
benchmark, as outras viram nós de inferência. O worker RPC e o discovery de
controle já sobem automaticamente no boot da ISO — não precisa iniciar nada.

Para usar numa máquina que já tem o repositório clonado (sem ISO),
basta rodar o worker manualmente:

```bash
split-llama worker --bind 192.168.0.10
```

Há dois discoveries intencionalmente separados:

- `split-llama nodes`: lista somente workers RPC realmente disponíveis;
- `split-llama fleet nodes`: lista todos os nós pelo plano de controle,
  inclusive réplicas e masters.

O segundo permanece ativo durante todas as trocas de modo e é usado por
`fleet bench --auto-discover`, `fleet reset --auto-discover`, `fleet reboot` e
pela preparação automática de `master --discover`.

## Fluxo rápido (copiar/colar)

> **Onde roda cada comando:** escolha uma máquina do lab como coordenadora e
> execute os comandos nela. A chave dedicada já é embutida na ISO durante o
> build e recebe modo `0600`; quem possui a ISO controla a frota, portanto use
> somente numa LAN isolada.

```bash
# 0. Na máquina escolhida, para inferência local mas mantém o UDP de controle
split-llama become_master

# 1. Confere todos os nós e seus modos atuais
split-llama fleet nodes --broadcast-address 192.168.0.255

# 2. Bench automático — réplica + HAProxy (testado com 3 nós)
split-llama fleet bench \
  --scenario replicated \
  --model small \
  --profile users-2k \
  --auto-discover \
  --broadcast-address 192.168.0.255 \
  --discovery-timeout 10

# 3. Bench automático — distribuído (master + workers RPC)
split-llama fleet bench \
  --scenario distributed \
  --model small \
  --profile users-2k \
  --auto-discover \
  --broadcast-address 192.168.0.255 \
  --discovery-timeout 10

# 4. Reinicia todos os nós (reboot limpo para começar de novo)
split-llama fleet reboot --broadcast-address 192.168.0.255
```

Resultado em `benchmark-results/capacity-summary.csv` (no nó coordenador,
que na ISO aponta para `/run/split-models/benchmark-results` — a RAM da
máquina, some no reboot; copie os arquivos antes de desligar).

## Pré-requisitos

- ISO construída: `dist/llamacpp-rpc/split-models-llamacpp-rpc-*.iso`
- Ambos os nós na mesma LAN (mesmo switch / segmento L2)
- HAProxy instalado no nó coordenador (`pacman -S haproxy`)
- Chave SSH da frota em `/root/.ssh/fleet_ed25519` com modo `0600`
  (embutida no build da ISO; a pública fica em `authorized_keys`)

## SSH nos nós

Para entrar numa máquina que bootou a ISO, use a chave da appliance
**a partir da sua máquina** (onde o repo está clonado):

```bash
ssh -i deploy/keys/split-models-appliance root@192.168.0.10
```

Ou depois de copiar para `~/.ssh`:

```bash
cp deploy/keys/split-models-appliance ~/.ssh/
ssh -i ~/.ssh/split-models-appliance root@192.168.0.10
```

Para o `fleet` funcionar de dentro de um nó, a chave fica no caminho padrão
`/root/.ssh/fleet_ed25519`. Em uma ISO antiga que mostre `Permissions ... are
too open`, corrija temporariamente:

```bash
chmod 600 /root/.ssh/fleet_ed25519
```

## Reiniciar todos os nós

```bash
# Descobre todos pelo UDP de controle e manda reboot, mesmo em modo réplica
split-llama fleet reboot

# Ou com broadcast específico da rede
split-llama fleet reboot --broadcast-address 192.168.0.255
```

## Fluxo manual (sem fleet — você vai na máquina e faz)

Entre na máquina que será o coordenador (console, SSH, físico). Lá:

```bash
# 1. Descobre todos os nós e seus modos no plano de controle
split-llama fleet nodes

# 2. Prepara a máquina coordenadora sem matar seu discovery
split-llama become_master

# 3. Escolhe o modo:

# Opção A — réplica (carrega o modelo e serve HTTP sozinho)
split-llama mode replica --model small --port 8080

# Opção B — master distribuído (usa workers RPC descobertos)
split-llama mode distributed-master \
  --model small \
  --rpc 192.168.0.10:50052,192.168.0.20:50052

# 4. Bench contra a própria máquina
split-llama bench-capacity --profile users-2k \
  --scenario replicated --endpoint http://127.0.0.1:8080

# 5. Para voltar ao worker depois (transição supervisionada)
split-llama mode worker
```

## Conversa distribuída após replicated

`master --discover` não reutiliza endpoints RPC mortos. Ele descobre todos os
nós pelo plano de controle, executa `split-llama mode worker` via SSH em cada
um e somente depois descobre os workers RPC vivos. Portanto pode ser executado
diretamente após um cenário replicated:

```bash
split-llama master --discover \
  --hf unsloth/Qwen3.5-9B-GGUF:Q6_K \
  -- -c 32768
```

### Benchmark de hardware e planejamento

Antes de dividir, `master --discover` mede cada nó descoberto via SSH
(`split-llama hardware-bench --json` em cada um) e imprime a tabela:

```text
NODE            RAM GiB    TOPS    MIPS  tok/s CPU tok/s Vulkan
10.0.0.2           14.0    8.00    5000       3.00        10.00
10.0.0.4           14.0    8.00    5000       3.00        20.00
```

Unidades medidas por `split-llama hardware-bench`:

- **TOPS / TFLOPS** — picos sintéticos da iGPU (vkpeak): matriz INT8
  (tera-operations/s) e FP16 (tera-floating-point ops/s).
- **MIPS** — rating inteiro da CPU do `7z b`, mantido na unidade nativa.
- **tok/s CPU / Vulkan** — throughput real do llama.cpp com o modelo pequeno
  oficial (`unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M`, prompt 128, geração 32): é o que
  decide a ordem dos workers.

Máquinas sem Vulkan (ou sem `int8-matrix`) não derrubam o planejamento: o
hardware-bench zera iGPU e tok/s Vulkan com aviso, e o nó é ordenado pelo
tok/s da CPU. Depois da medição o comando pergunta quantas máquinas usar no
total (master incluso), entre o mínimo seguro calculado (50% da RAM de cada
nó somada) e o máximo descoberto; `--machines N` pula a pergunta. Workers são
ordenados por tok/s Vulkan e os não selecionados ficam fora dos endpoints RPC.

### Cache do modelo: RAM por padrão, USB somente no master split

Todos os caminhos de runtime ficam em `/run`, que é RAM na ISO live. O disco
original nunca é formatado, montado ou usado para dados do modelo. No cenário
distributed, somente o master pode receber `--usb-device /dev/sda1` para guardar
o GGUF completo; os workers recebem apenas as layers via RPC e as mantêm em
RAM. Em replicated, cada nó mantém a cópia completa em RAM.

Sem `--usb-device`, o modo de servidor usa o cache temporário em RAM. O espaço
livre é validado **antes** de apagar um download anterior no USB (reserva de
0.5 GB), então um modelo que não cabe não destrói o cache existente.

### Modelo grande no USB de 16 GB

O maior GGUF do catálogo cabe em um USB ext4 de 16 GB:
`ggml-org/gpt-oss-20b-GGUF:MXFP4` tem 12.11 GB medidos (reserva de 0.5 GB
inclusa, ~15.5 GB livres no USB):

```bash
split-llama master --discover \
  --usb-device /dev/sda1 \
  --hf ggml-org/gpt-oss-20b-GGUF:MXFP4 \
  -- -c 32768
```

O master lê o GGUF completo do USB; os workers recebem apenas as layers
atribuídas via RPC.

Antes de alterar a frota, o plano pode ser inspecionado sem `--machines`:

```bash
split-llama master --discover --dry-run \
  --hf ggml-org/gpt-oss-20b-GGUF:MXFP4
```

O comando mede a capacidade por SSH, escolhe o menor prefixo seguro e mostra a
RAM segura, a parcela estimada do modelo e a faixa contínua de layers de cada
máquina. Modelos conhecidos têm a quantidade de layers detectada
automaticamente. Essas faixas são proporcionais; o `llama.cpp` pode ajustá-las
pelos tamanhos reais dos tensores. Para modelo fora do catálogo, informe também
`--model-size-gb N` e, se quiser faixas, `--model-layers N`.

O master baixa o GGUF completo para seu cache. Durante o load, os
workers recebem somente os tensors/layers atribuídos via RPC; cada worker mantém
os pesos e o KV cache dessas layers. O master continua limitado pela RAM
necessária para o GGUF completo, sua parcela local, runtime, sistema e contexto.
O repositório e o tamanho foram conferidos; esse 9B ainda precisa ser validado
por inferência no laboratório físico.

O cache mantém apenas um modelo selecionado. Repetir o mesmo `--hf` reutiliza o
download; selecionar outro remove o anterior antes de baixar o novo.

Para HAProxy (cenário réplica com várias máquinas), gere o `haproxy.cfg`
e suba na máquina que alcança todas as réplicas:

```bash
# Gera config apontando para as réplicas
uv run python -c "
from split_lm.llama_rpc.haproxy import haproxy_config
print(haproxy_config((
  ('node1', '192.168.0.10:8080'),
  ('node2', '192.168.0.20:8080'),
)))
" > haproxy.cfg

# Sobe o HAProxy
haproxy -f haproxy.cfg

# Bench contra o HAProxy
split-llama bench-capacity --profile users-2k \
  --scenario replicated --endpoint http://127.0.0.1:8080
```

## Fluxo automático (fleet — via SSH de qualquer máquina)

No nó coordenador, um comando só — descobre os nós, promove, sobe HAProxy,
roda bench:

```bash
# Cenário réplica (cada nó carrega o modelo + HAProxy distribui)
split-llama fleet bench \
  --scenario replicated \
  --model small \
  --profile users-2k \
  --auto-discover \
  --broadcast-address 255.255.255.255 \
  --discovery-timeout 10

# Cenário distribuído (1 llama-server + layers divididas via RPC)
split-llama fleet bench \
  --scenario distributed \
  --model small \
  --profile users-2k \
  --auto-discover
```

O que ele faz automaticamente:
1. Escuta o UDP de controle por 10s e descobre nós em qualquer modo
2. SSH em cada nó e executa `split-llama mode worker` para limpar o papel anterior
3. Em replicated, promove todos a réplica; em distributed, promove somente um
   master e mantém os demais como workers RPC
4. Gera `haproxy.cfg` e sobe o HAProxy local na porta 8088
5. Aguarda `/health` 200 em todos os nós
6. Executa `bench-capacity` contra o endpoint
7. Salva resultado em `benchmark-results/capacity-summary.csv`
8. Mata o HAProxy

## Fluxo passo a passo (manual)

### 1. Inventário (descobrir os IPs e estado dos nós)

```bash
split-llama fleet inventory --nodes 192.168.0.10,192.168.0.20
```

### 2. Cenário distribuído

```bash
# Promove
split-llama fleet distributed \
  --master 192.168.0.10 \
  --workers 192.168.0.20 \
  --model small

# Bench contra o master
split-llama bench-capacity \
  --profile users-2k \
  --scenario distributed \
  --endpoint http://192.168.0.10:8080
```

### 3. Cenário réplica + HAProxy

```bash
# Promove e gera haproxy.cfg
split-llama fleet replicated \
  --replicas 192.168.0.10,192.168.0.20 \
  --model small

# Sobe o HAProxy (no nó coordenador)
haproxy -f haproxy.cfg

# Bench contra o HAProxy
split-llama bench-capacity \
  --profile users-2k \
  --scenario replicated \
  --endpoint http://127.0.0.1:8080
```

### 4. Manual sem fleet (dentro de cada nó via SSH)

Se preferir fazer na mão, dentro de cada máquina:

```bash
# Nó vira réplica (carrega modelo + serve HTTP)
systemctl stop split-llama-worker.service
split-llama mode replica --model small --port 8080

# Nó vira master distribuído (precisa dos workers)
split-llama mode distributed-master \
  --model small \
  --rpc 192.168.0.10:50052,192.168.0.20:50052
```

### 5. Reset (volta ao worker)

```bash
# Descobre os nós de controle e volta todos a worker por SSH, sem reboot
split-llama fleet reset --auto-discover

# Ou informe os IPs manualmente
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

### O que significa `total_tok_s`

`total_tok_s` é a taxa média **por requisição**, ponderada por tokens:

```text
Σ tokens das requisições OK / Σ duração e2e das requisições OK
```

O denominador é a soma das durações, não o tempo de parede da onda. Com
requisições de duração parecida, o valor coincide com o tok/s de uma
requisição isolada e **não cresce** com o número de usuários. Quando a
latência sobe, ele cai — foi o que aconteceu no distribuído, de `9.38` para
`4.15 tok/s` entre 2 e 8 usuários.

O `tok_s` do JSONL é outra métrica: `tokens / e2e_s` daquela requisição
individual.

## Modelos oficiais

| Perfil | Modelo | Tamanho | RAM mínima por nó |
|---|---|---|---|
| `small` | Qwen3.5-0.8B (Q4_K_M) | 0.53 GB | 4 GB |
| `large` | Qwen3.5-4B (Q4_K_M) | 2.74 GB | 8 GB |

## Evidências de execução

Fotos recortadas de execuções reais com o código deste repositório, anteriores
à política de cache USB:

- [Distribuído, 12 nós, 2–8 usuários](docs/evidence/2026-08-21-fleet-bench-12-nodes/distributed-users-2-4-8.jpg)
- [Replicado, 12 nós, 2–32 usuários](docs/evidence/2026-08-21-fleet-bench-12-nodes/replicated-users-2-32.jpg)
- [Índice das evidências](docs/evidence/README.md)
