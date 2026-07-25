# Tensor Parallel Experiment

Este experimento mede se dividir as multiplicacoes de matrizes compensa o custo
de comunicacao antes de alterar o backend do LLM.

## Operacoes

- column parallel: divide as colunas de `QKV`, `gate` e `up`; cada no calcula
  parte da saida e os resultados sao concatenados;
- row parallel: divide a dimensao de entrada de `o_proj` e `down`; cada no
  calcula uma contribuicao e as saidas passam por `all_reduce`.
- MLP sharded: mantem `gate/up` divididos ate o `down`, sem gather/scatter do
  tensor intermediario; somente a saida hidden passa por `all_reduce`.

PyTorch usa Intel MKL, oneDNN e AVX2 para o matmul. Gloo executa os collectives
em C++, portanto o GIL do Python nao participa do caminho medido.

## Uso

Decode equivalente ao `up_proj` do Qwen2.5-0.5B:

```bash
uv run python -m experiments.tensor_parallel.benchmark \
  --nodes 2 --tokens 1 \
  --input-features 896 --output-features 4864 \
  --mode column
```

Prefill com 32 tokens:

```bash
uv run python -m experiments.tensor_parallel.benchmark \
  --nodes 2 --tokens 32 \
  --input-features 896 --output-features 4864 \
  --mode mlp
```

## Medicoes Locais

Core Ultra 5 225H, FP32, 14 threads totais divididas entre dois processos:

```text
decode column 896x4864: baseline 0.365 ms, distribuido 1.710 ms, 0.213x
decode row 4864x896:    baseline 0.514 ms, distribuido 5.774 ms, 0.089x
prefill column, 32:     baseline 1.742 ms, distribuido 2.688 ms, 0.648x
prefill row, 32:        baseline 4.826 ms, distribuido 6.224 ms, 0.775x
decode MLP Gloo:        baseline 3.261 ms, distribuido 4.649 ms, 0.702x
decode MLP TCP:         baseline 3.314 ms, distribuido 3.617 ms, 0.916x
prefill MLP, 32:         baseline 5.932 ms, distribuido 9.677 ms, 0.613x
```

O resultado local e mais lento porque os processos dividem os mesmos cores e a
mesma largura de memoria. Em maquinas fisicas, cada no adiciona compute, mas a
rede precisa superar o custo do all-reduce por camada.

O MLP completo reduziu o trafego teorico minimo de decode de aproximadamente
30 KB para 7 KB por camada ao eliminar gather/scatter intermediario. A entrada
replicated e sincronizada uma vez no setup; o steady-state comunica apenas a
saida hidden do `down_proj`. Dois processos locais ainda dividem os mesmos cores.

Rust pode reduzir overhead de transporte, mas nao acelera o matmul por si so.
Para superar este kernel, a implementacao Rust tambem precisaria usar MKL ou
oneDNN. Um worker Rust so se justifica se o profiling mostrar que transporte e
coordenacao, nao os collectives, dominam o tempo.

## Collective TCP Para Dois Nos

`--collective tcp` substitui o all-reduce Gloo do MLP por uma conexao persistente
com protocolo binario: rank 1 envia seu parcial, rank 0 soma e devolve o hidden
final. Setup, distribuicao inicial de pesos e barreiras continuam usando Gloo e
ficam fora do caminho steady-state medido.

```bash
uv run python -m experiments.tensor_parallel.benchmark \
  --nodes 2 --tokens 1 \
  --input-features 896 --output-features 4864 \
  --mode mlp --collective tcp
```

Aumentar artificialmente o pacote nao reduz a latencia. Para amortiza-la com
trabalho real, usar prefill (`--tokens 32`) ou batch de varias sequencias.

O relatorio separa calculo de comunicacao:

```text
matmuls_per_rank=3       gate, up e down por iteracao
collective_calls=50      um reduce por iteracao medida
sent_frames=100          dois frames TCP por reduce em dois ranks
sent_bytes=...           soma real dos frames, excluindo warmup
```

Gloo nao expoe sua fragmentacao TCP interna; nesse modo, frames e bytes aparecem
como `gloo-managed` e `min_network` continua sendo o limite teorico do tensor.

## Duas Maquinas Fisicas

As duas maquinas precisam usar os mesmos argumentos de modelo matematico. O
head `192.168.0.8` atua como rank 0 e rendezvous Gloo.

Head `192.168.0.8` (`wlan0`):

```bash
uv run python -m experiments.tensor_parallel.benchmark \
  --nodes 2 --rank 0 \
  --master-addr 192.168.0.8 --master-port 29500 --interface wlan0 \
  --collective tcp --collective-port 29501 \
  --tokens 1 --input-features 896 --output-features 4864 \
  --iterations 50 --warmup 5 --threads 14 --mode mlp
```

Remoto `192.168.0.11`:

```bash
uv run python -m experiments.tensor_parallel.benchmark \
  --nodes 2 --rank 1 \
  --master-addr 192.168.0.8 --master-port 29500 --interface enp37s0 \
  --collective tcp --collective-port 29501 \
  --tokens 1 --input-features 896 --output-features 4864 \
  --iterations 50 --warmup 5 --threads "$(nproc)" --mode mlp
```

Iniciar rank 0 primeiro e rank 1 em seguida. Se a interface do remoto mudar,
descobrir com `ip route get 192.168.0.8` e ajustar `--interface`. Gloo
pode abrir conexoes TCP adicionais entre os ranks; o firewall deve permitir
trafego TCP entre `192.168.0.8` e `192.168.0.11`, nao apenas a porta de
rendezvous `29500`.

O enlace medido com iperf3 atingiu aproximadamente 624 Mbit/s sem retransmissoes.
Nesse throughput, 7 KB levam cerca de 0.09 ms para transmitir; tempos de
comunicacao muito maiores sao latencia/sincronizacao do collective, nao falta de
banda sustentada.
