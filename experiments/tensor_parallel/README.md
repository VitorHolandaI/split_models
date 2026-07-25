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
decode MLP TCP, 2 threads/shard: baseline 3.100 ms, distribuido 1.107 ms, 2.802x
prefill MLP, 32:         baseline 5.932 ms, distribuido 9.677 ms, 0.613x
prefill FP32/BF16 wire:  baseline 9.716 ms, distribuido 4.026 ms, 2.413x
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
com protocolo binario. Os dois ranks enviam seus parciais simultaneamente,
recebem o parcial remoto e somam localmente. Isso evita o ping-pong serial de uma
ida e volta completa. Setup, distribuicao inicial de pesos e barreiras continuam
usando Gloo e ficam fora do caminho steady-state medido.

```bash
uv run python -m experiments.tensor_parallel.benchmark \
  --nodes 2 --tokens 1 \
  --input-features 896 --output-features 4864 \
  --mode mlp --collective tcp --worker-threads 2
```

Aumentar artificialmente o pacote nao reduz a latencia. Para amortiza-la com
trabalho real, usar prefill (`--tokens 32`) ou batch de varias sequencias.

`--wire-dtype bfloat16` mantem as multiplicacoes em FP32 e converte apenas o
tensor enviado. Isso reduz os bytes pela metade, mas e uma compressao com perda.
Na medicao de 32 tokens, o erro relativo foi aproximadamente 0.2%. A integracao
com o modelo real precisa comparar tokens e qualidade antes de habilitar isso.

```bash
--dtype float32 --wire-dtype bfloat16
```

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
  --iterations 50 --warmup 5 --threads 14 --worker-threads 2 \
  --dtype float32 --wire-dtype bfloat16 --mode mlp
```

Remoto `192.168.0.11`:

```bash
uv run python -m experiments.tensor_parallel.benchmark \
  --nodes 2 --rank 1 \
  --master-addr 192.168.0.8 --master-port 29500 --interface enp37s0 \
  --collective tcp --collective-port 29501 \
  --tokens 1 --input-features 896 --output-features 4864 \
  --iterations 50 --warmup 5 --threads "$(nproc)" --worker-threads 2 \
  --dtype float32 --wire-dtype bfloat16 --mode mlp
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

## Sweep E Grafico De Threshold

O sweep aumenta os tokens processados juntos, mede baseline local e distribuido
no mesmo ponto e gera CSV + dois PNGs. Antes de cada tamanho, todos os ranks estimam
pesos e ativacoes; a execucao para se qualquer maquina ultrapassar a fracao de
RAM configurada.

Instalar Matplotlib apenas na maquina que gera o grafico:

```bash
uv sync --extra plots
```

Sweep local:

```bash
uv run python -m experiments.tensor_parallel.benchmark \
  --nodes 2 \
  --token-sweep 1,2,4,8,16,32,64,128,256 \
  --input-features 896 --output-features 4864 \
  --iterations 20 --warmup 5 \
  --threads 14 --worker-threads 4 \
  --dtype float32 --wire-dtype bfloat16 \
  --collective tcp --mode mlp \
  --max-memory-fraction 0.5 \
  --output-prefix benchmark-results/local_threshold
```

Para o sweep fisico, adicionar a mesma lista `--token-sweep` aos comandos de
rank 0 e rank 1 desta pagina. No rank 0, usar:

```text
--max-memory-fraction 0.5
--output-prefix benchmark-results/physical_threshold
```

O rank 0 grava:

```text
benchmark-results/physical_threshold.csv
benchmark-results/physical_threshold.png
benchmark-results/physical_threshold_speedup.png
```

O primeiro PNG usa tokens no eixo X e tempo medio por iteracao no eixo Y, com
uma curva para o MLP local e outra para o distribuido. A primeira medicao com
`speedup >= 1.0x` e marcada como threshold. O PNG `_speedup` usa tokens no eixo
Y e speedup atingido no eixo X; vermelho perdeu e verde ganhou do local. A
velocidade efetiva mostra tambem a formula:

```text
(bytes reais por iteracao x 8) / segundos de comunicacao / 1.000.000
```
