# Tensor Parallel Experiment

Este experimento mede se dividir as multiplicacoes de matrizes compensa o custo
de comunicacao antes de alterar o backend do LLM.

Fluxo completo do benchmark:

- [Diagrama SVG](../../docs/diagrams/tensor-parallel-benchmark.svg)
- [Fonte Mermaid](../../docs/diagrams/tensor-parallel-benchmark.mmd)

## TODO

- [ ] Testar um modelo real com checkpoint acima de 6 GB, carregando apenas o
  shard necessario em cada maquina, e comparar decode de um token, prefill,
  uso de RAM, speedup e qualidade dos logits com BF16 na rede.

## Operacoes

- column parallel: divide as colunas de `QKV`, `gate` e `up`; cada no calcula
  parte da saida e os resultados sao concatenados;
- row parallel: divide a dimensao de entrada de `o_proj` e `down`; cada no
  calcula uma contribuicao e as saidas passam por `all_reduce`.
- MLP sharded: mantem `gate/up` divididos ate o `down`, sem gather/scatter do
  tensor intermediario; somente a saida hidden passa por `all_reduce`.

## Split Vertical Vs. Tensor Parallel

O split vertical nao acelera um unico token porque as duas maquinas nao calculam
a mesma camada simultaneamente. Porem, ele permite carregar um modelo que nao
cabe em uma maquina.

| Caracteristica | Split vertical | Tensor parallel |
|---|---|---|
| Divide | Camadas | Matrizes de cada camada |
| Objetivo principal | Capacidade | Capacidade e possivel velocidade |
| Comunicacao | Entre cortes do modelo | Em praticamente todas as camadas |
| Execucao de um token | Sequencial | Paralela |
| Complexidade | Menor | Maior |
| Rede lenta | Mais adequado | Muito prejudicado |
| Speedup de um usuario | Geralmente nao | Apenas com interconexao rapida |
| Throughput em batch | Pipeline pode ajudar | Pode ajudar bastante |

No hardware atual, o split vertical e a opcao mais simples quando o modelo nao
cabe em uma maquina. Se o modelo cabe e o objetivo e a menor latencia para um
usuario, executar em uma unica maquina continua sendo a melhor opcao. Tensor
parallel passa a fazer sentido para prefill grande, batches ou uma interconexao
de baixa latencia.

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

## O Que `--tokens` Significa

> [!IMPORTANT]
> **Nao existe exatamente um "super token".** O token continua sendo um ID, mas
> em um modelo maior ele e representado por um vetor muito maior e passa por
> matrizes muito maiores.

### Modelo Atual

No Qwen2.5-0.5B usado como referencia:

```text
hidden_size = 896
intermediate_size = 4864
```

Um token entra no MLP como:

```text
[1, 896]
```

As tres multiplicacoes custam:

```text
gate: 896 × 4864
up:   896 × 4864
down: 4864 × 896
```

`--tokens 128` cria uma entrada com 128 vetores de tokens processados ao mesmo
tempo:

```text
entrada = [128, 896]
```

Nao e uma multiplicacao que representa 128 tokens. O matmul calcula as 128
linhas em uma operacao grande:

```text
[128, 896] @ [896, 4864] = [128, 4864]
```

Matematicamente, isso equivale a transformar cada linha separadamente:

```text
token 1   [1, 896] @ peso
token 2   [1, 896] @ peso
...
token 128 [1, 896] @ peso
```

PyTorch agrupa essas linhas em um unico matmul, que aproveita melhor os kernels
de matriz. O MLP completo faz tres matmuls para o bloco inteiro:

```text
gate: [128, 896]  @ [896, 4864]
up:   [128, 896]  @ [896, 4864]
down: [128, 4864] @ [4864, 896]
```

Com duas maquinas, cada uma calcula metade da dimensao intermediaria:

```text
gate local: [128, 896]  @ [896, 2432]
up local:   [128, 896]  @ [896, 2432]
down local: [128, 2432] @ [2432, 896]
```

Cada rank produz um parcial `[128, 896]`; os dois parciais sao trocados e
somados. Assim, `--tokens 128` pode representar o prefill de um prompt com 128
tokens ou um batch de 128 sequencias. Nao representa gerar 128 tokens em ordem.
Com KV cache, o decode interativo de um usuario normalmente processa apenas uma
nova linha por passo, equivalente a `--tokens 1`.

No sweep fisico com 10 workers por maquina, `--tokens 1` levou `1.431 ms` no
baseline e `4.512 ms` distribuido: `0.317x`. O matmul de uma linha termina rapido,
mas a troca, espera, soma e sincronizacao continuam existindo. Mais threads
reduzem compute, mas nao removem essa latencia fixa da rede.

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
head `192.0.2.8` atua como rank 0 e rendezvous Gloo.

Head `192.0.2.8` (`wlan0`):

```bash
uv run python -m experiments.tensor_parallel.benchmark \
  --nodes 2 --rank 0 \
  --master-addr 192.0.2.8 --master-port 29500 --interface wlan0 \
  --collective tcp --collective-port 29501 \
  --tokens 1 --input-features 896 --output-features 4864 \
  --iterations 50 --warmup 5 --threads 14 --worker-threads 2 \
  --dtype float32 --wire-dtype bfloat16 --mode mlp
```

Remoto `192.0.2.11`:

```bash
uv run python -m experiments.tensor_parallel.benchmark \
  --nodes 2 --rank 1 \
  --master-addr 192.0.2.8 --master-port 29500 --interface enp37s0 \
  --collective tcp --collective-port 29501 \
  --tokens 1 --input-features 896 --output-features 4864 \
  --iterations 50 --warmup 5 --threads "$(nproc)" --worker-threads 2 \
  --dtype float32 --wire-dtype bfloat16 --mode mlp
```

Iniciar rank 0 primeiro e rank 1 em seguida. Se a interface do remoto mudar,
descobrir com `ip route get 192.0.2.8` e ajustar `--interface`. Gloo
pode abrir conexoes TCP adicionais entre os ranks; o firewall deve permitir
trafego TCP entre `192.0.2.8` e `192.0.2.11`, nao apenas a porta de
rendezvous `29500`.

O enlace medido com iperf3 atingiu aproximadamente 624 Mbit/s sem retransmissoes.
Nesse throughput, 7 KB levam cerca de 0.09 ms para transmitir; tempos de
comunicacao muito maiores sao latencia/sincronizacao do collective, nao falta de
banda sustentada.

## Comparacao De Interconexoes

Para converter largura de banda, dividir `Gbit/s` por 8 produz `GB/s`. Os valores
teoricos nao incluem headers nem overhead de software:

| Interconexao | Largura aproximada | Comparacao com o enlace medido |
|---|---:|---:|
| Ethernet medida por iperf3 | 0.078 GB/s | 1x |
| Ethernet 1 Gbit/s teorica | 0.125 GB/s | 1.6x |
| Ethernet 10 Gbit/s teorica | 1.25 GB/s | 16x |
| Thunderbolt 4, link bruto de 40 Gbit/s | 5 GB/s | 64x |
| Thunderbolt 5, link bruto bidirecional de 80 Gbit/s | 10 GB/s | 128x |
| Thunderbolt 5 Bandwidth Boost, 120 Gbit/s em um sentido | 15 GB/s | 192x |
| InfiniBand NDR 400 Gbit/s por porta | 50 GB/s | 641x |
| Memoria compartilhada DDR5, exemplo de CPU | 50-100 GB/s | 641-1282x |
| NVLink de uma H100 SXM, agregado | 900 GB/s | 11538x |

No resultado de 6000 tokens com 10 workers, a aplicacao mediu `664.2 Mbit/s`, ou
`0.083 GB/s`, agregados nos dois sentidos. Esse numero pode superar o iperf3 de
um sentido porque os dois ranks enviam simultaneamente. Ele nao significa que um
unico sentido da Ethernet ultrapassou sua capacidade.

Para um token, os dois ranks enviam apenas 3584 bytes BF16 no total. A `0.078
GB/s`, serializar esse volume levaria aproximadamente `0.046 ms`, mas a
comunicacao medida foi `3.619 ms`. Portanto, decode de um usuario e limitado
principalmente por latencia, sincronizacao e software, nao apenas por GB/s. Uma
interconexao mais rapida so ajuda de verdade se tambem oferecer collectives de
baixa latencia, como RDMA/InfiniBand, NVLink ou memoria compartilhada local.

Thunderbolt transporta PCIe, USB e possivelmente rede no mesmo link, portanto
`5`, `10` e `15 GB/s` sao limites brutos, nao throughput garantido para a
aplicacao. Bandwidth Boost tambem e assimetrico e nao representa `15 GB/s` em
cada sentido ao mesmo tempo. Em Linux, uma ligacao entre maquinas ainda passa
pelo driver e protocolo de rede Thunderbolt; ela nao vira memoria compartilhada.

Para visualizar o teto, o collective de 6000 tokens envia `21.504 MB` BF16
agregados. A serializacao ideal seria aproximadamente `276 ms` no enlace atual,
`4.30 ms` em Thunderbolt 4 e `2.15 ms` em Thunderbolt 5 bidirecional. Para um
token, entretanto, a serializacao atual ja e apenas `0.046 ms` diante dos `3.619
ms` medidos. Thunderbolt pode ajudar muito batches grandes, mas nao garante
speedup de decode unitario sem tambem reduzir a latencia do collective.

As especificacoes de referencia sao [NVIDIA Quantum-2, ate 400 Gbit/s por
porta](https://www.nvidia.com/en-us/networking/quantum2/) e [NVIDIA H100, NVLink
de 900 GB/s](https://www.nvidia.com/en-us/data-center/h100/), alem das paginas da
Intel para [Thunderbolt 4](https://www.intel.com/content/www/us/en/architecture-and-technology/thunderbolt/thunderbolt-4-vs-usb-c.html)
e [Thunderbolt 5](https://www.intel.com/content/www/us/en/architecture-and-technology/thunderbolt/thunderbolt-5.html).
Memoria DDR5 varia com canais, frequencia, CPU e plataforma; a faixa da tabela e
apenas um exemplo.

## Resultado Com 10 Workers Por Maquina

O sweep com `--worker-threads 10` melhorou compute e manteve ganho de 256 ate
6000 tokens:

| Tokens simultaneos | Baseline | Distribuido | Speedup |
|---:|---:|---:|---:|
| 1 | 1.431 ms | 4.512 ms | 0.317x |
| 128 | 9.812 ms | 13.212 ms | 0.743x |
| 256 | 29.383 ms | 18.611 ms | 1.579x |
| 1024 | 100.737 ms | 61.350 ms | 1.642x |
| 4096 | 489.192 ms | 291.527 ms | 1.678x |
| 6000 | 725.089 ms | 440.487 ms | 1.646x |

Esse speedup compara 14 threads no baseline local com 10 threads em cada rank,
20 threads distribuidas no total. Ele representa a capacidade adicional das duas
maquinas, nao uma comparacao com a mesma quantidade total de threads. O resultado
de quatro tokens (`1.951x`) e de 16 tokens (`2.152x`) nao deve ser tratado como
threshold: os baselines de `10.071 ms` e `12.340 ms` foram anormalmente lentos e
os tamanhos vizinhos voltaram a perder. O ganho sustentado desse sweep comeca em
256 tokens.

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
