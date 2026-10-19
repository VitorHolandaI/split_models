# Batching E O Ponto De Virada

Dez segundos para servir uma pessoa custam cerca de dez segundos para servir
vinte. Ate um teto, que este documento calcula.

Complementa `docs/memory-bandwidth-roof.md`, que estabelece por que decode em
CPU e limitado por memoria e nao por FLOPs.

## Batch E Horizontal, Nao Vertical

Dentro de uma conversa a geracao e estritamente sequencial: o token N+1 depende
do token N ter sido escolhido. Isso nao e paralelizavel e nunca sera.

Batch nao ataca isso. Batch e atender **varias conversas independentes ao mesmo
tempo**:

```
batch=1    ->  le 6.17 GB  ->  1 token   (um usuario)
batch=32   ->  le 6.17 GB  ->  32 tokens (32 usuarios, 1 token cada)
```

Mesmo trafego de DRAM, 32x a vazao total.

Funciona porque a matriz de pesos e a mesma para todos os usuarios. Com batch=1
a operacao e vetor x matriz: carrega a matriz inteira, usa cada peso **uma vez**,
descarta. Com batch=32 vira matriz x matriz, e cada peso e usado **32 vezes**
antes de ser descartado. A leitura foi amortizada.

O que batch **nao** faz: o usuario individual nao recebe a resposta mais rapido.
A latencia dele fica praticamente igual. Cresce apenas a soma do que a maquina
produz.

E por isso que servidor de inferencia e viavel e um laptop e lento com o mesmo
modelo: eles rodam batch de centenas, o laptop roda batch 1. **Batch 1 e o pior
caso possivel do hardware**, e e o caso do `split_models`.

## Onde Para De Ser De Graca

Com batch=1 a maquina passa quase todo o tempo esperando DRAM e quase nenhum
calculando. Ha CPU ociosa de sobra, e adicionar usuarios preenche essa ociosidade
sem custo. Quando a CPU deixa de estar ociosa, o custo volta a ser linear: 40
usuarios passam a custar o dobro de 20.

O ponto de virada e o *ridge point* do modelo Roofline:

```
B* = FLOP/s da maquina / GB/s da maquina
```

A forma fica limpa assim porque em bfloat16 cada peso ocupa 2 bytes e produz 2
FLOPs por item do batch (uma multiplicacao e uma soma). Os fatores 2 se cancelam,
e o ponto de virada vira o *machine balance* puro.

## Medindo Os Dois Termos

Banda: 33.3 GB/s de leitura sustentada, conforme
`docs/memory-bandwidth-roof.md`.

FLOPs: medido com o `torch` que o projeto realmente usa, nao com numpy.

```sh
python3 -c "
import torch, time
torch.set_num_threads(14)
for dt in (torch.float32, torch.bfloat16):
    for n in (2048, 4096):
        a = torch.randn(n, n, dtype=dt)
        b = torch.randn(n, n, dtype=dt)
        a @ b                                   # descarta a primeira, aquece
        best = 0.0
        for _ in range(3):
            start = time.perf_counter()
            a @ b
            elapsed = time.perf_counter() - start
            best = max(best, 2 * n**3 / elapsed / 1e9)
        print(f'{dt} {n}x{n}: {best:.1f} GFLOP/s')
"
```

Resultado no host de desenvolvimento (torch 2.13.0, 14 threads):

```
float32   2048x2048:   676.7 GFLOP/s
float32   4096x4096:   207.1 GFLOP/s
bfloat16  2048x2048:   429.1 GFLOP/s
bfloat16  4096x4096:   425.6 GFLOP/s
```

Duas observacoes obrigatorias antes de usar esses numeros:

- **Nao medir com numpy neste host.** `np.__config__` reporta `cblas`, a
  referencia netlib, que e um loop triplo ingenuo: deu 3 e 7 GFLOP/s. Nao
  descreve a maquina nem o stack de inferencia, que usa oneDNN via torch.
- **A queda de fp32 em 4096 (676 -> 207) esta em aberto.** bfloat16 ficou estavel
  nos dois tamanhos, entao o numero usado abaixo e o de bfloat16, que e tambem a
  precisao dos benchmarks do projeto.

## Ponto De Virada Neste Host

```
B* = 425.6 GFLOP/s / 33.3 GB/s = 13
```

Ate cerca de **13 conversas simultaneas**, batching e praticamente gratuito nesta
maquina. Acima disso o regime vira compute-bound e cada conversa nova custa tempo
proporcional.

Efeito no `Qwen2.5-3B-Instruct` em bfloat16:

| batch | tokens/s agregado | por usuario |
|-------|-------------------|-------------|
| 1     | 5.4               | 5.4 |
| 13    | 70.2              | 5.4 |
| 26    | ~70               | ~2.7 |

Ate 13, a vazao cresce e a experiencia individual nao piora. Depois de 13, a
vazao trava e o que piora e a latencia de cada um.

O valor de B* e propriedade da **maquina**, nao do modelo: os fatores do modelo
se cancelam na derivacao. Um host com GPU tem B* muito maior, na casa das
centenas, porque a razao FLOP/byte do hardware e outra. Por isso servidores de
GPU rodam batch grande e ainda ficam memory-bound.

## Custo Que Batch Nao Amortiza

Pesos sao compartilhados entre as conversas do batch. KV cache **nao e**: cada
conversa tem o seu, e a memoria cresce linear com o batch.

Para o Qwen2.5-3B, ~37 KB por token de contexto (36 camadas x 2 KV heads x 128
dim x 2 tensores x 2 bytes). Em 2048 tokens de contexto:

```
batch=1   ->   76 MB
batch=13  ->  988 MB
batch=32  ->  2.4 GB
```

Num host de 8 GB, como assume `docs/deployment-memory.md`, isso compete
diretamente com os pesos e pode virar o limite antes do ridge point.

## Implicacao Para O Projeto

O `split_models` e single-user, batch=1. Toda a analise de banda em
`docs/memory-bandwidth-roof.md` vale integralmente, e nenhuma das
otimizacoes de batching se aplica.

A consequencia inversa tambem vale, e e o que torna o projeto viavel: como o
decode em CPU e lento, sobra orcamento de tempo para a rede. Um custo de rede de
dezenas de milissegundos por token e pequeno contra os ~185 ms de leitura de
pesos. Numa GPU, onde o token custa poucos milissegundos, o mesmo custo de rede
seria fatal.

**A lentidao da CPU e o que abre espaco para Ethernet barata funcionar.**

## Custo Fixo Por Token E Modelos Pequenos

Observado em execucao split com `Qwen2.5-0.5B-Instruct` (bfloat16, 2026-07-31):

| prompt | gerados | `tokens_per_second` | decode real (`1/inter`) |
|--------|---------|---------------------|-------------------------|
| 73     | 12      | 6.41                | 12.97 |
| 97     | 12      | 5.53                | 12.80 |
| 124    | 252     | 11.34               | 12.21 |

`tokens_per_second` varia 2x entre os runs, enquanto `average_inter_token` fica
em 77.1 / 78.1 / 81.9 ms. O motor esta estavel; o que muda e quanto o TTFT pesa
na media. Com 12 tokens gerados o end-to-end captura 43-49% da taxa real de
decode; com 252 tokens, 93%.

Pesos do 0.5B: 988.097.824 bytes. A 12.5 tok/s isso da 12.3 GB/s efetivos, bem
abaixo dos 33.3 GB/s do host de desenvolvimento. A diferenca nao e banda pior: e
custo fixo por token. Se a maquina tivesse banda semelhante, os pesos levariam
~30 ms, contra os 78 ms medidos. Sobram ~48 ms de rede, Python e framing.

Esse custo fixo **nao encolhe com o modelo**. Num 0.5B ele e a maior parte do
orcamento por token; num 3B seria uma fracao menor. Modelo pequeno em modo split
e o pior caso de eficiencia: pouca leitura de peso para amortizar a rede.

Corolario para escolha de modelo no projeto: split compensa mais quanto **maior**
o modelo, ate o limite da RAM por host.
