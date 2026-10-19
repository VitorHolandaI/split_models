# MoE: Offload De Experts Bate A Frota

Para modelos Mixture of Experts grandes demais para a VRAM, tirar **parte** dos
experts para a RAM da propria maquina entrega 5x mais que dividir o modelo entre
duas maquinas.

Medido em 2026-08-02 com `unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ3_S`, 13.67 GB,
34.66 B de parametros com ~3 B ativos por token.

## O Resultado

```
frota: notebook (iGPU Arc) + RTX 3060, split 50/50     9.40 tok/s
uma RTX 3060 sozinha, experts na RAM (-ncmoe 12)      48.77 tok/s
```

**5.2x a favor de uma maquina so.** Nenhuma proporcao de `--tensor-split` chegou
perto: a frota deu 8.55 em 90/10, 8.54 em 78/22 e 9.40 em 50/50.

## Como Funciona

Cada camada de um MoE tem duas partes: **atencao**, pequena e usada em todo
token, e **experts**, grandes e ativados poucos por vez.

`-ncmoe N` mantem na CPU os experts das **N primeiras camadas** (texto da ajuda:
"keep the Mixture of Experts (MoE) weights of the first N layers in the CPU").
`-ngl 99` continua mandando tudo para a GPU; o `-ncmoe` retira de la so os
experts dessas N camadas.

| | VRAM | RAM |
|---|---|---|
| atencao de todas as camadas | sim | — |
| experts das camadas 0..N-1 | — | sim |
| experts das camadas N.. | sim | — |

## O Menor N Ganha

RTX 3060 12 GB, Ryzen 5 5500X3D, `-t 10`:

| `-ncmoe` | prefill | decode |
|---|---|---|
| **12** | **328.8 ± 50.2** | **48.77 ± 3.80** |
| 18 | 223.9 ± 81.3 | 47.28 ± 0.92 |
| 24 | 197.0 ± 40.0 | 39.57 ± 2.38 |
| 30 | 81.5 ± 40.5 | 36.41 ± 4.23 |
| 36 | 152.3 ± 2.1 | 32.20 ± 0.92 |
| 99 (todos) | 49.6 | 30.33 |
| 0 (nenhum) | falha: 13.67 GB nao cabem em 11.4 |

Monotonico. Cada camada tirada da VRAM custa, porque a CPU passa a fazer aquele
matmul e faz pior. O otimo e o **menor N que ainda carrega**:

```
N baixo demais  →  nao cabe na VRAM, falha ao carregar
N otimo         →  cabe por pouco, a GPU faz o maximo
N alto demais   →  cabe folgado, a CPU trabalha a toa
```

## Mais Threads Pioram

Mesmo `-ncmoe 24`, mesma maquina:

```
2 threads (padrao do llama-bench)   44.62 tok/s
10 threads                          39.57 tok/s
```

Doze por cento pior com cinco vezes mais threads, acima do desvio de 2.38.

Coerente com o resto: o matmul dos experts e limitado por **banda de memoria**,
nao por calculo. Threads a mais so disputam a mesma DRAM e somam sincronizacao.
Vale varrer `-t` em vez de assumir "quanto mais, melhor".

## Consequencia Para O Projeto

Para MoE grande a recomendacao inverte:

> **Compre RAM, nao some maquinas.**

Uma RTX 3060 de 12 GB com 16 GB de RAM roda um 35B a 48.8 tok/s. Duas maquinas
em rede rodam o mesmo modelo a 9.40.

O espaco onde dividir entre maquinas ainda serve ficou estreito: o modelo precisa
nao caber **nem com offload de experts** na maior maquina disponivel. Ver
`docs/heterogeneous-split.md` para as outras condicoes.

## Armadilhas Ao Medir

**A lista do `llama-bench` aborta na primeira falha.** `-ncmoe 4,6,8,10,12`
morreu inteiro porque 4 nao cabe -- nenhum dos outros chegou a rodar. Para
procurar o minimo, um valor por invocacao:

```sh
for n in 4 6 8 10 12; do
    llama-bench -m "$M" -ngl 99 -ncmoe "$n" -t 10 -p 128 -n 32 -r 2 2>/dev/null | tail -2
done
```

Isso importa justamente aqui, porque o que se procura e a fronteira entre caber
e nao caber -- ou seja, um ponto cercado de falhas.



`llama-bench` usa **2 threads por padrao** (`-t, --threads <n> (default: 2)`),
enquanto `llama-cli` usa `-1` e detecta os cores. Qualquer medicao com carga na
CPU -- e `-ncmoe` e exatamente isso -- mede um pedaco da maquina se `-t` nao for
passado. Medicoes puramente em GPU nao sao afetadas.
