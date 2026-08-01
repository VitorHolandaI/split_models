# Split Entre Dispositivos Desiguais

Split reparte **capacidade**, nao velocidade. Somar uma maquina lenta a uma
rapida deixa o conjunto mais lento que a rapida sozinha, e por uma margem
grande.

Medido em 2026-08-01 com hardware real, nao estimado.

## As Duas Maquinas

| | dispositivo | memoria livre |
|---|---|---|
| notebook | Intel Graphics (ARL), iGPU | ~7-10 GB, tirados da RAM do sistema |
| PC remoto | NVIDIA RTX 3060 | 11.4 GB de VRAM |

Ethernet entre elas: 65 MB/s medidos por `scp`, cerca de 520 Mbps.

Backends diferentes de proposito: Vulkan no notebook, CUDA no PC. O RPC do
llama.cpp transporta tensores, nao instrucoes, entao a combinacao funciona
desde que o commit seja o mesmo nas duas pontas.

## O Resultado

`unsloth/Qwen3.5-9B-GGUF:Q6_K`, 7.46 GB, mesmo prompt e mesma seed:

| configuracao | geracao | |
|---|---|---|
| notebook sozinho | 1.6 tok/s | modelo nao cabe no iGPU, transborda para CPU |
| notebook + 3060 em split | 6.8 tok/s | 4.2x melhor que o notebook |
| **3060 sozinha** | **42.6 tok/s** | **6.3x melhor que o split** |

O split ganhou do notebook e perdeu feio da 3060.

## Por Que O Split Perde

`--split-mode layer` da um intervalo contiguo de layers a cada dispositivo, e
**todo token atravessa todas as layers**. E um pipeline: a peca mais lenta
manda no conjunto inteiro.

O iGPU sozinho faz 1.6 tok/s contra 42.6 da 3060, cerca de 26x mais lento. Em
serie, a 3060 passa a maior parte do tempo esperando o Intel terminar a parte
dele. Os dispositivos nao somam vazao; o mais lento arrasta o mais rapido.

A rede nao e a causa principal aqui. Numa medicao anterior com o
`Qwen2.5-0.5B` inteiro na mesma maquina, o custo de um salto RPC em loopback
foi de 52.7 para 47.6 tok/s, cerca de 10%. A queda de 42.6 para 6.8 e outra
ordem de grandeza, e vem do dispositivo lento, nao do fio.

## Quando O Split Vale

Quando o modelo **nao cabe em maquina nenhuma**. Ai a comparacao nao e
"6.8 contra 42.6", e "6.8 contra nao rodar".

O caso oposto ja tinha aparecido antes por outro caminho: um `Qwen2.5-3B` em
bfloat16 rodando a 0.5 tok/s numa maquina onde os 6.17 GB de peso nao cabiam
na RAM. Isso da 3.1 GB/s efetivos, velocidade de disco, nao de DRAM -- a
maquina estava paginando. Vale o mesmo principio: o ganho do split vem de
tirar o modelo de um armazenamento lento, nao de somar processadores.

## Regra Pratica

1. Cabe na maquina mais rapida? Use so ela.
2. Nao cabe em nenhuma? Divida, e aceite a velocidade do elo mais lento.
3. As maquinas sao parecidas? Dividir custa pouco, porque nao ha elo fraco.

O caso 3 e o alvo original do projeto: varias maquinas CPU semelhantes numa
LAN. O caso 1 e a armadilha, porque a frota parece maior e entrega menos.

## O Que Falta Medir

`--tensor-split` permite dar proporcoes diferentes por dispositivo. Dando quase
tudo a 3060 e o minimo ao Intel, o gargalo deveria encolher. Nao chega aos 42.6
-- qualquer layer no dispositivo lento custa em todo token -- mas mede quanto
do prejuizo e evitavel.

Ate aqui o projeto nao define `--tensor-split` automaticamente, e por bom
motivo: as GPUs alvo compartilham memoria com o host, entao uma razao fixa so
valeria na maquina onde foi medida. Uma razao derivada de velocidade medida,
e nao de memoria livre, seria outra conversa.
