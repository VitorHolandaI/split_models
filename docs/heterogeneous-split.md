# Split Entre Dispositivos Desiguais

Split reparte **capacidade**, nao velocidade. Somar uma maquina lenta a uma
rapida deixa o conjunto mais lento que a rapida sozinha, e por uma margem
grande.

Medido em 2026-08-01 com hardware real, nao estimado.

## As Duas Maquinas

| | dispositivo | memoria do dispositivo | RAM do host |
|---|---|---|---|
| notebook | Intel Graphics (ARL), iGPU | sem memoria propria | 16 GB, compartilhados com o iGPU |
| PC remoto | NVIDIA RTX 3060 | 12 GB de VRAM, 11.4 livres | 16 GB |

O iGPU nao tem memoria propria: o que ele reporta como livre e RAM do sistema,
disputada com o proprio sistema operacional. A 3060 tem 12 GB dedicados, e a
maquina dela ainda tem 16 GB de RAM alem disso.

Ethernet entre elas: 65 MB/s medidos por `scp`, cerca de 520 Mbps.

Backends diferentes de proposito: Vulkan no notebook, CUDA no PC. O RPC do
llama.cpp transporta tensores, nao instrucoes, entao a combinacao funciona
desde que o commit seja o mesmo nas duas pontas.

## O Resultado

Mesmo prompt e mesma seed nas tres configuracoes:

| modelo | notebook so | split | 3060 so |
|---|---|---|---|
| Qwen3.5-9B Q6_K (7.46 GB) | 1.6 tok/s | 6.8 tok/s | **42.6 tok/s** |
| Qwen3.6-27B Q3_K_M (13.59 GB) | nao roda | 3.1 tok/s | **7.6 tok/s** |

**O split perdeu em todos os casos em que a maquina rapida conseguia rodar.**

"Nao roda" e literal: o notebook passou 16 minutos com o 27B, 6 GB em swap e
11% de CPU, sem produzir um token. Ao matar o processo a memoria disponivel
voltou de 3 GB para 11 GB no mesmo instante -- nao estava calculando, estava
esperando disco.

## O Caso Favoravel Tambem Perdeu

O 27B era o teste desenhado para o split ganhar: 13.59 GB nao cabem nos 11.4 GB
de VRAM da 3060. Mesmo assim ela sozinha fez 7.6 tok/s contra 3.1 do split.

O motivo desmonta a premissa: a maquina da 3060 tem 16 GB de RAM, e `--fit on`
descarregou o excedente na CPU **local** dela. Descarregar para a propria CPU e
mais rapido que mandar pela rede para um iGPU 26x mais lento.

Ou seja, "o modelo nao cabe na GPU" nao basta para justificar o split. Precisa
nao caber **na maquina inteira**, contando a RAM do host.

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

## tensor-split Nao Resolveu

A ideia obvia e dar quase tudo ao dispositivo rapido. Duas tentativas com o
27B, nenhuma melhor que a 3060 sozinha:

| razao | resultado |
|---|---|
| `--tensor-split 20,80` | falhou ao criar contexto: 80% de 13.59 GB mais o KV cache passa dos 11.4 GB da 3060 |
| `--tensor-split 30,70` | nao terminou em 10 minutos |

O espaco util e estreito por construcao. Dar pouco ao dispositivo lento nao
adianta, porque toda layer nele custa em **todo** token; dar muito ao rapido
estoura a memoria dele, que era a razao de dividir. Entre os dois extremos nao
sobrou nenhuma razao que ganhasse de simplesmente nao dividir.

`--fit on` tambem escolhe mal: ele reparte por **memoria livre**, viu 8.9 GB no
iGPU contra 11.4 GB na 3060 e deu quase metade das layers ao dispositivo 26x
mais lento. E a metrica errada -- e a memoria "livre" do iGPU e RAM do sistema,
que foi direto para swap.

Uma razao derivada de velocidade medida, e nao de memoria, poderia fazer melhor.
Mas o teto continua sendo a 3060 sozinha, entao o ganho maximo dessa otimizacao
e chegar perto de nao ter dividido.

## Conclusao

Split entrega **capacidade**, nunca velocidade. A condicao para valer a pena e
mais estreita do que parecia no inicio do projeto:

- nao basta o modelo nao caber na GPU;
- precisa nao caber na maquina inteira, host RAM incluida;
- e os dispositivos precisam ter velocidade parecida, senao o mais lento paga
  o preco em cada token.

Frota homogenea de maquinas CPU numa LAN -- o alvo original -- satisfaz a
terceira condicao por construcao. Misturar um iGPU com uma GPU dedicada, nao.
