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

## A Evidencia Mais Limpa: 11% De Ocupacao

Durante o split, `nvidia-smi` mostrou a RTX 3060 em **11% de utilizacao**. Esse
numero mede a causa, nao o efeito.

Os dispositivos trabalham em sequencia, nunca em paralelo:

```
Intel calcula as layers dele  ->  manda ativacoes pela rede
                                  ->  3060 calcula as layers dela
                                  ->  resultado volta, proximo token
```

Enquanto um calcula, o outro espera. A dependencia e estrita: a layer 20
precisa da saida da 19, entao nao ha como sobrepor trabalho dentro de um mesmo
token. Os 11% sao a fracao do tempo em que a 3060 de fato calcula.

Com maquinas iguais cada uma ficaria perto de 50% ocupada -- ainda desperdicio,
mas simetrico. A 3060 sendo ~26x mais rapida, ela despacha a parte dela e passa
o resto do tempo esperando; 11% e aproximadamente a razao de velocidade entre
as duas.

Isso nao e configuracao errada nem falta de ajuste. E o que dividir um pipeline
sequencial entre dispositivos desiguais produz.

## Por Que Batching Resolveria E Aqui Nao Resolve

Pipeline so atinge ocupacao alta com varias requisicoes em voo: enquanto a
3060 processa o token do usuario A, o Intel ja comeca o do usuario B. E assim
que servidor de inferencia mantem GPU ocupada.

`docs/batching.md` mostra por que nao serve aqui: com batch=1 nao existe
sobreposicao possivel, porque so ha uma cadeia de dependencia em andamento. O
`split_models` e single-user por construcao, entao a ociosidade do pipeline nao
tem de onde ser preenchida.

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

## A Ordem De --tensor-split Nao E A Do --list-devices

Armadilha que custou varias medicoes. `--list-devices` imprime:

```
Vulkan0: Intel(R) Graphics (ARL)
RPC0:    192.168.0.11:50052
```

Mas `src/llama.cpp:264` faz:

```cpp
model->devices.insert(model->devices.begin(), rpc_servers.begin(), rpc_servers.end());
```

Os workers RPC entram no **inicio** do vetor. A ordem que `--tensor-split` usa
e portanto a inversa da impressa:

```
indice 0 = RPC0    (a maquina remota)
indice 1 = Vulkan0 (o dispositivo local)
```

Entao `--tensor-split 10,90` da 10% ao remoto e 90% ao local -- o oposto do que
a saida do `--list-devices` sugere. Confirme sempre no log de
`llama_prepare_model_devices`, que lista na ordem real.

## tensor-split Nao Resolveu

A ideia obvia e dar quase tudo ao dispositivo rapido. Nenhuma tentativa com o
27B ficou melhor que a 3060 sozinha:

Com o `Qwen3.5-9B` em Q6_K, `-c 4096`, RPC no indice 0 e iGPU no indice 1:

| razao | carga real | geracao |
|---|---|---|
| `10,90` | 90% no iGPU | 5.1 tok/s |
| `--fit on` (sem razao fixa) | ~44% no iGPU | 6.8 - 8.4 tok/s |
| **`90,10`** | **90% na 3060** | **14.7 tok/s** |
| sem dividir | tudo na 3060 | **42.6 tok/s** |

Acertar a ordem quase triplicou o split, de 5.1 para 14.7 tok/s. A penalidade
contra usar so a maquina rapida caiu de 6.3x para 2.9x.

E continua penalidade. Mesmo com 90% do trabalho na 3060, os 10% restantes no
iGPU custam em cada token, e a rede entra na conta duas vezes por token. O
melhor split medido perde por quase tres vezes de simplesmente nao dividir.

Com o 27B, onde a proporcao importava mais, as tentativas foram feitas com a
ordem invertida e portanto sobrecarregando o iGPU:

| razao | carga real | resultado |
|---|---|---|
| `20,80` | 80% no iGPU | falhou ao criar contexto |
| `30,70` | 70% no iGPU | nao terminou em 10 minutos |

Falta refazer essas com a ordem correta.

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
