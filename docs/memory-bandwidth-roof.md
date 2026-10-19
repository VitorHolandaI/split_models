# Teto De Largura De Banda

Decode autoregressivo em CPU nao e limitado por FLOPs. E limitado por quantos
bytes de peso a maquina consegue puxar da DRAM por segundo. Este documento fixa
o denominador de qualquer analise de speedup do projeto: sem ele, nao da para
saber se um ganho veio de otimizacao real ou de estar longe do limite fisico.

Medicoes abaixo feitas em 2026-07-31, host de desenvolvimento
(Intel Core Ultra 5 225H, 14 cores, 1 thread por core, L3 de 18 MiB).

## Cada Token Rele Todos Os Pesos

Para produzir o token N+1 o modelo faz um forward pass completo por todas as
camadas. Cada matriz de peso e usada exatamente uma vez nesse pass. Terminou o
token, comeca o proximo: mesmo passeio, mesmas matrizes.

Nao da para cachear. O L3 tem 18 MiB e os pesos do `Qwen2.5-3B-Instruct` em
bfloat16 tem 6.17 GB. Cabe 0.3%. Praticamente todo byte vem da DRAM, toda vez.

Tamanho dos pesos confirmado em disco:

```
du -Lscb ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/*/*.safetensors
6171926992      total
```

O `-L` importa: o cache do HuggingFace guarda symlinks para `blobs/`, e sem
seguir o link o `du` soma o tamanho do symlink, nao do arquivo.

O KV cache entra por cima mas e ruido nesta escala: 36 camadas x 2 KV heads x
128 dim x 2 (K e V) x 2 bytes = ~37 KB por token de contexto. Em 2048 tokens sao
76 MB, contra 6170 MB de pesos.

## Como Medir A Banda Corretamente

STREAM do McCalpin, seguindo as regras descritas por Georg Hager em
<https://blogs.fau.de/hager/archives/8263>. Fonte usada:
<https://github.com/intel/memory-bandwidth-benchmarks/blob/master/stream.c>.

```sh
gcc -O3 -march=native -fopenmp -ffreestanding -mcmodel=medium \
    -DSTREAM_ARRAY_SIZE=200000000 -DNTIMES=20 stream.c -o stream_mccalpin
OMP_NUM_THREADS=14 OMP_PROC_BIND=close OMP_PLACES=cores ./stream_mccalpin
```

Quatro detalhes que mudam o resultado:

- **Array >= 4x a soma dos last-level caches.** Aqui 1.5 GiB por array contra
  18 MiB de L3. Abaixo disso mede-se cache, nao DRAM.
- **`-mcmodel=medium`.** Sem isso nem linka: os arrays estaticos de 4.5 GiB
  estouram o modelo small do x86-64 com
  `relocation truncated to fit: R_X86_64_PC32 against '.bss'`.
- **Pinning (`OMP_PROC_BIND`/`OMP_PLACES`).** Sem pinning a variacao entre
  repeticoes chegou a 28% neste host.
- **`-ffreestanding`.** Impede o compilador de substituir o loop por `memcpy()`,
  o que mediria a glibc em vez do kernel escrito.

## Correcao Write-Allocate

STREAM conta 16 bytes por iteracao em Copy/Scale e 24 em Add/Triad. Quando ha
write-allocate, cada store obriga uma leitura previa da linha de cache, e o
trafego real vira 24 e 32. O resultado precisa ser corrigido por 1.5x e 1.33x
respectivamente.

Sem essa correcao os kernels discordam entre si e nao ha como saber qual esta
certo. Com ela, convergem:

| kernel | reportado | corrigido |
|--------|-----------|-----------|
| Copy   | 21.5 GB/s | 32.2 GB/s |
| Scale  | 21.6 GB/s | 32.4 GB/s |
| Add    | 24.4 GB/s | 32.5 GB/s |
| Triad  | 24.2 GB/s | 32.2 GB/s |

Convergencia dos quatro em ~32 GB/s e o sinal de que a medicao esta sa.

Dois metodos independentes confirmam: um kernel de copy escrito a mao com
read-for-ownership deu 20.6 GB/s brutos, que corrigidos por 1.5x viram 30.9. E
`np.copyto` do numpy deu 31.6 GB/s sem correcao, porque `memcpy` usa stores
nao-temporais que pulam o read-for-ownership.

## Leitura Pura E O Numero Que Importa

STREAM nao tem kernel de leitura pura, e e esse o regime do decode. Kernel
proprio em `tools/read_bw.c`, mesma disciplina (1.5 GiB, NTIMES=20, first-touch
paralelo, threads pinadas). Leitura pura nao sofre write-allocate, entao nao
leva fator de correcao.

```
READ best  : 37.1 GB/s
READ avg   : 33.3 GB/s
READ worst : 29.1 GB/s
```

## Teto Resultante

```
pesos     6.17 GB por token
banda     37.1 GB/s (best)  ->  6.0 tok/s
          33.3 GB/s (avg)   ->  5.4 tok/s
```

Nenhuma otimizacao de codigo passa disso numa maquina so. Passar do teto e a
unica prova de que uma configuracao distribuida esta comprando banda de memoria
de verdade, e nao apenas sobrepondo trabalho.

## Comparando Com O Benchmark

`split_lm/metrics.py:41-57` produz duas taxas, e elas medem coisas diferentes:

| metrica | formula | valor | regime |
|---------|---------|-------|--------|
| `tokens_per_second` | `generated / total` | 3.37 tok/s | end-to-end, **inclui prefill** |
| `1 / average_inter_token_seconds` | `1 / 0.27774` | 3.60 tok/s | decode puro, **exclui prefill** |

Dados de `benchmark-results/poem-realistic-local.csv`, Qwen2.5-3B bfloat16,
prompt de 44 tokens, 128 tokens gerados.

Contra o teto de banda usa-se **sempre a segunda**, porque o teto descreve
decode e prefill e outro regime:

```
3.60 tok/s x 6.17 GB = 22.2 GB/s  ->  60% de 37.1 (best) | 67% de 33.3 (avg)
```

Ou seja: o decode local ja opera na faixa de 60-67% do limite fisico do host.

Verificacao de que a decomposicao fecha, com os campos do CSV:

```
ttft + (generated - 1) * inter_token = 2.708 + 127 * 0.27774 = 37.98181 s
total_seconds registrado                                     = 37.98197 s
```

Erro de 0.16 ms em 38 s.

## Decode E Plano No Contexto, Prefill Nao

De `benchmark-results/context-realistic-local.csv`:

| prompt | ttft (s) | e2e tok/s | decode tok/s |
|--------|----------|-----------|--------------|
| 128    | 5.38     | 3.12      | 3.56 |
| 256    | 4.28     | 3.44      | 3.86 |
| 512    | 6.64     | 3.13      | 3.70 |
| 1024   | 11.03    | 2.73      | 3.54 |
| 2048   | 21.35    | 0.86      | 1.00 |

Decode puro fica plano em ~3.5-3.9 tok/s de 128 a 1024, exatamente como o modelo
de banda preve: os pesos nao mudam de tamanho com o contexto. O end-to-end cai
no mesmo intervalo porque o TTFT dobra e dilui a media.

Consequencia pratica: comparar `tokens_per_second` entre tamanhos de contexto
mede majoritariamente prefill.

**Anomalia em aberto:** a 2048 o decode desaba para 1.00 tok/s, 3.5x abaixo do
patamar. Banda de memoria nao explica, os pesos continuam 6.17 GB. Candidatos
nao investigados, do mais barato ao mais caro de testar: swap (15 GB de RAM com
5 ja em uso), KV cache nao reutilizado, atencao O(n^2). O KV cache em si esta
descartado pelos 76 MB calculados acima.

## Reproduzir

```sh
gcc -O3 -march=native -fopenmp -mcmodel=medium -o read_bw tools/read_bw.c
OMP_NUM_THREADS=14 OMP_PROC_BIND=close OMP_PLACES=cores ./read_bw
```

Ajustar `OMP_NUM_THREADS` para o numero de cores fisicos do host, e conferir o
L3 com `lscpu -C` antes de confiar no tamanho de array.
