# Deployment E Memoria

Restricao alvo: cada host pode ter apenas 8 GB de RAM. O sistema PXE/live e o
processo worker precisam ser planejados juntos para ficar abaixo disso com folga.

## Imagem Docker Nao E Toda RAM

A imagem Docker fica em disco. O Linux carrega paginas em RAM sob demanda e usa
page cache, que pode ser liberado quando ha pressao de memoria.

O que realmente compete com os 8 GB:

- processo Python;
- bibliotecas carregadas (`torch`, `transformers`, kernels);
- pesos do modelo carregados como tensors;
- buffers temporarios de forward;
- hidden states recebidos/enviados;
- KV cache, quando existir;
- cache de arquivos do sistema, que o kernel pode descartar.

Portanto, imagem grande e ruim para distribuicao/disco, mas o risco fatal e o
RSS do processo durante inferencia.

## PXE / Live ISO

Para uma ISO Linux bootada por PXE, a imagem pode sim competir diretamente com
RAM se o boot usar `toram`, rootfs em `tmpfs`, initramfs grande, ou copiar o
squashfs inteiro para memoria.

Nesse modo, o orcamento vira:

```text
8 GB RAM total
- kernel + initramfs
- rootfs/ISO em RAM, se usar toram/tmpfs
- processo Python worker
- bibliotecas carregadas
- pesos do modelo
- buffers/hidden/KV cache
= margem livre
```

Para PXE com 8 GB, a regra pratica passa a ser:

```text
rootfs em RAM + worker RSS <= 6 GB
```

Se a ISO ficar montada via rede/NFS/iSCSI/HTTP e nao for copiada inteira para
RAM, a imagem pesa mais em rede/disco do que em RAM, e o limite principal volta
a ser o RSS do worker.

Para esse PoC, nao embutir pesos do modelo na ISO. Montar cache/pesos por rede
ou disco local persistente, senao a ISO cresce e rouba RAM no modo `toram`.

## Base Linux

Alpine usa musl libc. PyTorch oficial em Linux espera glibc. Por isso Alpine e
arriscado para este PoC: pode forcar build from source ou quebrar wheels.

Base recomendada para o PoC:

```text
python:3.12-slim-bookworm
```

Isso tende a ser maior que Alpine em disco, mas menor em risco e tempo.

## Dependencias Atuais

`.venv` local medido: 4.7 GB.

Principais blocos:

```text
2.7G  nvidia/*
1.1G  torch
690M  triton
55M   transformers
```

Para CPU-only, a imagem deve instalar torch CPU separadamente para evitar
carregar o stack NVIDIA no pacote:

```bash
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir -r requirements-cpu.txt
```

Nao usar `requirements.txt` generico no build CPU-only se ele puxar o wheel
padrao com CUDA/NVIDIA/Triton. CPU-only e CPU-only: sem `nvidia/*`, sem CUDA,
sem Triton quando nao for exigido pelo backend.

Venv CPU-only medida em `/tmp/opencode/split-cpu-venv`: 893 MB.

Principais blocos CPU-only:

```text
692M  torch
51M   transformers
30M   numpy
29M   sympy
27M   numpy.libs
12M   hf_xet
9.9M  tokenizers
```

## Orcamento Por Host

Com 8 GB RAM, o alvo deve ser no maximo 6 GB de RSS por worker, deixando folga
para kernel, Docker, rede e picos temporarios.

Regra pratica:

```text
worker_rss <= 6 GB
```

Se o worker passar disso, diminuir o numero de layers por host ou usar backend
quantizado/mais leve.

## Implicacao Para Modelos

Qwen2.5-0.5B cabe folgado em 2 ou mais workers.

Modelos 9B em BF16/FP16 podem passar facilmente do limite por host se cada
worker carregar muitas layers. Split vertical ajuda, mas precisa medir RSS real.

GPT-OSS 20B e MoE quantizado podem ser viaveis em memoria ativa menor, mas o
backend precisa respeitar MXFP4/MoE. Nao assumir que carregar como tensor comum
vai caber ou funcionar.

## Medicao Obrigatoria

Cada teste de modelo novo deve registrar:

- modelo e backend;
- layer range;
- dtype/quantizacao;
- RSS apos carregar pesos;
- pico de RSS durante 1 token;
- pico durante N tokens com KV cache, quando ativado.

Sem essa medicao, nao considerar o modelo suportado em hosts de 8 GB.

## Medicao CPU-Only Validada

Ambiente:

- venv CPU-only em `/tmp/opencode/split-cpu-venv`;
- `torch==2.13.0+cpu`;
- `torch.version.cuda == None`;
- modelo `Qwen/Qwen2.5-0.5B`;
- 2 processos locais simulando 2 terminais: head + 1 worker;
- worker com todas as 24 layers (`0..23`);
- `max_new_tokens=1`.

Resultado:

```text
stdout do head:  a
worker token: 264
worker peak RSS: 2141.5 MB
```

Precisao efetiva:

```text
nao quantizado
runtime efetivo: FP32
494M parametros * 4 bytes ~= 1.98 GB de pesos
```

Mesmo que o checkpoint venha em BF16, o modelo criado por `from_config()` usa
parametros FP32 por padrao, e `load_state_dict()` copia/casta os pesos para o
dtype dos parametros. O pico medido confirma FP32.

Conclusao: Qwen2.5-0.5B CPU-only cabe com folga em host de 8 GB mesmo com um
worker carregando todas as layers. Modelos maiores ainda precisam medicao por
range de layers.
