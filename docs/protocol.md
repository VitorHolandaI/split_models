# Tensor Protocol

Os canais de dados usam TCP persistente com `TCP_NODELAY`. Cada tensor usa um
frame binario, sem Pickle, `torch.save` ou `torch.load`:

```text
magic          2 bytes  "SL"
version        1 byte
flags          1 byte   bit 0 reinicia o KV cache
dtype          1 byte
dimensions     1 byte
payload size   4 bytes  big-endian
shape          4 bytes por dimensao, big-endian
tensor         bytes contiguos no dtype original
```

Um token `int64` com shape `[1]` ocupa 22 bytes: 10 de cabecalho, 4 de shape e
8 de tensor. As estatisticas de profiling medem esses frames da aplicacao. Elas
nao incluem headers TCP/IP nem representam os pacotes fisicos produzidos pelo
kernel.

Mensagens de controle (`READY`, `ERROR`, `STOP`) continuam usando JSON em um
socket separado porque nao participam do caminho critico por token.
