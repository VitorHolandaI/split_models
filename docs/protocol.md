# Protocolo Do Cluster Vertical

Este documento descreve o protocolo implementado pelo `split-cluster`: como os
nos aparecem na rede, como o master distribui as camadas, quais mensagens cada
maquina recebe e como os tensores circulam durante a geracao.

O projeto assume uma LAN confiavel. Nao ha TLS, autenticacao, eleicao de lider
ou consenso distribuido.

## Papeis

| Papel | Processo | Responsabilidade |
|---|---|---|
| Agente | `split-cluster worker` | Responder discovery e aceitar jobs validados |
| Master | `split-cluster master --discover` | Descobrir nos, criar plano, iniciar jobs e executar o head |
| Head | `split-lm head` | Tokenizar, executar as primeiras camadas e coordenar a geracao |
| Worker intermediario | `split-lm worker` | Executar um intervalo de camadas e encaminhar hidden states |
| Worker final | `split-lm worker` | Executar as ultimas camadas, `norm`, `lm_head` e escolher o token |

Todas as appliances iniciam como agentes. O operador escolhe uma delas,
interrompe o agente com `Ctrl-C` e executa `/root/split-master.sh`. Nao existe
promocao ou eleicao automatica de master.

## Portas

| Porta | Transporte | Origem -> destino | Uso |
|---|---|---|---|
| 29500 | UDP broadcast | master -> agentes | Discovery |
| 29510 | TCP | master -> agente | `START` e `STOP` de subprocessos |
| 9999 | TCP | head/worker anterior -> worker | Hidden states ou IDs de entrada |
| 9999 | TCP | worker final -> head | Token escolhido, em host diferente |
| 10001 | TCP | workers -> head | `READY`, `ERROR` e controle runtime |

O uso repetido de `9999` e valido porque cada listener esta em uma maquina
diferente. Os canais de dados usam conexoes persistentes com `TCP_NODELAY`.

## Estados Conceituais

Estes nomes explicam o ciclo de vida; eles nao sao inteiros serializados no
wire.

```text
AGENTE:
  AGENT_READY -> JOB_RUNNING -> AGENT_READY

MASTER:
  DISCOVER -> PLAN -> START_WORKERS -> WAIT_READY -> GENERATE -> STOP_WORKERS

WORKER DE INFERENCIA:
  SPAWNED -> LOAD_MODEL -> CONNECT -> READY -> FORWARD -> STOPPED | ERROR
```

## Inicio Passo A Passo

1. O systemd inicia uma sessao tmux com `split-cluster worker` em cada maquina.
2. Cada agente gera um `node_id` aleatorio, abre UDP/29500 e TCP/29510.
3. O operador para o agente somente na maquina escolhida como master.
4. O master envia duas probes UDP para o broadcast da LAN.
5. Os agentes respondem com identidade, CPU, memoria disponivel e porta de controle.
6. O master ordena os nos por `node_id` e divide as camadas em intervalos contiguos.
7. O master envia jobs `START` primeiro ao worker final e depois em direcao ao head.
8. Cada agente valida o job e cria um subprocesso `split-lm worker`, sem shell.
9. O head abre TCP/10001 para controle e TCP/9999 para o retorno de tokens.
10. Cada worker carrega sua fatia, conecta seus canais e envia `READY` ao head.
11. Depois de receber todos os `READY`, o head conecta ao primeiro worker.
12. A geracao comeca. Ao terminar, o master envia `STOP` aos agentes iniciados.

## Discovery UDP

### Probe

O payload e exatamente esta sequencia ASCII, sem JSON ou prefixo de tamanho:

```text
SPLIT_DISCOVER
```

Destino padrao:

```text
<broadcast-da-LAN>:29500/udp
```

`/root/split-master.sh` descobre o broadcast da interface usada pela rota
padrao. O endereco `192.168.122.255` pertence apenas ao NAT libvirt usado nos
testes; nao faz parte do protocolo.

### Reply

O agente responde ao IP/porta de origem da probe com JSON UTF-8 sem prefixo:

```json
{
  "node_id": "node-25a2a52c",
  "hostname": "archlinux",
  "control_port": 29510,
  "cpu_count": 2,
  "available_memory": 3445620736
}
```

O campo `available_memory` usa bytes de `MemAvailable` em `/proc/meminfo`. O
agente nao anuncia seu proprio IP no JSON: o master usa o endereco de origem do
datagrama para impedir que um payload anuncie outro host.

CPU e memoria sao exibidos ao operador, mas o planejador atual divide camadas
igualmente. Eles ainda nao ponderam automaticamente o tamanho de cada fatia.

## Plano Vertical

Para `L` camadas e `N` maquinas totais, `_split_layers` cria `N` intervalos
contiguos e balanceados. O resto da divisao e entregue aos primeiros intervalos.

Exemplo com 24 camadas e duas maquinas:

```text
master/head: layers [0, 12)
worker final: layers [12, 24)
```

Exemplo com 24 camadas e tres maquinas:

```text
master/head:        layers [0, 8)
worker intermediario: layers [8, 16)
worker final:       layers [16, 24)
```

O head conhece o primeiro worker. Cada worker conhece apenas o proximo worker,
quando existe, e o worker final conhece o endereco de token-back do head.

### Limites De Escala

- O limite configurado e 24 maquinas totais: um master e ate 23 workers.
- O total de maquinas nao pode superar o numero de camadas do modelo.
- Todos os agentes precisam ser alcancaveis pelo broadcast e pelas portas TCP.
- A topologia de duas maquinas esta validada; 24 maquinas ainda nao foram
  testadas simultaneamente.
- O plano atual balanceia quantidade de camadas, nao velocidade ou RAM por no.

## O Que Cada Maquina Recebe E Carrega

### Master/head

- Recebe replies de discovery e respostas dos agentes.
- Recebe `READY`/`ERROR` de todos os workers.
- Recebe um token `int64` do worker final a cada passo.
- Carrega embeddings e o primeiro intervalo de camadas.
- Mantem o prompt, historico, IDs gerados e KV cache de suas camadas.
- Nao recebe pesos de outros nos pela rede do cluster.

### Worker intermediario

- Recebe um `VerticalJob` por TCP/29510.
- Recebe hidden states do no anterior por TCP/9999.
- Carrega somente seu intervalo logico de camadas.
- Mantem KV cache apenas das suas camadas.
- Envia hidden states ao proximo worker.

### Worker final

- Recebe um `VerticalJob` por TCP/29510.
- Recebe hidden states do no anterior por TCP/9999.
- Carrega suas camadas, o `norm` final e o `lm_head`.
- Calcula logits, executa `argmax` e envia somente o ID do token ao head.

O loader baixa arquivos do Hugging Face em cada maquina. Se o checkpoint usa um
unico `model.safetensors`, cada maquina baixa o arquivo completo mesmo carregando
somente uma fatia. Em checkpoints divididos, baixa apenas os shards que contem
pesos necessarios para a fatia; um shard ainda pode conter pesos extras.

## Controle Do Agente

O agente em TCP/29510 recebe uma mensagem por conexao. O framing e:

```text
json_size    4 bytes, unsigned big-endian
json         json_size bytes em UTF-8
```

### Tipos De Request

| `type` | Campos | Efeito |
|---|---|---|
| `START` | `type`, `job` | Valida e inicia um subprocesso de inferencia |
| `STOP` | somente `type` | Termina o subprocesso atual, se existir |

Exemplo de `START` para o worker final:

```json
{
  "type": "START",
  "job": {
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "layer_start": 12,
    "layer_end": 24,
    "total_layers": 24,
    "listen_port": 9999,
    "control_to": "192.168.122.208:10001",
    "node_id": "node-25a2a52c",
    "connect_to": null,
    "token_back_to": "192.168.122.208:9999",
    "max_new_tokens": 50,
    "precision": "bfloat16"
  }
}
```

O schema do job e exato: campos ausentes ou desconhecidos sao rejeitados. O
agente valida nomes, IPv4:porta, intervalo de camadas, precisao e limites. Ele
constroi uma lista de argumentos e usa `subprocess.Popen` sem `shell=True`.

### Tipos De Response

| `type` | Campos adicionais | Significado |
|---|---|---|
| `STARTED` | `node`, `pid` | Processo criado; ainda nao significa modelo carregado |
| `STOPPED` | `node` | Processo encerrado ou nenhum processo ativo |
| `ERROR` | `node`, `message` | Request invalido ou agente ocupado |

Um segundo `START` e rejeitado enquanto o subprocesso anterior estiver vivo.
Quando o subprocesso termina, o agente pode aceitar um novo job.

## Controle Runtime

Cada subprocesso worker conecta diretamente ao head em TCP/10001. Usa o mesmo
framing `4-byte size + JSON`.

| `type` | Origem -> destino | Campos |
|---|---|---|
| `READY` | worker -> head | `type`, `node` |
| `ERROR` | worker -> head | `type`, `node`, `stage`, `message` |
| `STOP` | head -> worker | `type`, `node=head` |

`stage` pode ser `startup` ou `run`. O head nao inicia a geracao antes de
receber um `READY` de cada worker esperado. O `STOP` runtime e best-effort; a
limpeza garantida no fluxo normal e feita tambem pelo `STOP` enviado aos agentes.

## Frame Binario De Tensor

Hidden states, IDs de entrada e tokens usam um frame binario sem Pickle,
`torch.save` ou `torch.load`:

```text
magic          2 bytes  "SL"
version        1 byte   1
flags          1 byte
dtype          1 byte
dimensions     1 byte
payload_size   4 bytes  unsigned big-endian
shape          4 bytes por dimensao, unsigned big-endian
tensor         payload_size bytes contiguos no dtype original
```

O cabecalho fixo tem 10 bytes. O receptor valida magic, versao, flags, dtype e
se `payload_size == produto(shape) * tamanho(dtype)`.

### Flags

| Valor | Significado |
|---:|---|
| 0 | Continuar usando o KV cache atual |
| 1 | Reiniciar o KV cache antes deste request |

Outros valores sao rejeitados.

### Codigos De Dtype

| Codigo | Torch dtype |
|---:|---|
| 1 | `float16` |
| 2 | `bfloat16` |
| 3 | `float32` |
| 4 | `float64` |
| 5 | `int8` |
| 6 | `uint8` |
| 7 | `int16` |
| 8 | `int32` |
| 9 | `int64` |
| 10 | `bool` |

Um token `int64` com shape `[1]` ocupa 22 bytes: 10 de cabecalho, 4 de shape e
8 de payload. As estatisticas contam o frame da aplicacao, sem headers TCP/IP.

## Enumeracoes De Configuracao

### Precisao

```text
auto | bfloat16 | float32 | int8 | int4
```

O master usa `bfloat16` por padrao. `int8` e `int4` exigem TorchAO na imagem.

### Arquiteturas suportadas

```text
qwen2 | llama
```

Esses valores vem de `config.model_type`; o nome do repositorio Hugging Face
nao substitui essa verificacao.

## Fluxo Por Token

No primeiro passo de uma request:

1. O head aplica o template de chat e cria os IDs do prompt.
2. O head executa embeddings e suas camadas para todo o prompt.
3. O frame enviado ao primeiro worker usa `reset_cache=1`.
4. Cada worker processa a sequencia e preenche seu KV cache local.
5. O worker final calcula logits da ultima posicao e escolhe `argmax`.
6. O token volta diretamente ao head.

Nos passos seguintes:

1. O head acrescenta o token aos IDs gerados.
2. Cada no processa somente o novo token, usando seu KV cache.
3. Hidden state com shape tipico `[1, 1, hidden_size]` segue pela cadeia.
4. O token final volta ao head e o ciclo se repete.

A geracao para quando o token e EOS ou quando atinge `max_new_tokens`. Esse
parametro e um limite, nao uma quantidade obrigatoria.

## Encerramento E Falhas

| Evento | Comportamento atual |
|---|---|
| Master termina normalmente | `finally` envia `STOP` a todos os agentes iniciados |
| Operador usa `Ctrl-C` no master | O mesmo `finally` tenta encerrar todos os jobs |
| Worker reporta erro | Envia `ERROR`; o head aborta e o master limpa os agentes |
| Subprocesso worker fecha | O agente permanece vivo e pode aceitar outro `START` |
| Socket upstream fecha | O worker detecta `SocketClosedError` e encerra |
| Maquina worker reinicia | systemd volta a iniciar o agente, mas a geracao atual falha |
| Master recebe `SIGKILL` ou perde energia | Nao executa `finally`; depende do fechamento TCP |
| Particao de rede silenciosa | Um worker pode bloquear indefinidamente |
| Agente `split-cluster worker` crasha | O systemd atual nao garante restart automatico do agente dentro do tmux |

Ainda nao existem heartbeat, lease de job, timeout nos sockets persistentes,
TCP keepalive configurado, retry da geracao, replanejamento ou failover de
master. Em uma conexao meio aberta, o agente pode considerar o subprocesso
ocupado e rejeitar outro `START` ate intervencao manual ou reboot.

Recuperacao manual de uma appliance:

```bash
systemctl restart split-models
tmux attach -t split-models
```

## Seguranca E Compatibilidade

- A rede deve ser confiavel e filtrada por firewall.
- Nao ha autenticacao, criptografia, assinatura ou protecao contra replay.
- O agente rejeita comandos de shell e campos fora do schema, mas um cliente da
  LAN ainda pode solicitar downloads de modelos e consumir CPU/RAM.
- Apenas o frame tensorial possui `version=1`; discovery e JSON nao carregam
  numero de versao.
- Atualizacoes incompativeis de JSON exigem atualizar todas as maquinas juntas.

## Boot LCC Com Layout Casper

Quando um GRUB externo e imutavel procura `/casper/vmlinuz` e
`/casper/initrd`, a ISO publica copias do kernel e initramfs nesses caminhos.
O hook initramfs `casper-compat` traduz `iso-scan/filename`, `fromiso`,
`isofrom`, `findiso` ou `img_loop` para um loop device entregue ao Archiso como
`archisodevice`.

Esse caminho serve para uma ISO copiada como arquivo para um filesystem local.
O filesystem precisa estar visivel no initramfs sem desbloqueio adicional. Boot
USB nativo continua usando El Torito/GRUB ou SYSLINUX e ignora o shim.

O fluxo foi validado com um GRUB externo carregando `/casper/vmlinuz` e
`/casper/initrd` com `boot=casper iso-scan/filename=/split-models.iso`. O shim
criou `/dev/loop0` para a ISO copiada, e o Archiso montou esse loop em
`/run/archiso/bootmnt` antes de abrir o EROFS interno.
