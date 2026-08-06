SPLIT MODELS - VARIANTE EXPERIMENTAL llama.cpp RPC
==================================================

AVISO DE SEGURANCA
  O backend RPC do llama.cpp nao tem autenticacao nem criptografia. Quem
  alcanca a porta 50052 executa codigo neste no. Use somente em LAN isolada,
  nunca em rede compartilhada ou com acesso a internet.

Esta imagem e experimental e paralela a estavel. A estavel roda o split
vertical em PyTorch com split-cluster; esta roda llama.cpp por RPC com
split-llama. Sao motores diferentes e comandos diferentes.

O tty1 anexa a sessao tmux "split-models". Painel esquerdo: logs das unidades.
Painel direito: bpytop. Ctrl-b + setas alterna. O painel esquerdo so segue os
logs: worker e servidor sao supervisionados pelo systemd
(split-llama-worker.service, split-llama-server.service), nao pelo tmux.

PAPEIS (TROCA SEM REBOOT)
  split-llama mode status
  split-llama mode worker
  split-llama mode idle
  split-llama mode replica --model small --parallel 4
  split-llama mode distributed-master --model large --parallel 8 \
      --rpc <ip1>:50052,<ip2>:50052

  O distributed-master exige --rpc explicito. A transicao tem lock, espera
  GET /health e, em falha, limpa o estado e restaura o modo worker. O estado
  oficial do papel fica em /run/split-models/state.json.

AJUDA COMPLETA
  split-llama exemplos          comandos prontos, com modelos GGUF
  split-llama --help
  split-llama worker --help
  split-llama nodes --help
  split-llama master --help
  split-llama mode --help

MODELOS
  Qualquer repositorio GGUF serve, nao so os Qwen. `split-llama exemplos`
  lista os que ja foram conferidos, com a razao de cada um. Nome de
  repositorio inexistente responde 401 "Invalid username or password":
  parece erro de login e nao e, confira o nome em huggingface.co.

MASTER DE UM TURNO (ATALHO INTERATIVO, FORA DO CONTROLE DE MODOS)
  /root/split-llama-master.sh

Sobrescrever os defaults:
  SPLIT_BROADCAST=10.1.2.255 /root/split-llama-master.sh
  SPLIT_LLAMA_MODEL=Qwen/Qwen2.5-3B-Instruct-GGUF /root/split-llama-master.sh

Argumentos extras vao para o llama-cli depois de --:
  /root/split-llama-master.sh -- -p "Explique memoria RAM." -n 40

DESCOBRIR WORKERS
  split-llama nodes
  split-llama nodes --broadcast <broadcast-da-LAN>

UNIDADES SUPERVISIONADAS
  systemctl status split-llama-worker.service
  systemctl status split-llama-server.service
  journalctl -fu split-llama-worker.service

CONFERIR DISPOSITIVOS
  llama-cli --list-devices
  llama-cli --rpc <ip1>:50052,<ip2>:50052 --list-devices
  vulkaninfo --summary

Um no sem Vulkan nao falha: o ggml-rpc-server cai em CPU. Se um acelerador
que voce esperava nao aparecer no --list-devices, o no esta contribuindo so
CPU, e a inferencia funciona mas sem ganho.

QUAL BUILD ESTA RODANDO
  cat /usr/share/doc/split-llama-cpp/build-info.txt

O RPC nao negocia versao entre master e worker. Se os nos discordarem, a
falha aparece no handshake, longe da causa: confira que o commit e o mesmo
em todas as maquinas.

REDE
  ip -brief address
  ip route

ACESSO REMOTO
  ssh -i split-models-appliance root@<ip-do-no>
  ssh -i split-models-appliance root@<ip-do-no> poweroff

REINICIAR OU DESLIGAR
  reboot
  poweroff

Modelos, cache e temporarios ficam em /run/split-models e somem no reboot.
Esta appliance nao usa disco local persistente.
