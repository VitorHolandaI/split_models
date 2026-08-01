SPLIT MODELS - VARIANTE EXPERIMENTAL llama.cpp RPC
==================================================

AVISO DE SEGURANCA
  O backend RPC do llama.cpp nao tem autenticacao nem criptografia. Quem
  alcanca a porta 50052 executa codigo neste no. Use somente em LAN isolada,
  nunca em rede compartilhada ou com acesso a internet.

Esta imagem e experimental e paralela a estavel. A estavel roda o split
vertical em PyTorch com split-cluster; esta roda llama.cpp por RPC com
split-llama. Sao motores diferentes e comandos diferentes.

O tty1 anexa a sessao tmux "split-models". Painel esquerdo: worker RPC.
Painel direito: bpytop. Ctrl-b + setas alterna. Ctrl-C no painel do worker
antes de promover o no a master.

AJUDA COMPLETA
  split-llama exemplos          comandos prontos, com modelos GGUF
  split-llama --help
  split-llama worker --help
  split-llama nodes --help
  split-llama master --help

MODELOS
  Qualquer repositorio GGUF serve, nao so os Qwen. `split-llama exemplos`
  lista os que ja foram conferidos, com a razao de cada um. Nome de
  repositorio inexistente responde 401 "Invalid username or password":
  parece erro de login e nao e, confira o nome em huggingface.co.

INICIAR MASTER (RECOMENDADO)
  /root/split-llama-master.sh

Sobrescrever os defaults:
  SPLIT_BROADCAST=10.1.2.255 /root/split-llama-master.sh
  SPLIT_LLAMA_MODEL=Qwen/Qwen2.5-3B-Instruct-GGUF /root/split-llama-master.sh

Argumentos extras vao para o llama-cli depois de --:
  /root/split-llama-master.sh -- -p "Explique memoria RAM." -n 40

DESCOBRIR WORKERS
  split-llama nodes
  split-llama nodes --broadcast <broadcast-da-LAN>

INICIAR WORKER MANUALMENTE
  split-llama worker
  split-llama worker --bind 192.168.0.11 --rpc-port 50052

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
