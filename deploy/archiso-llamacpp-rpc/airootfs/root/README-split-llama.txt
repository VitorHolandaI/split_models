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
Painel direito: btop. Ctrl-b + setas alterna. O painel esquerdo so segue os
logs: worker e servidor sao supervisionados pelo systemd
(split-llama-worker.service, split-llama-server.service), nao pelo tmux. O
split-llama-fleet-discovery.service permanece ativo em todos os papeis.

GUIA RAPIDO
  cat /root/README-split-llama.txt   este guia (cabe tudo em uma tela)
  split-llama --help                 todos os subcomandos
  split-llama exemplos               comandos prontos, com modelos GGUF
  split-llama mode status            papel atual deste no

MEDIR ESTE NO (CPU + iGPU + llama.cpp small)
  split-llama hardware-bench         tabela legivel
  split-llama hardware-bench --json  um registro por no, para a frota

  O resultado tambem aparece na propria tela da maquina: a barra do tmux mostra
  `backend=CPU | CPU 31.00 tok/s` ou `backend=Vulkan | iGPU 17.76 TOPS |
  Vulkan 20.00 tok/s` ao lado do papel. Confira sem precisar de SSH de volta.

  Unidades que o hardware-bench mede e reporta:
    TOPS          iGPU INT8 (vkpeak), tera-operacoes por segundo
    TFLOPS        iGPU FP16 (vkpeak), tera-flops por segundo
    MIPS          CPU, rating nativo do `7z b` (nao e TOPS, nao converter)
    tok/s CPU     llama.cpp no CPU, modelo pequeno oficial
    tok/s Vulkan  llama.cpp na iGPU, mesmo modelo pequeno

  Maquina sem Vulkan (ou sem int8-matrix) nao falha: iGPU e tok/s Vulkan
  zeram com aviso e o no passa a ser ordenado pelo tok/s da CPU.

PAPEIS (TROCA SEM REBOOT)
  split-llama become_master
  split-llama mode status
  split-llama mode worker
  split-llama mode idle
  split-llama mode replica --model small --parallel 4
  split-llama mode distributed-master --model large --parallel 8 \
      --auto-discover
  split-llama mode distributed-master --model large --parallel 8 \
      --auto-discover --usb-device /dev/sda1

  Pre-visualizar o plano sem alterar nos nem baixar modelo:
  split-llama master --discover --dry-run \
      --hf ggml-org/gpt-oss-20b-GGUF:MXFP4

  Sem --auto-discover, distributed-master aceita --rpc explicito. Com
  --auto-discover, descobre os nos de controle, devolve os remotos a worker e
  monta os endpoints RPC sem IPs manuais. A transicao tem lock, espera
  GET /health e, em falha, limpa o estado e restaura o modo worker. O estado
  oficial do papel fica em /run/split-models/state.json.

CHAT DISTRIBUIDO COM PLANEJAMENTO (master --discover)
  split-llama master --discover \
      --hf unsloth/Qwen3.5-9B-GGUF:Q6_K \
      -- -c 32768

  O que o --discover faz:
    1. descobre todos os nos pelo plano de controle UDP;
    2. devolve cada no ao modo worker via SSH;
    3. mede cada no (hardware-bench) e imprime a tabela de capacidade;
    4. calcula o minimo seguro (50% da RAM de cada no somada);
    5. pergunta quantas maquinas usar no total, master incluso;
    6. seleciona so os mais rapidos (tok/s Vulkan) e monta os endpoints RPC.

  Tabela que o master imprime, por no:
    NODE            RAM GiB    TOPS    MIPS  tok/s CPU tok/s Vulkan
    10.0.0.2           14.0    8.00    5000       3.00        10.00

  Opcoes do planejamento:
    --machines N           pula a pergunta e usa exatamente N maquinas
    --model-size-gb <GB>   tamanho medido para GGUF fora do catalogo
    --hardware-timeout <s> timeout por no para o hardware-bench (900)

  Nos nao selecionados ficam fora dos endpoints RPC.

CACHE DO MODELO (RAM por padrao; USB somente no master split)
  Todo runtime fica em /run, que e RAM da ISO live. O disco original nunca e
  formatado, montado ou usado para dados do modelo. No distributed, somente o
  master pode usar --usb-device para guardar o GGUF completo; os workers
  recebem somente as layers via RPC e as mantem em RAM. Replica usa RAM para
  a copia completa e nao precisa de USB.
  Sem --usb-device, o servidor usa cache temporario em RAM.
  Ao usar USB, o espaco livre e validado ANTES de apagar o download anterior
  (reserva de 0.5 GB): modelo que nao cabe nao destroi o cache existente.

  Conferir o que esta montado:
  findmnt /run/split-models/model-cache
  lsblk -o NAME,FSTYPE,TRAN,TYPE,MOUNTPOINTS
  split-llama mode replica --model small

MODELO GRANDE (USB de 16 GB)
  O maior GGUF conferido do catalogo cabe em um USB ext4 de 16 GB
  (12.11 GB + reserva; ~15.5 GB livres):
    split-llama master --discover \
        --usb-device /dev/sda1 \
        --hf ggml-org/gpt-oss-20b-GGUF:MXFP4 \
        -- -c 32768

  O master le o GGUF completo do USB; os workers recebem somente as layers
  atribuidas via RPC.

  O dry-run escolhe automaticamente o menor prefixo seguro quando --machines
  nao e informado e mostra RAM segura, GB estimados e faixas contiguas de
  layers. Modelos conhecidos detectam o total de layers automaticamente. As
  faixas sao proporcionais; o llama.cpp pode ajusta-las pelos tamanhos reais
  dos tensores. --model-layers so e necessario como override fora do catalogo.

FROTA (coordena os nos por SSH, sem IPs manuais)
  split-llama fleet nodes                        todos os nos e seus modos
  split-llama fleet bench \
      --scenario replicated --model small --profile users-2k --auto-discover
  split-llama fleet bench \
      --scenario distributed --model small --profile users-2k --auto-discover
  split-llama fleet reset --auto-discover        volta todos a worker
  split-llama fleet reboot                       reinicia todos os nos

DESCOBRIR WORKERS RPC (so os vivos)
  split-llama nodes
  split-llama nodes --broadcast <broadcast-da-LAN>

DESCOBRIR TODA A FROTA (INCLUSIVE REPLICAS E MASTERS)
  split-llama fleet nodes
  split-llama fleet nodes --broadcast-address <broadcast-da-LAN>

MASTER DE UM TURNO (ATALHO INTERATIVO, FORA DO CONTROLE DE MODOS)
  /root/split-llama-master.sh

  Sobrescrever os defaults:
  SPLIT_BROADCAST=203.0.113.255 /root/split-llama-master.sh
  SPLIT_LLAMA_MODEL=Qwen/Qwen2.5-3B-Instruct-GGUF /root/split-llama-master.sh
  SPLIT_USB_DEVICE=/dev/sda1 /root/split-llama-master.sh

Argumentos extras vao para o llama-cli depois de --:
  /root/split-llama-master.sh -- -p "Explique memoria RAM." -n 40

AJUDA COMPLETA (cada subcomando tem a propria)
  split-llama worker --help
  split-llama nodes --help
  split-llama master --help
  split-llama mode --help
  split-llama fleet --help
  split-llama bench-capacity --help

MODELOS
  Qualquer repositorio GGUF serve, nao so os Qwen. `split-llama exemplos`
  lista os que ja foram conferidos, com a razao de cada um. Nome de
  repositorio inexistente responde 401 "Invalid username or password":
  parece erro de login e nao e, confira o nome em huggingface.co.

UNIDADES SUPERVISIONADAS
  systemctl status split-llama-worker.service
  systemctl status split-llama-server.service
  systemctl status split-llama-fleet-discovery.service
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
Esta appliance nao usa disco local persistente; o USB ext4 e opcional e so
deve ser selecionado no master split para o GGUF grande.
