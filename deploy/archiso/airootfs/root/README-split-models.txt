SPLIT MODELS - GUIA RAPIDO
==========================

O tty1 anexa automaticamente a sessao tmux "split-models". O painel esquerdo
mostra o worker/log (65%) e o direito abre bpytop (35%). Use Ctrl-b + setas para
alternar. Pressione q no bpytop para deixar um shell no painel direito.
Pressione Ctrl-C no painel do worker antes de promover o no a master.
A barra inferior mostra sempre role, node ID e IPv4 local desta maquina.

AJUDA COMPLETA
  split-cluster --help
  split-cluster worker --help
  split-cluster master --help
  split-cluster exemplos
  split-cluster --comandos-exemplos
  split-lm --help
  cat /opt/split-models/docs/protocol.md

INICIAR MASTER (RECOMENDADO)
  /root/split-master.sh

O script detecta automaticamente o broadcast da LAN da rota padrao. Ele nao
depende do range usado no libvirt e funciona em bare metal com DHCP.

Sobrescrever os defaults quando necessario:
  SPLIT_BROADCAST=10.1.2.255 /root/split-master.sh
  SPLIT_MODEL=Qwen/Qwen2.5-3B-Instruct /root/split-master.sh
  SPLIT_MAX_NEW_TOKENS=100 /root/split-master.sh

DESCOBRIR WORKERS
  split-cluster nodes
  split-cluster nodes --broadcast <broadcast-da-LAN>

INICIAR WORKER MANUALMENTE
  split-cluster worker

BENCHMARKS
  split-cluster benchmarks
  split-cluster master --discover --benchmark-profile <perfil>
  split-cluster benchmark-local --benchmark-profile <perfil>

SERVICO E TMUX
  systemctl status split-models
  systemctl restart split-models
  tmux attach -t split-models
  tmux list-sessions

REDE
  ip -brief address
  ip route

REINICIAR OU DESLIGAR
  reboot
  poweroff

Todos os modelos, temporarios e resultados ficam em /run/split-models e somem
no reboot. Esta appliance nao usa disco local persistente.
