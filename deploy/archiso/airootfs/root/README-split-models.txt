SPLIT MODELS - GUIA RAPIDO
==========================

O worker inicia automaticamente no boot dentro da sessao tmux "split-models".
Pressione Ctrl-C nessa sessao antes de promover o no a master.
No console fisico, bpytop abre automaticamente. Pressione q para voltar ao shell.

AJUDA COMPLETA
  split-cluster --help
  split-cluster worker --help
  split-cluster master --help
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
