SPLIT MODELS - GUIA RAPIDO
==========================

O tty1 anexa automaticamente a sessao tmux "split-models". O painel esquerdo
mostra o worker/log (65%) e o direito abre btop (35%). Use Ctrl-b + setas para
alternar. Pressione q no btop para deixar um shell no painel direito.
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

ACESSO REMOTO (SSH)
  Login root por chave. Senha esta desligada de proposito: a conta root tem
  senha vazia, entao aceitar senha liberaria root a qualquer host da LAN.

  Da estacao do operador, com a chave da appliance:
    ssh -i split-models-appliance root@<ip-do-no>
    ssh -i split-models-appliance root@<ip-do-no> poweroff

  A imagem e stateless e regenera as chaves de host a cada boot, entao o ssh
  reclama de host key trocada depois de todo reiniciar. Para a LAN dos nos,
  em ~/.ssh/config da estacao:
    Host 192.168.50.*
      IdentityFile ~/.ssh/split-models-appliance
      StrictHostKeyChecking no
      UserKnownHostsFile /dev/null

DESLIGAR A LAN INTEIRA
  for ip in 192.168.50.11 192.168.50.12; do
    ssh -i split-models-appliance root@$ip poweroff
  done

REINICIAR OU DESLIGAR (LOCAL)
  reboot
  poweroff

Todos os modelos, temporarios e resultados ficam em /run/split-models e somem
no reboot. Esta appliance nao usa disco local persistente.
