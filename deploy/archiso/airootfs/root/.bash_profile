# tty1: console fisica; ttyS0: serial do harness de VM (console=ttyS0 no
# cmdline faz o serial-getty abrir o login em vez do getty do tty1).
# return, nao exit: este arquivo e sourced pelo shell de login (SSH incluso).
case "$(tty)" in /dev/tty1|/dev/ttyS0) ;; *) return 0 ;; esac
if command -v tmux >/dev/null; then
    for ((attempt = 0; attempt < 30; attempt++)); do
        tmux has-session -t split-models 2>/dev/null && break
        sleep 1
    done
    if tmux has-session -t split-models 2>/dev/null; then
        tmux attach-session -t split-models
    else
        printf 'Sessao tmux split-models indisponivel apos 30 segundos.\n' >&2
    fi
    printf 'Comandos da appliance: cat /root/README-split-models.txt\n'
fi
