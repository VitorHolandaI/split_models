if [[ "$(tty)" == "/dev/tty1" ]] && command -v bpytop >/dev/null; then
    bpytop
    printf 'Comandos da appliance: cat /root/README-split-models.txt\n'
fi
