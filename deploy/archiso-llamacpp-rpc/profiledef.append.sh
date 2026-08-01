# Acrescentado ao profiledef do profile base por
# deploy/lib_archiso_build.sh:append_overlay_profiledef. Reatribuir iso_name
# depois da base e o que separa o artefato experimental do estavel: o glob de
# limpeza e ancorado nesse prefixo e nunca alcanca dist/split-models-*.iso.

iso_name="split-models-llamacpp-rpc"
iso_application="Split Models llama.cpp RPC appliance (experimental)"

# Os binarios compilados do llama.cpp recebem 755 por um bloco gerado em
# deploy/lib_llamacpp_build.sh, a partir do que foi de fato instalado. Aqui so
# entram os scripts versionados da variante.
file_permissions+=(
  ["/root/split-llama-master.sh"]="0:0:755"
  ["/usr/local/bin/split-llama"]="0:0:755"
  ["/usr/local/bin/split-llama-session"]="0:0:755"
  ["/usr/local/bin/split-llama-tmux"]="0:0:755"
)
