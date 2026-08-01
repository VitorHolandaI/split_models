# Acrescentado ao profiledef do profile base por
# deploy/lib_archiso_build.sh:append_overlay_profiledef. Reatribuir iso_name
# depois da base e o que separa o artefato experimental do estavel: o glob de
# limpeza e ancorado nesse prefixo e nunca alcanca dist/split-models-*.iso.

iso_name="split-models-llamacpp-rpc"
iso_application="Split Models llama.cpp RPC appliance (experimental)"

# file_permissions das entradas da variante entram junto com os scripts em
# airootfs/. mkarchiso:324 so avisa quando o arquivo nao existe, e aviso
# silencioso em build longo passa despercebido.
