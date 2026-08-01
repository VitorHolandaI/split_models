#!/usr/bin/env bash
# Compila o llama.cpp fixado e injeta o resultado no airootfs do profile.
# Feito para "source" a partir de deploy/build_llamacpp_archiso.sh.
#
# Nao ha pacote nem repositorio local de proposito. A ISO e imutavel e
# stateless: nunca havera upgrade, remocao ou resolucao de conflito dentro dela,
# entao o que o pacman ofereceria nao se aplica. mkarchiso copia airootfs/ direto
# para o rootfs (mkarchiso:316), e isso basta.
#
# O chamador ja definiu repo_root, build_root e staged_profile via
# deploy/lib_archiso_build.sh.

# shellcheck source=deploy/llama-version.env
source "$repo_root/deploy/llama-version.env"

# Fora dos mktemp que o trap descarta: compilar llama.cpp com todas as variantes
# de CPU e os shaders Vulkan e caro demais para refazer a cada ISO.
llama_cache="$build_root/llamacpp-src"
llama_tree="$llama_cache/llama.cpp-$LLAMA_TAG"
llama_tarball="$llama_cache/llama.cpp-$LLAMA_TAG.tar.gz"
llama_destdir="$llama_cache/dest-$LLAMA_TAG"

# GGML_RPC_RDMA=OFF e obrigatorio, nao preferencia. ggml/src/ggml-rpc/
# CMakeLists.txt:12-22 procura libibverbs NO HOST DE BUILD e liga RDMA sozinho se
# achar. Este host tem rdma-core; a ISO nao, entao libggml-rpc.so sairia linkada
# contra uma lib ausente da imagem e o backend RPC nao carregaria -- so na
# maquina fisica, longe daqui. O alvo e Ethernet, RDMA nao serve para nada aqui.
#
# GGML_BACKEND_DL carrega backend como plugin, entao a mesma imagem serve host
# com e sem Vulkan: CPU vira fallback em vez de erro de link.
# GGML_CPU_ALL_VARIANTS exige GGML_NATIVE=OFF e escolhe a ISA em runtime --
# obrigatorio numa ISO compilada numa maquina e bootada em outras.
# LLAMA_TOOLS_INSTALL e o que instala o ggml-rpc-server: tools/rpc/CMakeLists.txt
# poe a regra de install atras dessa opcao, e tools/CMakeLists.txt so entra no
# diretorio rpc quando GGML_RPC esta ligado.
llama_cmake_options=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX=/usr
    -DBUILD_SHARED_LIBS=ON
    -DGGML_BACKEND_DL=ON
    -DGGML_CPU=ON
    -DGGML_CPU_ALL_VARIANTS=ON
    -DGGML_NATIVE=OFF
    -DGGML_VULKAN=ON
    -DGGML_RPC=ON
    -DGGML_RPC_RDMA=OFF
    -DLLAMA_BUILD_TESTS=OFF
    -DLLAMA_BUILD_EXAMPLES=OFF
    -DLLAMA_BUILD_TOOLS=ON
    -DLLAMA_BUILD_SERVER=ON
    -DLLAMA_TOOLS_INSTALL=ON
)

require_llama_build_commands() {
    local command_name
    for command_name in cmake curl gcc g++ glslc; do
        command -v "$command_name" >/dev/null || {
            printf 'comando ausente: %s\n' "$command_name" >&2
            printf 'instale com: sudo pacman -S --needed cmake shaderc vulkan-headers spirv-headers\n' >&2
            exit 1
        }
    done
    # spirv-headers e pacote separado de spirv-tools no Arch, e a falta dele so
    # aparece no configure do backend Vulkan (ggml/src/ggml-vulkan/CMakeLists.txt).
    local header_file
    for header_file in /usr/include/vulkan/vulkan.h \
        /usr/share/cmake/SPIRV-Headers/SPIRV-HeadersConfig.cmake; do
        [[ -f "$header_file" ]] || {
            printf 'header de build ausente: %s\n' "$header_file" >&2
            printf 'instale com: sudo pacman -S --needed vulkan-headers spirv-headers\n' >&2
            exit 1
        }
    done
}

# O digest e conferido sempre, nao so no download: um tarball em cache corrompido
# ou trocado passaria despercebido e viraria binario ruim dentro da ISO.
fetch_llama_source() {
    mkdir -p "$llama_cache"
    [[ -f "$llama_tarball" ]] || curl -fsSL -o "$llama_tarball" \
        "https://github.com/ggml-org/llama.cpp/archive/refs/tags/$LLAMA_TAG.tar.gz"
    printf '%s  %s\n' "$LLAMA_SHA256" "$llama_tarball" | sha256sum -c --quiet || {
        printf 'digest do llama.cpp %s nao confere; esperado %s\n' \
            "$LLAMA_TAG" "$LLAMA_SHA256" >&2
        exit 1
    }
    [[ -d "$llama_tree" ]] || tar xzf "$llama_tarball" -C "$llama_cache"
}

build_llama_binaries() {
    cmake -B "$llama_tree/build" -S "$llama_tree" "${llama_cmake_options[@]}"
    cmake --build "$llama_tree/build" --config Release --parallel "$(nproc)"
}

# ggml-rpc-server e o binario cuja ausencia nao aparece no build: compila e
# simplesmente nao e instalado quando LLAMA_TOOLS_INSTALL sai do ar. Os backends
# entram na lista pelo mesmo motivo -- GGML_BACKEND_DL os torna plugin, entao um
# backend que nao compilou vira "sem aceleracao" em runtime, nunca erro de build.
#
# Plugin de backend e instalado ao lado do executavel, em usr/bin, nao em
# usr/lib: e onde o ggml procura em tempo de execucao.
assert_llama_artifacts() {
    local expected
    for expected in usr/bin/llama-cli usr/bin/llama-server usr/bin/ggml-rpc-server \
                    usr/bin/libggml-rpc.so usr/bin/libggml-vulkan.so; do
        [[ -f "$llama_destdir/$expected" ]] || {
            printf 'artefato ausente apos install: %s\n' "$expected" >&2
            exit 1
        }
    done
    assert_no_host_only_libraries
}

# Confere o binario, nao a opcao: a auto-deteccao de RDMA depende do que estiver
# instalado no host de build, e um NEEDED a mais so quebraria no boot da ISO.
assert_no_host_only_libraries() {
    local library
    for library in "$llama_destdir"/usr/bin/libggml*.so \
                   "$llama_destdir"/usr/lib/libggml*.so*; do
        [[ -f "$library" ]] || continue
        readelf -d "$library" | grep -q 'NEEDED.*libibverbs' && {
            printf '%s depende de libibverbs, ausente na ISO; use -DGGML_RPC_RDMA=OFF\n' \
                "$library" >&2
            exit 1
        }
    done
    return 0
}

install_llama_to_destdir() {
    rm -rf "$llama_destdir"
    DESTDIR="$llama_destdir" cmake --install "$llama_tree/build"
    assert_llama_artifacts
}

# Sem isto nao da para saber, num host ja rodando, qual commit gerou o binario --
# e o backend RPC nao negocia versao entre master e worker, entao a incompati-
# bilidade so aparece em runtime, longe da causa.
write_llama_build_info() {
    local info="$staged_profile/airootfs/usr/share/doc/split-llama-cpp/build-info.txt"
    mkdir -p "$(dirname -- "$info")"
    {
        printf 'upstream: https://github.com/ggml-org/llama.cpp\n'
        printf 'tag: %s\n' "$LLAMA_TAG"
        printf 'commit: %s\n' "$LLAMA_COMMIT"
        printf 'source-sha256: %s\n' "$LLAMA_SHA256"
        printf 'cmake:\n'
        printf '  %s\n' "${llama_cmake_options[@]}"
    } > "$info"
}

# mkarchiso copia airootfs com "cp -af --no-preserve=ownership,mode"
# (mkarchiso:316), entao o bit de execucao se perde e os binarios compilados
# chegam na imagem em 644, como "Permission denied" -- o build passa limpo e a
# falha so aparece rodando o comando dentro da ISO.
#
# A lista e gerada do que foi instalado, nao escrita a mao: o conjunto de
# binarios do llama.cpp muda entre versoes do upstream.
#
# Cobre usr/bin inteiro, executaveis e plugins de backend juntos. Plugin so
# precisa de leitura para o dlopen, e as libs em usr/lib ficam em 644 por isso
# mesmo, mas separar os dois casos aqui exigiria classificar arquivo por
# conteudo sem ganho nenhum.
append_llama_file_permissions() {
    local installed
    {
        printf '\nfile_permissions+=(\n'
        for installed in "$llama_destdir"/usr/bin/*; do
            printf '  ["/usr/bin/%s"]="0:0:755"\n' "${installed##*/}"
        done
        printf ')\n'
    } >> "$staged_profile/profiledef.sh"
}

stage_variant_extras() {
    fetch_llama_source
    build_llama_binaries
    install_llama_to_destdir
    cp -a "$llama_destdir/." "$staged_profile/airootfs/"
    append_llama_file_permissions
    write_llama_build_info
}

# vim: set ft=sh:
