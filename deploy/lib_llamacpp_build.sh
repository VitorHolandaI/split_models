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
            printf 'instale com: sudo pacman -S --needed cmake shaderc vulkan-headers\n' >&2
            exit 1
        }
    done
    [[ -f /usr/include/vulkan/vulkan.h ]] || {
        printf 'headers Vulkan ausentes: /usr/include/vulkan/vulkan.h\n' >&2
        printf 'instale com: sudo pacman -S --needed vulkan-headers\n' >&2
        exit 1
    }
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

# ggml-rpc-server e o unico binario cuja ausencia nao aparece no build: ele
# compila e simplesmente nao e instalado quando LLAMA_TOOLS_INSTALL sai do ar.
assert_llama_artifacts() {
    local expected
    for expected in usr/bin/llama-cli usr/bin/llama-server usr/bin/ggml-rpc-server; do
        [[ -x "$llama_destdir/$expected" ]] || {
            printf 'artefato ausente apos install: %s\n' "$expected" >&2
            exit 1
        }
    done
    compgen -G "$llama_destdir/usr/lib/libggml-rpc.so*" >/dev/null || {
        printf 'backend RPC ausente: %s/usr/lib/libggml-rpc.so*\n' "$llama_destdir" >&2
        exit 1
    }
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

stage_variant_extras() {
    fetch_llama_source
    build_llama_binaries
    install_llama_to_destdir
    cp -a "$llama_destdir/." "$staged_profile/airootfs/"
    write_llama_build_info
}

# vim: set ft=sh:
