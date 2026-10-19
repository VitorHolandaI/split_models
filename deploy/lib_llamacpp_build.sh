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
# shellcheck source=deploy/vkpeak-version.env
source "$repo_root/deploy/vkpeak-version.env"

# Fora dos mktemp que o trap descarta: compilar llama.cpp com todas as variantes
# de CPU e os shaders Vulkan e caro demais para refazer a cada ISO.
llama_cache="$build_root/llamacpp-src"
llama_tree="$llama_cache/llama.cpp-$LLAMA_TAG"
llama_tarball="$llama_cache/llama.cpp-$LLAMA_TAG.tar.gz"
llama_destdir="$llama_cache/dest-$LLAMA_TAG"
vulkan_toolchain_stamp="$llama_tree/build/.split-models-glslc-version"
vkpeak_zip="$llama_cache/vkpeak-$VKPEAK_RELEASE-linux.zip"
vkpeak_tree="$llama_cache/vkpeak-$VKPEAK_RELEASE-linux"

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
    # Sem isto o Debian/Ubuntu instala em usr/lib/x86_64-linux-gnu (multiarch) e
    # o Arch em usr/lib. Duas consequencias, uma barulhenta e uma silenciosa: o
    # LD_LIBRARY_PATH impresso no fim aponta para o lugar errado, e o glob de
    # assert_no_host_only_libraries deixa de casar, checando nada.
    -DCMAKE_INSTALL_LIBDIR=lib
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

# A frota mistura Arch (a appliance) e Ubuntu (maquinas com GPU que nao bootam
# a ISO), entao a mensagem nomeia os dois gerenciadores: dizer so "pacman" numa
# maquina Ubuntu manda o operador procurar um pacote que nao existe ali.
_install_hint() {
    printf '  Arch:   sudo pacman -S --needed %s\n' "$1" >&2
    printf '  Ubuntu: sudo apt install -y %s\n' "$2" >&2
}

require_llama_build_commands() {
    local command_name
    for command_name in cmake curl gcc g++ glslc unzip; do
        command -v "$command_name" >/dev/null || {
            printf 'comando ausente: %s\n' "$command_name" >&2
            _install_hint "cmake shaderc vulkan-headers spirv-headers" \
                "build-essential cmake curl glslc libvulkan-dev spirv-headers"
            exit 1
        }
    done
    # spirv-headers e pacote separado de spirv-tools nas duas distribuicoes, e a
    # falta dele so aparece no configure do backend Vulkan
    # (ggml/src/ggml-vulkan/CMakeLists.txt). O caminho do arquivo e o mesmo nas
    # duas, entao uma checagem serve para ambas.
    # openssl entra na lista porque sua falta nao quebra o build: o llama.cpp
    # compila sem HTTPS e so recusa na hora de baixar, com "HTTPS is not
    # supported", depois de meia hora de compilacao ja gasta.
    local header_file
    for header_file in /usr/include/vulkan/vulkan.h \
        /usr/share/cmake/SPIRV-Headers/SPIRV-HeadersConfig.cmake \
        /usr/include/openssl/ssl.h; do
        [[ -f "$header_file" ]] || {
            printf 'header de build ausente: %s\n' "$header_file" >&2
            _install_hint "vulkan-headers spirv-headers openssl" \
                "libvulkan-dev spirv-headers libssl-dev"
            exit 1
        }
    done
}

fetch_vkpeak_release() {
    [[ -f "$vkpeak_zip" ]] || curl -fsSL -o "$vkpeak_zip" \
        "https://github.com/nihui/vkpeak/releases/download/$VKPEAK_RELEASE/vkpeak-$VKPEAK_RELEASE-linux.zip"
    printf '%s  %s\n' "$VKPEAK_SHA256" "$vkpeak_zip" | sha256sum -c --quiet || {
        printf 'digest do vkpeak %s nao confere; esperado %s\n' \
            "$VKPEAK_RELEASE" "$VKPEAK_SHA256" >&2
        exit 1
    }
    [[ -x "$vkpeak_tree/vkpeak" ]] || unzip -q -o "$vkpeak_zip" -d "$llama_cache"
}

stage_vkpeak_to_destdir() {
    fetch_vkpeak_release
    install -Dm755 "$vkpeak_tree/vkpeak" "$llama_destdir/usr/bin/vkpeak"
    install -Dm644 "$vkpeak_tree/LICENSE" \
        "$llama_destdir/usr/share/licenses/vkpeak/LICENSE"
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

# CUDA e opcional e so para build local, nunca para a ISO. O backend CUDA usa
# CMAKE_CUDA_ARCHITECTURES "native" por padrao (ggml/src/ggml-cuda/CMakeLists.txt),
# entao o binario serve so a GPU que compilou -- inaceitavel numa imagem que roda
# em maquinas diferentes, e o toolkit ainda dobraria o tamanho dela.
#
# Num worker RPC dedicado e outra historia: compila na propria maquina, para a
# propria placa. Backends diferentes entre master e worker nao sao problema, o
# RPC transporta tensores e nao instrucoes; o que precisa casar e o commit.
llama_cuda_options() {
    [[ "${SPLIT_LLAMA_CUDA:-0}" == "1" ]] || return 0
    printf '%s\n' -DGGML_CUDA=ON
}

# Os outputs customizados do Vulkan nao dependem do executavel glslc no CMake.
# Depois de um upgrade do shaderc, o configure habilitou E2M1/E4M3 no C++, mas
# manteve o header de shaders antigo sem os simbolos OCP FP4. Limpar somente
# quando a assinatura muda preserva o build incremental comum.
vulkan_toolchain_signature() {
    glslc --version
}

invalidate_vulkan_shader_cache() {
    local build_dir="$llama_tree/build" current previous=""
    [[ -f "$build_dir/CMakeCache.txt" ]] || return 0
    current="$(vulkan_toolchain_signature)"
    [[ -f "$vulkan_toolchain_stamp" ]] && previous="$(<"$vulkan_toolchain_stamp")"
    [[ "$current" == "$previous" ]] && return 0
    printf 'glslc mudou; limpando shaders Vulkan incrementais\n' >&2
    cmake --build "$build_dir" --target clean
}

record_vulkan_toolchain_signature() {
    vulkan_toolchain_signature > "$vulkan_toolchain_stamp"
}

build_llama_binaries() {
    local extra=()
    mapfile -t extra < <(llama_cuda_options)
    cmake -B "$llama_tree/build" -S "$llama_tree" \
        "${llama_cmake_options[@]}" ${extra[@]+"${extra[@]}"}
    invalidate_vulkan_shader_cache
    cmake --build "$llama_tree/build" --config Release --parallel "$(nproc)"
    record_vulkan_toolchain_signature
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
    for expected in usr/bin/llama-cli usr/bin/llama-server usr/bin/llama-bench \
                    usr/bin/llama-batched-bench usr/bin/ggml-rpc-server \
                    usr/bin/libggml-rpc.so usr/bin/libggml-vulkan.so; do
        [[ -f "$llama_destdir/$expected" ]] || {
            printf 'artefato ausente apos install: %s\n' "$expected" >&2
            exit 1
        }
    done
    assert_requested_cuda_backend
    assert_no_host_only_libraries
}

# Pedir CUDA e receber um build sem ela e silencioso: os backends sao plugins,
# entao o worker sobe, cai em Vulkan e serve a mesma GPU mais devagar. Quem
# pediu CUDA merece saber que nao recebeu.
assert_requested_cuda_backend() {
    [[ "${SPLIT_LLAMA_CUDA:-0}" == "1" ]] || return 0
    [[ -f "$llama_destdir/usr/bin/libggml-cuda.so" ]] || {
        printf 'SPLIT_LLAMA_CUDA=1 pedido, mas libggml-cuda.so nao foi instalado.\n' >&2
        printf 'Confira se o nvcc esta presente: nvcc --version\n' >&2
        exit 1
    }
}

# Confere o binario, nao a opcao: a auto-deteccao de RDMA depende do que estiver
# instalado no host de build, e um NEEDED a mais so quebraria no boot da ISO.
assert_no_host_only_libraries() {
    local library found=0
    for library in "$llama_destdir"/usr/bin/libggml*.so \
                   "$llama_destdir"/usr/lib/libggml*.so*; do
        [[ -f "$library" ]] || continue
        found=1
        readelf -d "$library" | grep -q 'NEEDED.*libibverbs' && {
            printf '%s depende de libibverbs, ausente na ISO; use -DGGML_RPC_RDMA=OFF\n' \
                "$library" >&2
            exit 1
        }
    done
    # Um glob que nao casa nada percorre zero arquivos e "passa". Foi o que
    # aconteceu no Ubuntu, onde as libs foram para usr/lib/x86_64-linux-gnu: a
    # checagem de RDMA nao reprovou nada porque nao olhou nada.
    (( found )) || {
        printf 'nenhuma libggml encontrada em %s; layout de instalacao mudou\n' \
            "$llama_destdir" >&2
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

# A ISO roda em maquinas diferentes da que compilou; um build CUDA "native" so
# serviria a placa do host de build, e o toolkit dobraria a imagem. Chamado pelo
# script da variante antes de encenar nada: recusar depois do uv sync gastaria
# minutos para so entao reclamar de uma variavel de ambiente.
refuse_cuda_for_image() {
    [[ "${SPLIT_LLAMA_CUDA:-0}" == "1" ]] || return 0
    printf 'SPLIT_LLAMA_CUDA=1 nao vale para a ISO: o build CUDA e nativo da\n' >&2
    printf 'GPU que compila. Use deploy/build_llamacpp_binaries.sh na maquina.\n' >&2
    exit 1
}

# O controlador llama_rpc usa somente a stdlib. O fonte encenado fica no
# sys.path ao executar Python a partir de /opt/split-models, entao instalar o
# projeto puxaria Torch e Transformers sem necessidade para dentro da imagem.
stage_python_environment() { :; }

stage_variant_extras() {
    fetch_llama_source
    build_llama_binaries
    install_llama_to_destdir
    stage_vkpeak_to_destdir
    cp -a "$llama_destdir/." "$staged_profile/airootfs/"
    append_llama_file_permissions
    write_llama_build_info
    embed_fleet_key
}

# A ISO ja traz a publica em authorized_keys; sem a privada o fleet nao roda
# de dentro do no (caminho fixo /root/.ssh/fleet_ed25519). A chave fica no
# diretorio deploy/keys (gitignored) e nunca no historico: quem tem a ISO
# controla a frota toda, entao ela e para LAN isolada de laboratorio.
embed_fleet_key() {
    local source_key="$repo_root/deploy/keys/split-models-appliance"
    local target="$staged_profile/airootfs/root/.ssh/fleet_ed25519"
    if [[ ! -f "$source_key" ]]; then
        printf 'chave privada ausente em %s; gere em deploy/keys/ primeiro\n' \
            "$source_key" >&2
        exit 1
    fi
    install -d -m 0700 "$(dirname "$target")"
    install -m 0600 "$source_key" "$target"
}

# vim: set ft=sh:
