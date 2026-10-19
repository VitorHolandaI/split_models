# Backends Por Familia De Modelo

Este PoC nao deve fingir que todo modelo causal tem a mesma estrutura interna.
Cada familia precisa ser analisada por `config.json` e `model.safetensors.index.json`
antes de virar backend.

## Contrato Comum Do PoC

- Head tokeniza e orquestra controle.
- Workers recebem um intervalo contiguo de layers.
- Workers carregam pesos seletivamente via safetensors.
- Dados seguem `HEAD -> W1 -> W2 -> ...` como hidden state.
- O ultimo worker aplica final norm/lm head e manda token de volta.
- Controle (`READY`, `ERROR`, `STOP`) fica em canal separado.

## Backend Atual: Qwen2.5 Text

Status: funcional no PoC com `Qwen/Qwen2.5-0.5B`.

Estrutura usada:

- `model.embed_tokens.weight`
- `model.layers.N.*`
- `model.norm.weight`
- `lm_head.weight` ou tied com `model.embed_tokens.weight`
- `model.model.layers` no objeto Transformers criado por `AutoModelForCausalLM`

Detalhes de forward:

- calcula embeddings no primeiro worker;
- calcula RoPE uma vez por forward;
- passa `position_embeddings=(cos, sin)` para cada layer;
- usa mascara causal explicita;
- ultimo worker roda `norm` e `lm_head`.

## Qwen3.5 9B

Status: pesquisado, nao implementado.

Repo pesquisado: `Qwen/Qwen3.5-9B`.

Arquitetura:

- `architectures`: `Qwen3_5ForConditionalGeneration`
- `model_type`: `qwen3_5`
- multimodal: possui `text_config` e `vision_config`
- texto tem 32 layers;
- hidden size 4096;
- vocab 248320;
- context 262144;
- mistura `linear_attention` e `full_attention`;
- possui MTP (`mtp_num_hidden_layers: 1`).

Estrutura de pesos:

- `model.language_model.embed_tokens.weight`
- `model.language_model.layers.N.*`
- `model.language_model.norm.weight`
- `lm_head.weight`
- `model.visual.*`
- `mtp.*`

Implicacao:

- nao encaixa no backend Qwen2.5;
- precisa backend proprio para splitar apenas `model.language_model.*`;
- precisa decidir como ignorar visao e MTP no modo texto;
- precisa confirmar se Transformers permite instanciar so o language model sem alocar o visual.

## GPT-OSS

Status: pesquisado, nao implementado.

Repos pesquisados:

- `openai/gpt-oss-20b`
- `openai/gpt-oss-120b`

Arquitetura comum:

- `architectures`: `GptOssForCausalLM`
- `model_type`: `gpt_oss`
- text generation, nao multimodal;
- MoE com roteador;
- `experts_per_token`: 4;
- `num_experts_per_tok`: 4;
- attention alterna `sliding_attention` e `full_attention`;
- `sliding_window`: 128;
- RoPE YaRN em `rope_scaling`;
- quantizacao MXFP4 nos pesos MoE;
- formato de chat recomendado: Harmony.

`openai/gpt-oss-20b`:

- 24 layers;
- hidden size 2880;
- 32 experts locais;
- 21B parametros, 3.6B ativos segundo README;
- weight map sharded em `model-00000-of-00002.safetensors` e `model-00001-of-00002.safetensors`.

`openai/gpt-oss-120b`:

- 36 layers;
- hidden size 2880;
- 128 experts locais;
- 117B parametros, 5.1B ativos segundo README.

Estrutura de pesos observada no 20B:

- `model.embed_tokens.weight`
- `model.layers.N.input_layernorm.weight`
- `model.layers.N.self_attn.*`
- `model.layers.N.self_attn.sinks`
- `model.layers.N.mlp.router.*`
- `model.layers.N.mlp.experts.*_blocks`
- `model.layers.N.mlp.experts.*_scales`
- `model.norm.weight`
- `lm_head.weight`

Implicacao:

- o path de layers parece mais proximo do Qwen2.5 (`model.layers.N`), mas o forward nao e igual;
- precisa preservar sliding/full attention por layer;
- precisa suporte ao MoE quantizado MXFP4;
- precisa confirmar se `AutoModelForCausalLM.from_config()` instancia os kernels/quantizacao corretamente;
- Harmony/chat template precisa ser usado para respostas corretas, nao prompt cru.

## Regra Para Novos Modelos

Antes de implementar qualquer familia nova:

1. Ler `config.json`.
2. Ler `model.safetensors.index.json` quando existir.
3. Identificar prefixos reais dos pesos.
4. Identificar onde ficam embeddings, layers, norm e lm head.
5. Identificar se ha vision/audio/MTP/MoE/quantizacao especial.
6. Confirmar assinatura de forward dos layers no Transformers instalado.
7. So entao criar backend especifico.

Nao adicionar compatibilidade por chute de nome do modelo.
