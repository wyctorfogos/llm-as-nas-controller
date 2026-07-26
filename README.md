# LLM as NAS Controller

Busca de Arquiteturas Neurais (NAS — *Neural Architecture Search*) para classificação multimodal de lesões de pele, usando **LLMs locais servidos via [Ollama](https://ollama.com/)** como controlador da busca. A cada passo, o LLM recebe o espaço de busca e o histórico de resultados, propõe uma nova configuração de arquitetura em JSON, o modelo é treinado e a *Balanced Accuracy* (BACC) obtida é devolvida ao LLM como recompensa para guiar as próximas propostas.

Projeto desenvolvido no mestrado (PPGI/UFES).

## Como funciona

1. Um prompt é montado com o **espaço de busca** e o **histórico** de configurações já avaliadas (modos `full`, `last_k` ou `top_k`).
2. O prompt é enviado ao Ollama (`POST /api/generate`, com `format: json` para modelos compatíveis como `qwen*` e `gpt-oss*`) — ver [request_to_llm.py](src/scripts/benchmark/utils/request_to_llm.py).
3. A resposta é filtrada (remoção de `<think>`, extração do primeiro JSON válido) e validada com **Pydantic** ([pydantic_llm_response_formats.py](src/scripts/benchmark/models/pydantic_llm_response_formats.py)). Configurações inválidas ou repetidas são descartadas.
4. A configuração válida instancia uma CNN multimodal dinâmica ([dynamicMultimodalmodel.py](src/scripts/benchmark/models/dynamicMultimodalmodel.py)), que é treinada com *early stopping* e avaliada em validação.
5. A BACC vira a recompensa registrada no histórico, e o ciclo se repete por `SEARCH_STEPS` passos. Tudo é logado no **MLflow** e em CSV/JSON na pasta de resultados.

## Espaço de busca

| Hiperparâmetro | Valores |
| --- | --- |
| `num_blocks` | 2, 5, 10 |
| `initial_filters` | 16, 32, 64 |
| `kernel_size` | 3, 5 |
| `layers_per_block` | 1, 2 |
| `use_pooling` | true, false |
| `common_dim` | 64, 128, 256, 512 |
| `attention_mechanism` | `no-metadata`, `concatenation`, `crossattention`, `metablock` |
| `num_layers_text_fc` | 1, 2, 3 |
| `neurons_per_layer_size_of_text_fc` | 64, 128, 256, 512 |
| `num_layers_fc_module` | 1, 2 |
| `neurons_per_layer_size_of_fc_module` | 256, 512 |

## Estrutura do projeto

```text
conf/
  .env                  # variáveis de ambiente (criar a partir do .env.test)
src/scripts/
  benchmark/
    nas/                # scripts de busca e treino final
      optimization_train_process_pad_20_llm-as-controller.py   # NAS com LLM como controlador
      optimization_train_process_pad_20_using_random-search.py # baseline: busca aleatória
      optimization_train_process_pad_20.py                     # busca exaustiva/grid
      train_pad_20_optimized_model.py                          # treino final (PAD-UFES-20)
      train_isic_2019_optimized_model.py                       # treino final (ISIC-2019)
      train_milk10k_optimized_model.py                         # treino final (MILK-10k)
      calculate_flops.py                                       # FLOPs/parâmetros dos modelos
      run_script_via_bash.sh                                   # lança a busca em background
    models/             # datasets, CNN dinâmica, mecanismos de atenção (MetaBlock, MetaNet, cross-attention), focal loss, schemas Pydantic
    utils/              # cliente Ollama, filtragem de resposta do LLM, métricas, early stopping, logs de experimentos
    interpretability/   # Grad-CAM, Grad-CAM++, Score-CAM, flip rate, análise de incerteza
    plots/              # gráficos de resultados, matrizes de confusão, GIFs
  data_preprocessing/   # pré-processamento (ISIC-2019, PAD-UFES-20), data augmentation, LIME
  aggreation/           # agregação de métricas e testes estatísticos (Wilcoxon)
```

## Requisitos e instalação

Crie o ambiente virtual e instale as dependências:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision pydantic mlflow scikit-learn pandas numpy requests python-dotenv tqdm Pillow
```

> Recomenda-se GPU com CUDA para o treino dos modelos.

Além disso, é necessário o **Ollama** rodando localmente em `http://localhost:11434`, com o modelo controlador baixado:

```bash
ollama pull qwen3:0.6b   # ou outro modelo (qwen*, gpt-oss* suportam format=json e thinking)
```

- Dataset (ex.: [PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1)) com a estrutura:

```text
<DATASET_FOLDER_PATH>/
  metadata.csv
  images/
```

## Configuração

Crie `conf/.env` (use [conf/.env.test](conf/.env.test) como modelo):

```env
NUM_EPOCHS=100
BATCH_SIZE=32
K_FOLDS=5
LIST_NUM_HEADS=[8]
COMMON_DIM=512
DATASET_FOLDER_NAME="PAD-UFES-20"
DATASET_FOLDER_PATH="/caminho/para/PAD-UFES-20"
RESULTS_FOLDER_PATH="./src/results"
UNFREEZE_WEIGHTS=False
LLM_MODEL_NAME_SEQUENCE_GENERATOR="qwen3:0.6b"   # modelo do Ollama usado como controlador
HISTORY_MODE="full"                              # full | last_k | top_k
SEARCH_STEPS=500                                 # número de passos da busca
```

## Execução

Com o ambiente virtual ativado, rode a busca NAS com o LLM como controlador via script `.sh` (o processo é lançado em background, com log em `logs/`):

```bash
bash ./src/scripts/benchmark/nas/run_script_via_bash.sh
```

Ou rode o script Python diretamente:

```bash
python3 ./src/scripts/benchmark/nas/optimization_train_process_pad_20_llm-as-controller.py
```

Baseline com busca aleatória, para comparação:

```bash
python3 ./src/scripts/benchmark/nas/optimization_train_process_pad_20_using_random-search.py
```

Treino final da melhor arquitetura encontrada:

```bash
python3 ./src/scripts/benchmark/nas/train_pad_20_optimized_model.py
```

## Resultados

Os resultados são gravados em `RESULTS_FOLDER_PATH/<HISTORY_MODE>/<thinking>/controller-<llm>/<dataset>/...`, incluindo histórico da busca (JSON/CSV), melhor configuração encontrada e métricas por passo. Os experimentos também são rastreados no MLflow (`mlflow ui` para visualizar), com parâmetros como `search_space`, `history_mode` e `final_best_reward`.

# Citação

Este trabalho é faz parte de um artigo de nome "LLM-Driven Neural Architecture Search for Multimodal Skin Lesion Classification under Deployment Constraints" atualmente submetido para conferência.

Caso use o código em questão, fazer a devida citação do trabalho/artigo.