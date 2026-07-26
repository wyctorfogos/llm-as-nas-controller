#!/bin/bash
set -e

# ============================================================
# Ambiente
# ============================================================
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Carrega o .env
set -a
source ./conf/.env
set +a

# ============================================================
# Logs
# ============================================================
LOG_DIR=logs
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Sanitiza nome do modelo para nome de arquivo
SAFE_LLM_NAME=$(echo "$LLM_MODEL_NAME_SEQUENCE_GENERATOR" | sed 's/[:\/]/_/g')
HISTORY_MODE="full" # "full" # "top_k"

#LOG_FILE="$LOG_DIR/nas_25042026_${TIMESTAMP}.log"
LOG_FILE="$LOG_DIR/nas_20052026_${TIMESTAMP}_ABLATION-COT-IMPACT.log"

echo "📄 Log: $LOG_FILE"
echo "🚀 Iniciando NAS..."

# ============================================================
# Execução
# ============================================================
nohup python3 -u ./src/scripts/benchmark/nas/optimization_train_process_pad_20_llm-as-controller.py \
  > "$LOG_FILE" 2>&1 &

echo "✅ Processo iniciado em background (PID $!)"
