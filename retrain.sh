#!/usr/bin/env bash
# retrain.sh — Retreina a Inoria com o dataset expandido no RunPod
# Cole e execute no terminal do pod interativo RunPod:
#   bash retrain.sh
set -e

echo "══════════════════════════════════════════"
echo "  Inoria — Retreino (dataset expandido)"
echo "══════════════════════════════════════════"

# ── 1. Atualiza o repo ────────────────────────────────────────────
cd /workspace/inoria-serverless 2>/dev/null || {
    echo "[1/6] Clonando repo..."
    git clone https://github.com/youkaiin/inoria-serverless.git /workspace/inoria-serverless
    cd /workspace/inoria-serverless
}

echo "[1/6] Atualizando repo..."
git pull origin main

# ── 2. Instala dependências ──────────────────────────────────────
echo "[2/6] Instalando dependências..."
pip install -q -r requirements-runpod.txt

# ── 3. Gera o dataset ────────────────────────────────────────────
echo "[3/6] Gerando dataset (228 exemplos)..."
python prepare_dataset.py

# ── 4. Treino ────────────────────────────────────────────────────
echo "[4/6] Iniciando treino QLoRA..."
NUM_EPOCHS=6 BATCH_SIZE=4 GRADIENT_ACCUMULATION=4 python training/train.py

# ── 5. Merge dos adapters ────────────────────────────────────────
echo "[5/6] Fazendo merge dos adapters LoRA..."
python - <<'PYEOF'
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE   = "Qwen/Qwen2.5-1.5B-Instruct"
LORA   = "./output/inoria-model"
MERGED = "./inori-brain/merged-model"

print(f"  Carregando base: {BASE}")
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

print(f"  Carregando adapter: {LORA}")
model = PeftModel.from_pretrained(model, LORA)

print("  Fazendo merge...")
model = model.merge_and_unload()

print(f"  Salvando em {MERGED}...")
Path(MERGED).mkdir(parents=True, exist_ok=True)
model.save_pretrained(MERGED)
tok.save_pretrained(MERGED)
print("  ✅ Merge concluído!")
PYEOF

# ── 6. Upload para HuggingFace ───────────────────────────────────
echo "[6/6] Fazendo upload para HuggingFace..."
python - <<'PYEOF'
import os
from huggingface_hub import upload_folder

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Defina a variável HF_TOKEN antes de rodar: export HF_TOKEN=hf_...")
upload_folder(
    folder_path="./inori-brain/merged-model",
    repo_id="youka9987/inoria-model",
    token=HF_TOKEN,
    commit_message="retrain: dataset 228 exemplos, fix generation params",
)
print("✅ Upload concluído! https://huggingface.co/youka9987/inoria-model")
PYEOF

echo ""
echo "══════════════════════════════════════════"
echo "  ✅ Retreino finalizado com sucesso!"
echo "  Agora reinicie os workers do RunPod:"
echo "  endpoint bp1i4d3kuan8hi → pare e inicie"
echo "══════════════════════════════════════════"
