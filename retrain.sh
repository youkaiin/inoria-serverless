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
pip install -q datasets trl peft transformers accelerate bitsandbytes rich python-dotenv huggingface_hub

# ── 3. Gera o dataset ────────────────────────────────────────────
echo "[3/6] Gerando dataset (228 exemplos)..."
python prepare_dataset.py

# ── 4. Treino ────────────────────────────────────────────────────
echo "[4/6] Iniciando treino QLoRA..."
NUM_EPOCHS=6 BATCH_SIZE=4 GRADIENT_ACCUMULATION=4 python training/train.py

# ── 5. Merge dos adapters ────────────────────────────────────────
echo "[5/6] Fazendo merge dos adapters LoRA..."
python - <<'PYEOF'
import torch, json
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE   = "Qwen/Qwen2.5-1.5B-Instruct"
LORA   = "./output/inoria-model"
MERGED = "./inori-brain/merged-model"

print(f"  Carregando base: {BASE}")
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)

# Guarda o chat_template ANTES do save (transformers 5.x perde o campo)
chat_template_backup = tok.chat_template

model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16, device_map="auto", trust_remote_code=True)

print(f"  Carregando adapter: {LORA}")
model = PeftModel.from_pretrained(model, LORA)

print("  Fazendo merge...")
model = model.merge_and_unload()

print(f"  Salvando em {MERGED}...")
Path(MERGED).mkdir(parents=True, exist_ok=True)
model.save_pretrained(MERGED)
tok.save_pretrained(MERGED)

# Restaura o chat_template no tokenizer_config.json salvo
tok_cfg_path = Path(MERGED) / "tokenizer_config.json"
tok_cfg = json.load(open(tok_cfg_path))
if "chat_template" not in tok_cfg or not tok_cfg["chat_template"]:
    print("  Restaurando chat_template no tokenizer_config.json...")
    tok_cfg["chat_template"] = chat_template_backup
    json.dump(tok_cfg, open(tok_cfg_path, "w"), ensure_ascii=False, indent=2)

print("  ✅ Merge concluído com chat_template preservado!")
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
