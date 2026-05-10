# ─────────────────────────────────────────────────────────────────────────────
# Inoria Serverless — Dockerfile para RunPod
# Usa python:3.11-slim como base (leve ~200MB).
# PyTorch com CUDA é instalado via pip no build.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# Dependências do sistema para compilar libs nativas
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Instala PyTorch com CUDA 12.4 primeiro (layer separada para cache)
RUN pip install --no-cache-dir torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Copia e instala o resto das dependências
COPY requirements-runpod.txt .
RUN pip install --no-cache-dir -r requirements-runpod.txt

# Pré-baixa o Whisper small (~461MB) durante o build
RUN python -c "import whisper; whisper.load_model('small')"

# Copia apenas o handler
COPY handler.py .

# Variáveis de ambiente (sobrescreva no painel do RunPod)
ENV MODEL_PATH="/runpod-volume/inoria-model"
ENV WHISPER_MODEL="small"
ENV BOT_OWNER_NAME="Youka"
ENV MAX_NEW_TOKENS="256"
ENV TEMPERATURE="0.85"
ENV TOP_P="0.92"
ENV REPETITION_PEN="1.1"
ENV PYTHONUNBUFFERED="1"

CMD ["python", "-u", "handler.py"]
