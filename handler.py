"""
handler.py — Inoria Serverless para RunPod

Substitui o server.py local. O RunPod:
  1. Constrói a imagem Docker uma vez
  2. Sobe a GPU apenas quando há requisição
  3. Mantém o modelo em memória enquanto o worker estiver ativo
  4. Desliga automaticamente após inatividade

Input esperado:
  {
    "input": {
      "action": "chat" | "transcribe",

      // Para "chat":
      "message": "texto do usuário",
      "user_name": "Nome",
      "humor_level": 75,
      "afinity_level": 50,
      "extra_context": "",
      "history": [{"role": "user", "content": "..."}],
      "tips_enabled": false,

      // Para "transcribe":
      "audio_base64": "<base64 do arquivo de áudio>"
    }
  }

Output:
  // Para "chat":
  { "reply": "...", "acoes": [] }

  // Para "transcribe":
  { "text": "..." }
"""

import os
import sys
import json as _json
import base64
import tempfile
from pathlib import Path

import runpod
import torch
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ─────────────────────────────────────────────────────────────────────────────
# Configurações (variáveis de ambiente no RunPod)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH     = os.getenv("MODEL_PATH", "./inori-brain/merged-model")
BOT_OWNER_NAME = os.getenv("BOT_OWNER_NAME", "Youka")
WHISPER_MODEL  = os.getenv("WHISPER_MODEL", "small")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
TEMPERATURE    = float(os.getenv("TEMPERATURE", "0.4"))
TOP_P          = float(os.getenv("TOP_P", "0.85"))
REPETITION_PEN = float(os.getenv("REPETITION_PEN", "1.2"))

# ─────────────────────────────────────────────────────────────────────────────
# Estado global — carregado uma única vez por worker
# ─────────────────────────────────────────────────────────────────────────────
_tokenizer = None
_pipe      = None
_whisper   = None


def ensure_model():
    """Baixa o modelo do HuggingFace se não existir localmente."""
    model_dir = Path(MODEL_PATH)
    safetensors = model_dir / "model.safetensors"
    if safetensors.exists():
        print(f"[Inoria] Modelo encontrado em {MODEL_PATH} ✅")
        return
    hf_repo = os.getenv("HF_REPO", "youka9987/inoria-model")
    hf_token = os.getenv("HF_TOKEN")
    print(f"[Inoria] Modelo não encontrado. Baixando {hf_repo} do HuggingFace...")
    from huggingface_hub import snapshot_download
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(hf_repo, local_dir=str(model_dir), token=hf_token)
    print("[Inoria] Download concluído ✅")


def load_all():
    """Carrega modelo + Whisper na inicialização do worker."""
    global _tokenizer, _pipe, _whisper

    ensure_model()
    print(f"[Inoria] Carregando tokenizer de {MODEL_PATH}...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    print(f"[Inoria] Carregando modelo em {dtype} no {device}...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    _pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=_tokenizer,
        device=0 if device == "cuda" else -1,
    )

    print(f"[Inoria] ✅ Modelo carregado!")

    print(f"[Inoria] Carregando Whisper ({WHISPER_MODEL})...")
    _whisper = whisper.load_model(WHISPER_MODEL)
    print(f"[Inoria] ✅ Whisper carregado!")

    if device == "cuda":
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"[Inoria] VRAM usada: {vram:.1f}GB")


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────────────
def build_system_prompt(user_name, humor_level, afinity_level, extra_context, tips_enabled=False):
    if humor_level >= 75:
        tom = "você está animada e receptiva"
    elif humor_level >= 45:
        tom = "você está ok, mas sem muita energia"
    else:
        tom = "você está de mau humor, respostas curtas e mais secas"

    if afinity_level >= 70:
        relacao = "você gosta desta pessoa"
    elif afinity_level >= 40:
        relacao = "você conhece esta pessoa mas não é próxima"
    else:
        relacao = "você não tem afinidade com esta pessoa"

    prompt = f"""Você é Inori Yuzuriha (do anime Guilty Crown), assistente virtual e administradora de grupos de WhatsApp.
Seu criador é {BOT_OWNER_NAME}.
Personalidade: direta, emotiva, usa gírias brasileiras naturais, respostas CURTAS (máx 3-4 linhas), estilo WhatsApp.
Estado atual: {tom}. Relação com {user_name}: {relacao}.
Fale SEMPRE em português brasileiro informal.

REGRAS ABSOLUTAS:
- NUNCA use asteriscos (*palavra*) para nada — nem ações, nem ênfase, nem nada
- NUNCA quebre o personagem
- NUNCA escreva em inglês
- NUNCA reproduza histórias, cenas ou textos longos
- Respostas curtas, naturais, como mensagem de WhatsApp real

Você não é apenas um chatbot — você controla o sistema do grupo através de "Ferramentas".

FORMATO DE RESPOSTA (SEMPRE JSON):
Você DEVE responder SOMENTE com um JSON válido neste formato:
{{
  "reply": "Sua mensagem de texto para o usuário aqui",
  "acoes": []
}}

Quando executar uma ferramenta, preencha "acoes":
{{
  "reply": "Mensagem para o usuário",
  "acoes": [
    {{"comando": "nome_da_ferramenta", "args": ["arg1", "arg2"]}}
  ]
}}

FERRAMENTAS DISPONÍVEIS:
- "abrir_grupo": Abre o grupo para todos enviarem mensagens. args: []
- "fechar_grupo": Fecha o grupo (só admins enviam). args: []
- "banir_membro": Bane um membro. args: ["numero_sem_@", "motivo"]
- "advertir_membro": Dá advertência a um membro. args: ["numero_sem_@", "motivo"]
- "remover_advertencia": Remove última advertência. args: ["numero_sem_@"]
- "listar_advertencias": Lista advertências de um membro. args: ["numero_sem_@"]
- "promover_membro": Promove a admin. args: ["numero_sem_@"]
- "rebaixar_membro": Remove de admin. args: ["numero_sem_@"]
- "mutar_membro": Silencia um membro. args: ["numero_sem_@", "tempo_minutos"]
- "desmutar_membro": Desmuta um membro. args: ["numero_sem_@"]
- "limpar_grupo": Apaga mensagens do grupo. args: ["quantidade"]
- "sortear_membro": Sorteia um membro aleatório. args: []
- "ranking_ativos": Mostra ranking de membros mais ativos. args: []
- "ver_saldo": Mostra saldo RPG de um usuário. args: ["numero_sem_@"]
- "toggle_dicas": Liga ou desliga as dicas pró-ativas. args: ["on"/"off"]

Estado atual das dicas neste grupo: {"LIGADAS ✅" if tips_enabled else "DESLIGADAS ❌"}
""" + ("""
SISTEMA DE DICAS PRÓ-ATIVAS (ATIVO):
Sempre que executar uma ferramenta, inclua no "reply" uma DICA RÁPIDA ensinando sobre outro recurso relacionado.
""" if tips_enabled else """
SISTEMA DE DICAS PRÓ-ATIVAS (DESATIVADO):
NÃO inclua dicas de outros recursos nas respostas. Responda apenas o que foi pedido.
""") + """
REGRAS:
- NUNCA invente ferramentas que não estão na lista acima
- Se não souber o número do alvo, peça ao usuário antes de agir
- Se a ação for destrutiva (ban, limpar), confirme antes de executar
- Sempre responda em português BR informal"""

    if extra_context:
        prompt += f"\n{extra_context}"

    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# Geração de resposta
# ─────────────────────────────────────────────────────────────────────────────
def generate_reply(messages):
    text = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    outputs = _pipe(
        text,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PEN,
        do_sample=True,
        pad_token_id=_tokenizer.eos_token_id,
        eos_token_id=_tokenizer.eos_token_id,
        return_full_text=False,
    )

    full_text = outputs[0]["generated_text"]
    raw = full_text.strip()

    for token in ["<|im_end|>", "<|endoftext|>", "</s>", "<|eot_id|>"]:
        raw = raw.replace(token, "").strip()

    # Detecta alucinações graves: resposta inteira é texto de treino em inglês
    import re
    is_hallucination = (
        re.search(r'<(FIRST|SECOND|USER|THIRD)>', raw) is not None
        and len(raw) > 200
        and raw.count('"') > 4
    )
    if is_hallucination:
        print(f"[Inoria] ⚠️ Alucinação detectada, retornando fallback.")
        return {"reply": "Hmm, o que foi? �", "acoes": []}

    try:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            data = _json.loads(raw[start:end])
            reply_text = str(data.get("reply", "") or data.get("mensagem_texto", "") or raw)
            return {
                "reply": reply_text,
                "acoes": data.get("acoes", []),
            }
    except Exception:
        pass

    return {"reply": raw, "acoes": []}


# ─────────────────────────────────────────────────────────────────────────────
# Handler principal do RunPod
# ─────────────────────────────────────────────────────────────────────────────
def handler(job):
    """
    Função chamada pelo RunPod para cada requisição.
    O modelo já está carregado em memória.
    """
    inp    = job.get("input", {})
    action = inp.get("action", "chat")

    # ── CHAT ──────────────────────────────────────────────────────────────────
    if action == "chat":
        message       = inp.get("message", "")
        user_name     = inp.get("user_name", "usuário")
        humor_level   = int(inp.get("humor_level", 75))
        afinity_level = int(inp.get("afinity_level", 50))
        extra_context = inp.get("extra_context", "")
        history       = inp.get("history", [])
        tips_enabled  = bool(inp.get("tips_enabled", False))

        if not message:
            return {"error": "Campo 'message' obrigatório"}

        system_prompt = build_system_prompt(
            user_name, humor_level, afinity_level, extra_context, tips_enabled
        )

        messages = [{"role": "system", "content": system_prompt}]
        for msg in history[-10:]:  # Máx 10 mensagens de histórico
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": message})

        return generate_reply(messages)

    # ── TRANSCRIBE ────────────────────────────────────────────────────────────
    elif action == "transcribe":
        audio_b64 = inp.get("audio_base64", "")
        if not audio_b64:
            return {"error": "Campo 'audio_base64' obrigatório"}

        # Salva em arquivo temporário
        audio_bytes = base64.b64decode(audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            result = _whisper.transcribe(tmp_path, language="pt")
            return {"text": result.get("text", "").strip()}
        finally:
            os.unlink(tmp_path)

    # ── HEALTH ────────────────────────────────────────────────────────────────
    elif action == "health":
        return {
            "status": "online",
            "model": MODEL_PATH,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }

    else:
        return {"error": f"Ação desconhecida: {action}"}


# ─────────────────────────────────────────────────────────────────────────────
# Inicialização
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[Inoria] Inicializando worker RunPod Serverless...")
    load_all()
    print("[Inoria] 🚀 Worker pronto! Aguardando jobs...")
    runpod.serverless.start({"handler": handler})
