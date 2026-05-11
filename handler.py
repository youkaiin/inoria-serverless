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
from transformers import AutoModelForCausalLM, AutoTokenizer

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

# Frases do system prompt que indicam vazamento (model fazendo text completion).
# Se a saída contém alguma dessas, o modelo não aprendeu o chat template.
_SYSTEM_PROMPT_LEAKS = [
    "respostas curtas", "nunca quebre", "nunca use asteriscos",
    "formato de resposta", "ferramentas disponíveis", "regras absolutas",
    "personalidade:", "estado atual:", "seu criador",
    "fale sempre em português", "você é inori",
    "você controla o sistema", "sistema de dicas",
]

# ─────────────────────────────────────────────────────────────────────────────
# Estado global — carregado uma única vez por worker
# ─────────────────────────────────────────────────────────────────────────────
_tokenizer = None
_model     = None
_whisper   = None


def ensure_model():
    """Baixa o modelo do HuggingFace se não existir ou estiver incompleto."""
    import json as _json
    model_dir = Path(MODEL_PATH)
    safetensors = model_dir / "model.safetensors"
    tokenizer_cfg = model_dir / "tokenizer_config.json"

    # Verifica se o modelo existe E se o tokenizer tem chat_template
    needs_download = not safetensors.exists()
    if not needs_download and tokenizer_cfg.exists():
        try:
            cfg = _json.load(open(tokenizer_cfg))
            if "chat_template" not in cfg:
                print("[Inoria] tokenizer_config.json sem chat_template — re-baixando modelo...")
                needs_download = True
        except Exception:
            needs_download = True

    if not needs_download:
        print(f"[Inoria] Modelo encontrado em {MODEL_PATH} ✅")
        return

    hf_repo = os.getenv("HF_REPO", "youka9987/inoria-model")
    hf_token = os.getenv("HF_TOKEN")
    print(f"[Inoria] Baixando {hf_repo} do HuggingFace...")
    from huggingface_hub import snapshot_download
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(hf_repo, local_dir=str(model_dir), token=hf_token)
    print("[Inoria] Download concluído ✅")


def load_all():
    """Carrega modelo + Whisper na inicialização do worker."""
    global _tokenizer, _model, _whisper

    ensure_model()
    print(f"[Inoria] Carregando tokenizer de {MODEL_PATH}...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    print(f"[Inoria] Carregando modelo em {dtype} no {device}...")

    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    _model.eval()

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
def _extract_outermost_json(text):
    """
    Extrai o JSON mais externo de uma string, respeitando chaves aninhadas.
    Resolve o bug do regex não-guloso que parava no primeiro '}' encontrado,
    quebrando o parse quando 'acoes' continha comandos (JSON aninhado).
    """
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def generate_reply(messages):
    import re
    device = next(_model.parameters()).device

    text = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = _tokenizer(text, return_tensors='pt').to(device)
    input_length = inputs['input_ids'].shape[1]

    with torch.no_grad():
        outputs_raw = _model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.45,
            top_p=0.88,
            top_k=40,
            repetition_penalty=1.25,
            pad_token_id=_tokenizer.eos_token_id,
            eos_token_id=_tokenizer.eos_token_id,
            do_sample=True,
        )

    # ── Fix 1: Recorte por string (anti-leak) ────────────────────────────────
    # O recorte por input_length depende do tokenizer contar certo. Se o treino
    # bagunçou as tags especiais, a contagem erra e o system prompt vaza.
    # Solução: decodificar TUDO e cortar pelo marcador de texto do Qwen2.5.
    # O Qwen sempre coloca "<|im_start|>assistant\n" antes de falar.
    # Pegamos só o que vem DEPOIS desse marcador — imune a erros de contagem.
    resposta_bruta_completa = _tokenizer.decode(outputs_raw[0], skip_special_tokens=False)
    MARCADOR_ASSISTANT = "<|im_start|>assistant"
    if MARCADOR_ASSISTANT in resposta_bruta_completa:
        raw = resposta_bruta_completa.split(MARCADOR_ASSISTANT)[-1]
        raw = raw.replace("<|im_end|>", "").strip()
        print(f'[Inoria] Recorte por string OK. Raw: {raw[:300]}')
    else:
        # Fallback: recorte por token count (método original)
        raw = _tokenizer.decode(outputs_raw[0][input_length:], skip_special_tokens=True).strip()
        print(f'[Inoria] Recorte por token count. Raw: {raw[:300]}')

    # ── Fix 2: Tenta extrair JSON ─────────────────────────────────────────────
    # Usa _extract_outermost_json (respeita aninhamento) em vez de regex não-guloso.
    json_str = _extract_outermost_json(raw)
    if json_str:
        try:
            data = _json.loads(json_str)
            reply_text = str(data.get('reply') or data.get('mensagem_texto') or '').strip()
            acoes = data.get('acoes', [])
            if not isinstance(acoes, list):
                acoes = []
            if reply_text:
                print(f'[Inoria] JSON válido extraído. Reply: {reply_text[:80]}')
                return {'reply': reply_text, 'acoes': acoes}
        except Exception as e:
            print(f'[Inoria] Falha ao parsear JSON: {e} | json_str={json_str[:100]}')

    # ── Fix 3: Extrai texto de tags LimaRP (<FIRST>texto</FIRST>) ────────────
    # O modelo foi treinado em roleplay e pode responder com tags LimaRP.
    # Extraímos o conteúdo e removemos asteriscos de ação (*sorri*, etc.).
    lirarp_match = re.search(
        r'<(?:FIRST|SECOND|THIRD|USER)>(.*?)</(?:FIRST|SECOND|THIRD|USER)>',
        raw, re.DOTALL
    )
    if lirarp_match:
        texto_extraido = re.sub(r'\*[^*]+\*', '', lirarp_match.group(1)).strip()
        texto_extraido = re.sub(r'\s+', ' ', texto_extraido).strip()
        if texto_extraido and len(texto_extraido) < 500:
            print(f'[Inoria] Texto de tag LimaRP: {texto_extraido[:100]}')
            return {'reply': texto_extraido, 'acoes': []}

    # ── Fix 4: Envelopamento inteligente — texto puro vira JSON ──────────────
    # Se a IA respondeu naturalmente (sem JSON, sem tags), pegamos o texto
    # direto e embalamos num JSON válido para o bot. Não desperdiçamos a resposta.
    # Antes de usar, filtra vazamento de system prompt.
    raw_lower = raw.lower()
    is_system_leak = any(leak in raw_lower for leak in _SYSTEM_PROMPT_LEAKS)
    if is_system_leak:
        print(f'[Inoria] Vazamento de system prompt detectado, descartando.')
        # Mesmo vazando, tenta pegar só a primeira frase antes das instruções
        primeira_frase = raw.split('\n')[0].strip()
        if primeira_frase and len(primeira_frase) < 200 and not any(
            leak in primeira_frase.lower() for leak in _SYSTEM_PROMPT_LEAKS
        ):
            return {'reply': primeira_frase, 'acoes': []}
        return {'reply': 'Oi! Tô aqui 😊', 'acoes': []}

    # Texto limpo — embala e entrega
    raw_limpo = re.sub(r'\*[^*]+\*', '', raw).strip()
    raw_limpo = re.sub(r'\s+', ' ', raw_limpo).strip()
    if raw_limpo and len(raw_limpo) < 500:
        print(f'[Inoria] Envelopamento de texto puro: {raw_limpo[:80]}')
        return {'reply': raw_limpo, 'acoes': []}

    return {'reply': 'Oi! Pode repetir?', 'acoes': []}


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
        import json as _j
        tok_cfg = Path(MODEL_PATH) / "tokenizer_config.json"
        has_tmpl = False
        if tok_cfg.exists():
            try:
                has_tmpl = "chat_template" in _j.load(open(tok_cfg))
            except Exception:
                pass
        return {
            "status": "online",
            "model": MODEL_PATH,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "chat_template_ok": has_tmpl,
        }

    # ── PATCH TOKENIZER — sobrescreve tokenizer_config.json do HF ────────
    elif action == "patch_tokenizer":
        import json as _j
        from huggingface_hub import hf_hub_download as _hf_dl
        hf_repo  = os.getenv("HF_REPO", "youka9987/inoria-model")
        hf_token = os.getenv("HF_TOKEN")
        try:
            src = _hf_dl(hf_repo, "tokenizer_config.json", token=hf_token, force_download=True)
            cfg = _j.load(open(src))
            dest = Path(MODEL_PATH) / "tokenizer_config.json"
            _j.dump(cfg, open(dest, "w"), ensure_ascii=False, indent=2)
            has_tmpl = "chat_template" in cfg
            return {"patched": True, "chat_template_ok": has_tmpl, "dest": str(dest)}
        except Exception as e:
            return {"patched": False, "error": str(e)}

    # ── DEBUG (temporário — remove após diagnóstico) ───────────────────────
    elif action == "debug_raw":
        # Retorna o texto bruto gerado pelo modelo sem nenhum processamento.
        # Usar para diagnosticar o formato real de saída do modelo.
        import re as _re
        message       = inp.get("message", "Bom dia!")
        user_name     = inp.get("user_name", "Youka")
        humor_level   = int(inp.get("humor_level", 75))
        afinity_level = int(inp.get("afinity_level", 90))
        extra_context = inp.get("extra_context", "")
        tips_enabled  = bool(inp.get("tips_enabled", False))

        system_prompt = build_system_prompt(user_name, humor_level, afinity_level, extra_context, tips_enabled)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": message},
        ]
        device = next(_model.parameters()).device
        text = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = _tokenizer(text, return_tensors='pt').to(device)
        input_length = inputs['input_ids'].shape[1]
        with torch.no_grad():
            outputs_raw = _model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.45,
                top_p=0.88,
                top_k=40,
                repetition_penalty=1.25,
                pad_token_id=_tokenizer.eos_token_id,
                eos_token_id=_tokenizer.eos_token_id,
                do_sample=True,
            )
        raw = _tokenizer.decode(outputs_raw[0][input_length:], skip_special_tokens=True).strip()
        json_found = _extract_outermost_json(raw)
        return {
            "raw_output":      raw,
            "raw_length":      len(raw),
            "input_tokens":    input_length,
            "json_found":      json_found,
            "has_lirarp_tags": bool(_re.search(r'<(FIRST|SECOND|USER|THIRD)>', raw)),
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
