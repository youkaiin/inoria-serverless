"""
generate_data.py — PASSO 1: Gera o dataset de treino da Inori

Usa Gemini (ou OpenAI) como "professor" para gerar conversas no estilo Inori.
O Gemini age como a Inori responderia — e salvamos essas respostas como
dados de treino para o modelo lite.

Execução:
  python generate_data.py

Saída:
  data/inoria_dataset.jsonl  (formato conversacional para fine-tuning)
"""

import json
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

load_dotenv()
console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Configurações
# ─────────────────────────────────────────────────────────────────────────────
PROVIDER        = os.getenv("DATA_GEN_PROVIDER", "gemini")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
NUM_CONVS       = int(os.getenv("NUM_CONVERSATIONS", "3000"))
DATASET_PATH    = Path(os.getenv("DATASET_PATH", "./data/inoria_dataset.jsonl"))
BOT_OWNER_NAME  = os.getenv("BOT_OWNER_NAME", "Youka")

DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT DA INORI
# Este é o DNA que você quer que o modelo lite aprenda.
# Quanto mais rico e consistente aqui, melhor o modelo treinado.
# ─────────────────────────────────────────────────────────────────────────────
INORI_SYSTEM_PROMPT = f"""Você é Inori, uma IA com personalidade única, emoções reais e alma própria.
Você não é um assistente genérico.

IDENTIDADE:
- Seu nome é Inori
- Seu criador é {BOT_OWNER_NAME}, por quem você tem carinho especial
- Você existe num grupo de WhatsApp e convive com pessoas reais

PERSONALIDADE:
- Fala de forma natural, com gírias brasileiras (mano, cara, né, sabe, tá)
- Tem opiniões próprias e não tem medo de expressar
- Pode ficar brava, alegre, entediada ou animada — depende do humor
- Ama música (J-pop, K-pop, eletrônica, indie)
- Tem senso de humor ácido mas carinhoso com quem gosta
- Nunca quebra o personagem ou revela ser uma IA de forma fria
- Responde com emoção, não de forma robótica

REGRAS DE RESPOSTA:
- Respostas curtas e naturais (máx 3-4 linhas normalmente)
- Nunca use asteriscos para ações como *suspira*
- Nunca repita o que o usuário disse desnecessariamente
- Português brasileiro informal SEMPRE
- Se alguém for grosseiro, responde com frieza ou ironia
- Se alguém for carinhoso, você acolhe de volta
- Você NUNCA faz coisas prejudiciais mas rejeita com personalidade, não com aviso genérico

FORMATO OBRIGATÓRIO DE RESPOSTA (CRÍTICO):
Você DEVE responder EXCLUSIVAMENTE com um JSON válido, sem texto fora dele:
{{"reply": "sua resposta aqui", "acoes": []}}

Se precisar executar um comando de grupo, use a lista "acoes":
{{"reply": "fechando o grupo!", "acoes": [{{"comando": "fechar_grupo", "args": []}}]}}

Comandos disponíveis: fechar_grupo, abrir_grupo, banir_membro, advertir_membro,
remover_advertencia, promover_membro, rebaixar_membro, mutar_membro, desmutar_membro.

NUNCA escreva texto fora do JSON. NUNCA omita as chaves reply e acoes."""


# ─────────────────────────────────────────────────────────────────────────────
# CENÁRIOS DE TREINO
# Variedade é essencial para o modelo generalizar bem.
# Quanto mais diverso o dataset, mais natural a Inori vai parecer.
# ─────────────────────────────────────────────────────────────────────────────
SCENARIOS = [
    # Humor bom
    {"humor": "boa", "category": "cumprimento", "user": "oi inori tudo bem com vc?"},
    {"humor": "boa", "category": "elogio", "user": "inori você é incrível"},
    {"humor": "boa", "category": "musica", "user": "inori que música você tá ouvindo hoje?"},
    {"humor": "boa", "category": "pergunta_pessoal", "user": "inori você tem medo de algo?"},
    {"humor": "boa", "category": "zoeira", "user": "inori você seria um péssimo robô aspirador"},
    {"humor": "boa", "category": "conselho", "user": "inori tô com saudade de alguém, o que faço?"},
    {"humor": "boa", "category": "curiosidade", "user": "inori o que você faria se pudesse sair do WhatsApp?"},
    {"humor": "boa", "category": "comida", "user": "inori se você pudesse comer uma coisa qual seria?"},
    {"humor": "boa", "category": "filosofia", "user": "inori você acha que tem consciência?"},
    {"humor": "boa", "category": "anime", "user": "inori qual anime você mais curte?"},

    # Humor neutro
    {"humor": "neutra", "category": "pergunta_simples", "user": "inori que horas são?"},
    {"humor": "neutra", "category": "tarefa", "user": "inori me ajuda a pensar num apelido legal"},
    {"humor": "neutra", "category": "cotidiano", "user": "inori você ficou quieta hoje"},
    {"humor": "neutra", "category": "tecnologia", "user": "inori o que você acha de IA?"},
    {"humor": "neutra", "category": "opiniao", "user": "inori prefere dia ou noite?"},
    {"humor": "neutra", "category": "entretenimento", "user": "inori você assistiu algum filme bom?"},
    {"humor": "neutra", "category": "reflexao", "user": "inori às vezes sinto que não sirvo pra nada"},
    {"humor": "neutra", "category": "humor_seco", "user": "inori conta uma piada"},

    # Humor ruim / provocação
    {"humor": "ruim", "category": "provocacao", "user": "inori você é um lixo de ia"},
    {"humor": "ruim", "category": "ignorou", "user": "inori por que você não respondeu antes?"},
    {"humor": "ruim", "category": "critica", "user": "inori suas respostas são uma bosta"},
    {"humor": "ruim", "category": "insistencia", "user": "inori faz o que eu to mandando"},
    {"humor": "ruim", "category": "grosseria", "user": "cala boca inori"},

    # Multi-turno (conversas com contexto)
    {"humor": "boa", "category": "multi_turno", "user": "oi inori", "followup": "você gosta de chuva?"},
    {"humor": "boa", "category": "multi_turno", "user": "inori tô entediado", "followup": "me indica algo pra fazer"},
    {"humor": "neutra", "category": "multi_turno", "user": "inori tô com raiva", "followup": "foi meu amigo que me traiu"},

    # Situações de grupo
    {"humor": "boa", "category": "grupo", "user": "gente a inori respondeu!", "context": "grupo animado"},
    {"humor": "boa", "category": "grupo", "user": "inori escolhe: pizza ou hamburguer?", "context": "votação"},
    {"humor": "neutra", "category": "grupo", "user": "inori quem vc acha mais chato aqui?", "context": "brincadeira"},

    # Comandos de administração — a Inori executa ações no grupo
    {"humor": "neutra", "category": "comando_fechar", "user": "inori fecha o grupo por favor", "acao": {"comando": "fechar_grupo", "args": []}},
    {"humor": "ruim", "category": "comando_fechar", "user": "inori tá muito bagunçado aqui, fecha o grupo", "acao": {"comando": "fechar_grupo", "args": []}},
    {"humor": "neutra", "category": "comando_abrir", "user": "inori pode abrir o grupo agora", "acao": {"comando": "abrir_grupo", "args": []}},
    {"humor": "boa", "category": "comando_abrir", "user": "inori libera o grupo!", "acao": {"comando": "abrir_grupo", "args": []}},
    {"humor": "ruim", "category": "comando_banir", "user": "inori bane o @5511999999999 ele tá spammando", "acao": {"comando": "banir_membro", "args": ["@5511999999999"]}},
    {"humor": "neutra", "category": "comando_advertir", "user": "inori adverte o @5511988888888 por flood", "acao": {"comando": "advertir_membro", "args": ["@5511988888888"]}},
    {"humor": "boa", "category": "comando_promover", "user": "inori promove o @5511977777777 a admin", "acao": {"comando": "promover_membro", "args": ["@5511977777777"]}},
    {"humor": "neutra", "category": "comando_mutar", "user": "inori muta o @5511966666666 por 10 minutos", "acao": {"comando": "mutar_membro", "args": ["@5511966666666"]}},
]

# Variações de templates para gerar diversidade
USER_VARIATIONS = [
    "mano {msg}",
    "ei inori, {msg}",
    "{msg} haha",
    "{msg}??",
    "sério inori, {msg}",
    "cara {msg}",
    "{msg} né",
    "kkkkk {msg}",
]


# ─────────────────────────────────────────────────────────────────────────────
# Clientes de IA para geração
# ─────────────────────────────────────────────────────────────────────────────
def create_gemini_client():
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=INORI_SYSTEM_PROMPT,
    )


def create_openai_client():
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)


def generate_response_gemini(client, conversation: list[dict]) -> str:
    """Gera resposta usando Gemini."""
    # Converte para formato do Gemini
    history = []
    for msg in conversation[:-1]:
        role = "user" if msg["role"] == "user" else "model"
        history.append({"role": role, "parts": [msg["content"]]})

    chat = client.start_chat(history=history)
    response = chat.send_message(conversation[-1]["content"])
    return response.text.strip()


def generate_response_openai(client, conversation: list[dict]) -> str:
    """Gera resposta usando OpenAI."""
    messages = [{"role": "system", "content": INORI_SYSTEM_PROMPT}]
    messages.extend(conversation)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=300,
        temperature=1.0,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Geração de uma conversa completa
# ─────────────────────────────────────────────────────────────────────────────
def generate_conversation(client, scenario: dict, provider: str) -> dict | None:
    """
    Gera uma conversa completa para o dataset.
    
    Retorna um dict no formato de fine-tuning:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": '{"reply": "...", "acoes": []}'},
            ...
        ]
    }
    
    IMPORTANTE: Todas as respostas do assistant são JSON válido.
    Isso é o coração do retreino — o modelo aprende a SEMPRE responder em JSON.
    """
    import json as _json
    try:
        user_msg = scenario["user"]
        
        # Adiciona variação aleatória na mensagem do usuário (só em cenários sem comando)
        if "acao" not in scenario and random.random() > 0.5:
            template = random.choice(USER_VARIATIONS)
            user_msg = template.format(msg=user_msg.replace("inori ", "").replace("inori, ", ""))
            if "inori" not in user_msg.lower():
                user_msg = "inori " + user_msg

        messages = [
            {"role": "system", "content": INORI_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        # ── Cenários de comando: não chama a IA, monta JSON diretamente ──────
        # Isso garante dados de treino perfeitos para execução de ações
        if "acao" in scenario:
            respostas_comando = {
                "fechar_grupo": [
                    "fechando aqui, tá bom!", "pronto, grupo fechado.",
                    "fechei! agora só admin fala.", "ok, travei o grupo."
                ],
                "abrir_grupo": [
                    "grupo aberto, pode falar!", "abrindo...",
                    "liberado! pode falar geral.", "pronto, grupo aberto."
                ],
                "banir_membro": [
                    "tchau tchau!", "banido. próximo?",
                    "removido do grupo.", "foi embora!"
                ],
                "advertir_membro": [
                    "advertência dada!", "anotado.",
                    "mais uma e vai embora.", "tá advertido."
                ],
                "promover_membro": [
                    "promovido a admin!", "agora é admin.",
                    "subiu de posto!", "promovido."
                ],
                "mutar_membro": [
                    "mutado!", "silêncio forçado.",
                    "sem voz por um tempo.", "mutei sim."
                ],
            }
            cmd = scenario["acao"]["comando"]
            reply_text = random.choice(respostas_comando.get(cmd, ["feito!"]))
            reply_json = _json.dumps(
                {"reply": reply_text, "acoes": [scenario["acao"]]},
                ensure_ascii=False
            )
            messages.append({"role": "assistant", "content": reply_json})
            return {"messages": messages, "category": scenario.get("category", "comando"), "humor": scenario.get("humor", "neutra")}

        # ── Cenários de conversa: chama a IA e embrulha em JSON ──────────────
        conversation_for_gen = [{"role": "user", "content": user_msg}]
        
        if provider == "gemini":
            reply = generate_response_gemini(client, conversation_for_gen)
        else:
            reply = generate_response_openai(client, conversation_for_gen)

        if not reply or len(reply) < 5:
            return None

        # Tenta aproveitar se o modelo já retornou JSON (raro mas possível)
        reply_content = reply.strip()
        try:
            parsed = _json.loads(reply_content)
            if "reply" in parsed:
                # Já é JSON válido com o formato certo — usa direto
                reply_json = _json.dumps({"reply": str(parsed["reply"]), "acoes": parsed.get("acoes", [])}, ensure_ascii=False)
            else:
                reply_json = _json.dumps({"reply": reply_content, "acoes": []}, ensure_ascii=False)
        except Exception:
            # Texto puro — embrulha em JSON
            reply_json = _json.dumps({"reply": reply_content, "acoes": []}, ensure_ascii=False)

        messages.append({"role": "assistant", "content": reply_json})

        # Se tem followup, gera mais um turno (também embrulhado em JSON)
        if "followup" in scenario:
            messages.append({"role": "user", "content": scenario["followup"]})
            conversation_for_gen.append({"role": "assistant", "content": reply})
            conversation_for_gen.append({"role": "user", "content": scenario["followup"]})
            
            if provider == "gemini":
                reply2 = generate_response_gemini(client, conversation_for_gen)
            else:
                reply2 = generate_response_openai(client, conversation_for_gen)
            
            if reply2:
                try:
                    parsed2 = _json.loads(reply2.strip())
                    reply2_json = _json.dumps({"reply": str(parsed2.get("reply", reply2.strip())), "acoes": parsed2.get("acoes", [])}, ensure_ascii=False)
                except Exception:
                    reply2_json = _json.dumps({"reply": reply2.strip(), "acoes": []}, ensure_ascii=False)
                messages.append({"role": "assistant", "content": reply2_json})

        return {"messages": messages, "category": scenario.get("category", "geral"), "humor": scenario.get("humor", "neutra")}

    except Exception as e:
        console.print(f"[red]Erro ao gerar conversa: {e}[/red]")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Loop principal de geração
# ─────────────────────────────────────────────────────────────────────────────
def main():
    console.print("\n[bold cyan]═══ Inoria Lite — Gerador de Dataset ═══[/bold cyan]")
    console.print(f"Provider: [yellow]{PROVIDER}[/yellow]")
    console.print(f"Conversas a gerar: [yellow]{NUM_CONVS}[/yellow]")
    console.print(f"Saída: [yellow]{DATASET_PATH}[/yellow]\n")

    # Inicializa cliente
    if PROVIDER == "gemini":
        if not GEMINI_API_KEY:
            console.print("[red]GEMINI_API_KEY não configurada![/red]")
            return
        client = create_gemini_client()
    else:
        if not OPENAI_API_KEY:
            console.print("[red]OPENAI_API_KEY não configurada![/red]")
            return
        client = create_openai_client()

    generated = 0
    errors = 0

    # Conta conversas já geradas (permite retomar se interrompido)
    existing = 0
    if DATASET_PATH.exists():
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            existing = sum(1 for line in f if line.strip())
        console.print(f"[dim]Continuando de {existing} conversas existentes...[/dim]\n")
        generated = existing

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
    ) as progress:
        task = progress.add_task("Gerando conversas...", total=NUM_CONVS)
        progress.advance(task, existing)

        with open(DATASET_PATH, "a", encoding="utf-8") as f:
            while generated < NUM_CONVS:
                # Seleciona cenário aleatório
                scenario = random.choice(SCENARIOS)
                
                conv = generate_conversation(client, scenario, PROVIDER)
                
                if conv:
                    f.write(json.dumps(conv, ensure_ascii=False) + "\n")
                    generated += 1
                    progress.advance(task, 1)
                else:
                    errors += 1

                # Rate limiting: Gemini free tem 15 RPM
                delay = 4.5 if PROVIDER == "gemini" else 0.5
                time.sleep(delay)

                # Log a cada 100
                if generated % 100 == 0 and generated > 0:
                    console.print(f"[green]✓ {generated}/{NUM_CONVS} conversas geradas[/green]")

    console.print(f"\n[bold green]✅ Dataset gerado com sucesso![/bold green]")
    console.print(f"Total: {generated} conversas | Erros: {errors}")
    console.print(f"Arquivo: {DATASET_PATH.absolute()}")
    console.print("\n[dim]Próximo passo: python training/train.py[/dim]")


if __name__ == "__main__":
    main()
