"""
train.py — PASSO 2: Fine-tuning QLoRA da Inori

Treina um modelo pequeno (1.5B-7B) com o dataset gerado pelo generate_data.py.
Usa QLoRA: 4-bit quantization + LoRA adapters = treina em GPU de 6-8GB.

O resultado é um modelo salvo em ./output/inoria-model/
que pode ser carregado pelo servidor.

Execução:
  python training/train.py

Requisitos de GPU (mínimo):
  Qwen2.5-1.5B → 6GB VRAM
  Qwen2.5-3B   → 10GB VRAM
  Qwen2.5-7B   → 18GB VRAM
"""

import os
import sys
from pathlib import Path

# Garante que imports relativos funcionem
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
try:
    from trl import DataCollatorForCompletionOnlyLM
except ImportError:
    from trl.trainer import DataCollatorForCompletionOnlyLM
from rich.console import Console

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Configurações (lidas do .env)
# ─────────────────────────────────────────────────────────────────────────────
BASE_MODEL      = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DATASET_PATH    = os.getenv("DATASET_PATH", "./data/inoria_dataset.jsonl")
OUTPUT_DIR      = os.getenv("OUTPUT_DIR", "./output/inoria-model")
LORA_RANK       = int(os.getenv("LORA_RANK", "32"))
LORA_ALPHA      = int(os.getenv("LORA_ALPHA", "64"))
LORA_DROPOUT    = float(os.getenv("LORA_DROPOUT", "0.05"))
LEARNING_RATE   = float(os.getenv("LEARNING_RATE", "2e-4"))
NUM_EPOCHS      = int(os.getenv("NUM_EPOCHS", "4"))
BATCH_SIZE      = int(os.getenv("BATCH_SIZE", "4"))
GRAD_ACCUM      = int(os.getenv("GRADIENT_ACCUMULATION", "4"))
MAX_SEQ_LENGTH  = int(os.getenv("MAX_SEQ_LENGTH", "2048"))


def main():
    console.print("\n[bold cyan]═══ Inoria Lite — Fine-tuning QLoRA ═══[/bold cyan]")
    console.print(f"Modelo base: [yellow]{BASE_MODEL}[/yellow]")
    console.print(f"Dataset: [yellow]{DATASET_PATH}[/yellow]")
    console.print(f"Saída: [yellow]{OUTPUT_DIR}[/yellow]")
    console.print(f"GPU: [yellow]{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (não recomendado)'}[/yellow]")
    console.print(f"VRAM disponível: [yellow]{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB[/yellow]\n")

    # ── 1. Carrega e prepara o dataset ────────────────────────────────────────
    console.print("[bold]Carregando dataset...[/bold]")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    
    # Divide em treino e validação (90/10)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    console.print(f"  Treino: {len(dataset['train'])} | Validação: {len(dataset['test'])}")

    # ── 2. Tokenizer ─────────────────────────────────────────────────────────
    console.print("\n[bold]Carregando tokenizer...[/bold]")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 3. Modelo com quantização 4-bit (QLoRA) ───────────────────────────────
    console.print("\n[bold]Carregando modelo em 4-bit (QLoRA)...[/bold]")
    
    # Configuração da quantização — reduz VRAM ~4x
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NormalFloat4 — melhor qualidade
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,     # quantiza os pesos de quantização tbm
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",                  # distribui automático entre GPU/CPU
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Prepara o modelo para treino com gradientes em 4-bit
    model = prepare_model_for_kbit_training(model)

    # ── 4. Configuração LoRA ──────────────────────────────────────────────────
    console.print("\n[bold]Configurando LoRA adapters...[/bold]")
    
    # Detecta os módulos de atenção automaticamente
    target_modules = _find_target_modules(model)
    console.print(f"  Target modules: {target_modules}")

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Mostra % de parâmetros treináveis

    # ── 5. Formata dataset para o chat template do modelo ─────────────────────
    console.print("\n[bold]Formatando dataset com ChatML (apply_chat_template)...[/bold]")
    console.print("[dim]Isso tatua o ChatML no modelo — corrige a amnésia de formatação.[/dim]")

    def format_conversation(example):
        """
        Aplica o chat template NATIVO do Qwen2.5 ao dataset.
        
        Isso é o coração do retreino: forçamos o modelo a ver as conversas
        no formato ChatML correto (<|im_start|>...<|im_end|>) em vez de
        texto bruto ou tags LimaRP (<FIRST>/<SECOND>).
        
        add_generation_prompt=False é ESSENCIAL durante o treino —
        o modelo precisa ver a resposta completa, não apenas o prompt de geração.
        """
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    dataset = dataset.map(format_conversation, remove_columns=dataset["train"].column_names)

    # ── 5b. Verificação de sanidade ───────────────────────────────────────────
    # Confirma que o ChatML está presente e o JSON está nas respostas
    sample = dataset["train"][0]["text"]
    console.print(f"\n[dim]Exemplo de dado formatado:[/dim]")
    console.print(f"[dim]{sample[:400]}...[/dim]\n")
    assert "<|im_start|>" in sample, "ERRO: ChatML não encontrado no dataset!"
    assert "<|im_start|>assistant" in sample, "ERRO: marcador de assistant não encontrado!"
    console.print("[green]✓ ChatML detectado corretamente no dataset[/green]")

    # ── 5c. DataCollator — só treina na resposta do assistant ─────────────────
    # Sem isso, o modelo aprende a "prever" perguntas do usuário também,
    # desperdiçando capacidade. Com isso, só o token da resposta gera loss.
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # ── 6. Argumentos de treino ───────────────────────────────────────────────
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        
        # Otimizações de memória
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,        # Troca velocidade por memória
        optim="paged_adamw_32bit",          # Otimizador paginado (menos VRAM)
        
        # Logging e avaliação
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,                 # Mantém só os 2 melhores checkpoints
        load_best_model_at_end=True,
        
        # Relatórios
        report_to="none",                   # Muda para "wandb" se quiser tracking
        run_name="inoria-lite",
    )

    # ── 7. Trainer ────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        data_collator=collator,   # <-- só treina no output do assistant
        packing=False,            # packing=False é obrigatório com DataCollator
    )

    # ── 8. Treina ─────────────────────────────────────────────────────────────
    console.print("\n[bold green]Iniciando treino...[/bold green]")
    console.print("[dim]Isso pode levar de 1h a 8h dependendo da GPU e tamanho do dataset.[/dim]\n")

    trainer.train()

    # ── 9. Salva o modelo final ───────────────────────────────────────────────
    console.print("\n[bold]Salvando modelo final...[/bold]")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    console.print(f"\n[bold green]✅ Treino completo![/bold green]")
    console.print(f"Modelo salvo em: [yellow]{Path(OUTPUT_DIR).absolute()}[/yellow]")
    console.print("\n[dim]Próximo passo: python server/server.py[/dim]")


def _find_target_modules(model) -> list[str]:
    """
    Detecta automaticamente os módulos de atenção para aplicar LoRA.
    Funciona com Qwen2.5, Phi-3, LLaMA, Mistral, etc.
    """
    # Nomes comuns de módulos de atenção em diferentes arquiteturas
    common_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    found = []
    for name, module in model.named_modules():
        module_name = name.split(".")[-1]
        if module_name in common_targets and module_name not in found:
            found.append(module_name)
    
    return found if found else ["q_proj", "v_proj"]  # fallback


if __name__ == "__main__":
    main()
