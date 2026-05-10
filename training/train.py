"""
train.py — Fine-tuning QLoRA da Inoria (TRL 1.4+)

Treina Qwen2.5-1.5B-Instruct com QLoRA no dataset curado da Inoria.
Usa SFTConfig.assistant_only_loss=True (nativo TRL 1.4) para
calcular loss apenas nas respostas do assistant — sem DataCollator manual.

Execução:
  python training/train.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from rich.console import Console

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Configurações
# ─────────────────────────────────────────────────────────────────────────────
BASE_MODEL     = os.getenv("BASE_MODEL",             "Qwen/Qwen2.5-1.5B-Instruct")
DATASET_PATH   = os.getenv("DATASET_PATH",           "./data/inoria_dataset.jsonl")
OUTPUT_DIR     = os.getenv("OUTPUT_DIR",             "./output/inoria-model")
LORA_RANK      = int(os.getenv("LORA_RANK",          "32"))
LORA_ALPHA     = int(os.getenv("LORA_ALPHA",         "64"))
LORA_DROPOUT   = float(os.getenv("LORA_DROPOUT",     "0.05"))
LEARNING_RATE  = float(os.getenv("LEARNING_RATE",    "2e-4"))
NUM_EPOCHS     = int(os.getenv("NUM_EPOCHS",         "4"))
BATCH_SIZE     = int(os.getenv("BATCH_SIZE",         "4"))
GRAD_ACCUM     = int(os.getenv("GRADIENT_ACCUMULATION", "4"))
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH",     "2048"))


def main():
    console.print("\n[bold cyan]═══ Inoria Lite — Fine-tuning QLoRA (TRL 1.4) ═══[/bold cyan]")
    console.print(f"Modelo base  : [yellow]{BASE_MODEL}[/yellow]")
    console.print(f"Dataset      : [yellow]{DATASET_PATH}[/yellow]")
    console.print(f"Saída        : [yellow]{OUTPUT_DIR}[/yellow]")
    if torch.cuda.is_available():
        console.print(f"GPU          : [yellow]{torch.cuda.get_device_name(0)}[/yellow]")
        console.print(f"VRAM         : [yellow]{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB[/yellow]\n")

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    console.print("[bold]Carregando dataset...[/bold]")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    console.print(f"  Treino: {len(dataset['train'])} | Validação: {len(dataset['test'])}")

    # ── 2. Tokenizer ─────────────────────────────────────────────────────────
    console.print("\n[bold]Carregando tokenizer...[/bold]")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Verificação rápida do dataset
    sample = tokenizer.apply_chat_template(
        dataset["train"][0]["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    assert "<|im_start|>" in sample, "ERRO: ChatML não encontrado!"
    console.print("[green]✓ ChatML detectado no dataset[/green]")

    # ── 3. Modelo em 4-bit (QLoRA) ────────────────────────────────────────────
    console.print("\n[bold]Carregando modelo em 4-bit (QLoRA)...[/bold]")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )

    # ── 4. LoRA ───────────────────────────────────────────────────────────────
    console.print("\n[bold]Configurando LoRA adapters...[/bold]")
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

    # ── 5. SFTConfig (TRL 1.4 — tudo centralizado aqui) ──────────────────────
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        # Parâmetros SFT nativos do TRL 1.4
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
        # ✅ Loss só nas respostas do assistant — nativo TRL 1.4
        # Substitui DataCollatorForCompletionOnlyLM sem gambiarras
        assistant_only_loss=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        run_name="inoria-lite",
    )

    # ── 6. Trainer ────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=lora_config,
    )

    trainer.model.print_trainable_parameters()

    # ── 7. Treina ─────────────────────────────────────────────────────────────
    console.print("\n[bold green]Iniciando treino...[/bold green]")
    console.print("[dim]RTX 4090 + 1.5B + 89 exemplos → ~5-10 minutos[/dim]\n")
    trainer.train()

    # ── 8. Salva ──────────────────────────────────────────────────────────────
    console.print("\n[bold]Salvando modelo final...[/bold]")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    console.print(f"\n[bold green]✅ Treino completo![/bold green]")
    console.print(f"Modelo salvo em: [yellow]{Path(OUTPUT_DIR).absolute()}[/yellow]")


def _find_target_modules(model) -> list:
    common = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    found = []
    for name, _ in model.named_modules():
        part = name.split(".")[-1]
        if part in common and part not in found:
            found.append(part)
    return found or ["q_proj", "v_proj"]


if __name__ == "__main__":
    main()
