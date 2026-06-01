#!/usr/bin/env python
"""LoRA SFT entry point for Qwen3.5-0.8B on Qwen-style chat JSONL.

This script is designed for a single consumer GPU. It freezes the base model and
trains only LoRA adapter weights, which greatly reduces VRAM usage compared with
full-parameter fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen3.5-0.8B", help="Base Qwen3.5-0.8B model path or model id.")
    parser.add_argument("--train-file", type=Path, default=Path("outputs/data/train_qwen.jsonl"))
    parser.add_argument("--val-file", type=Path, default=Path("outputs/data/val_qwen.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/checkpoints/qwen35_08b_data_agent_lora"))
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--fp16", action="store_true", help="Use float16 weights on CUDA.")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 weights on CUDA when supported.")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names for LoRA injection.",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from torch.utils.data import DataLoader
        from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:
        raise SystemExit(
            "Missing LoRA training dependencies. Install them first:\n"
            "  conda activate agent\n"
            "  pip install torch transformers accelerate peft\n"
            f"Original import error: {exc}"
        )

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is False.")

    if device == "cuda" and args.bf16 and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"Loading base model={args.model} device={device} dtype={dtype}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=dtype).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.gradient_checkpointing and hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()
        base_model.config.use_cache = False
    if hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[x.strip() for x in args.lora_target_modules.split(",") if x.strip()],
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    model.train()

    train_records = load_jsonl(args.train_file)
    val_records = load_jsonl(args.val_file) if args.val_file.exists() else []

    def encode(example: dict) -> dict[str, torch.Tensor]:
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        encoded = tokenizer(text, max_length=args.max_length, truncation=True, padding="max_length", return_tensors="pt")
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    train_limit = max(args.max_steps * args.batch_size * args.gradient_accumulation_steps, 1)
    dataset = [encode(x) for x in train_records[:train_limit]]
    val_dataset = [encode(x) for x in val_records[: min(len(val_records), 64)]]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) if val_dataset else None

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    warmup_steps = max(1, int(args.max_steps * args.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=args.max_steps)

    def move_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(device) for key, value in batch.items()}

    @torch.no_grad()
    def evaluate() -> float:
        if val_loader is None:
            return math.nan
        model.eval()
        losses = []
        for batch in val_loader:
            outputs = model(**move_batch(batch))
            losses.append(outputs.loss.item())
        model.train()
        return sum(losses) / len(losses) if losses else math.nan

    step = 0
    micro_step = 0
    best_val_loss = math.inf
    optimizer.zero_grad(set_to_none=True)
    for batch in loader:
        outputs = model(**move_batch(batch))
        loss = outputs.loss / args.gradient_accumulation_steps
        loss.backward()
        micro_step += 1
        if micro_step % args.gradient_accumulation_steps != 0:
            continue

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        train_loss = loss.item() * args.gradient_accumulation_steps
        print(f"step={step} train_loss={train_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}")

        if args.eval_steps > 0 and step % args.eval_steps == 0:
            val_loss = evaluate()
            print(f"step={step} val_loss={val_loss:.4f}")
            if not math.isnan(val_loss) and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_dir = args.output_dir / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                print(f"Saved best LoRA adapter to {best_dir}")

        if args.save_steps > 0 and step % args.save_steps == 0:
            step_dir = args.output_dir / f"step_{step}"
            step_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(step_dir)
            tokenizer.save_pretrained(step_dir)
            print(f"Saved LoRA adapter to {step_dir}")

        if step >= args.max_steps:
            break

    final_dir = args.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved final LoRA adapter to {final_dir}")


if __name__ == "__main__":
    main()
