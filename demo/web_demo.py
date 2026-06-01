#!/usr/bin/env python
"""Data-analysis oriented Qwen web demo with optional LoRA adapter loading."""

from __future__ import annotations

import argparse
from pathlib import Path


SYSTEM_PROMPT = """You are a Data Agent for data analysis and mathematical modeling.
When the user asks a data question, respond with: problem abstraction, analysis plan,
code or SQL sketch, validation checks, and final answer format. Ask for missing data
only when it is necessary."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="models/Qwen3.5-0.8B")
    parser.add_argument("--adapter-path", "-a", default="outputs/checkpoints/qwen35_08b_data_agent_lora/final")
    parser.add_argument("--checkpoint-path", "-c", default=None, help="Backward-compatible alias for --adapter-path.")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main() -> None:
    try:
        import gradio as gr
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
        from threading import Thread
    except ImportError as exc:
        raise SystemExit(
            "Missing demo dependencies. Install them first:\n"
            "  conda activate agent\n"
            "  pip install gradio torch transformers accelerate peft\n"
            f"Original import error: {exc}"
        )

    args = parse_args()
    adapter_path = args.checkpoint_path or args.adapter_path
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.float16 if not args.cpu_only else torch.float32,
        device_map="cpu" if args.cpu_only else "auto",
    )
    if adapter_path and Path(adapter_path).exists():
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    def stream_chat(message, history):
        conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
        for user, assistant in history:
            conversation.append({"role": "user", "content": user})
            conversation.append({"role": "assistant", "content": assistant})
        conversation.append({"role": "user", "content": message})
        text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        thread = Thread(target=model.generate, kwargs={**inputs, "streamer": streamer, "max_new_tokens": 1536})
        thread.start()
        output = ""
        for piece in streamer:
            output += piece
            yield output

    examples = [
        "Given a sales table with date, region, sku, price and quantity, design an end-to-end pipeline to forecast next month's revenue.",
        "I have customer churn data. Abstract the ML problem, choose metrics, and write a baseline modeling plan.",
        "Write SQL to compute month-over-month retention and explain how to validate the result.",
    ]

    demo = gr.ChatInterface(
        fn=stream_chat,
        title="Data Agent Demo",
        description="Ask data analysis, SQL, statistics, and modeling questions.",
        examples=examples,
    )
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
