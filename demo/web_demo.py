#!/usr/bin/env python
"""Data-analysis oriented Qwen web demo with CSV profiling and optional LoRA loading."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are a Data Agent for data analysis and mathematical modeling.
When the user asks a data question, use the uploaded dataset profile when available.
Respond with: problem abstraction, data understanding, analysis plan, SQL or Python
sketch, modeling choices when relevant, validation checks, and final answer format.
If the data profile is insufficient, ask for the minimum missing information."""


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


def read_table(file_obj: Any):
    import pandas as pd

    if file_obj is None:
        return None, "No file uploaded."
    path = Path(file_obj.name if hasattr(file_obj, "name") else str(file_obj))
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, nrows=50000)
    else:
        df = pd.read_csv(path, nrows=50000)
    return df, f"Loaded `{path.name}` with up to 50,000 rows for profiling."


def profile_dataframe(df) -> str:
    import pandas as pd

    lines: list[str] = []
    lines.append(f"Rows: {len(df):,}")
    lines.append(f"Columns: {len(df.columns):,}")
    lines.append("")
    lines.append("Columns and types:")
    for col in df.columns:
        missing = int(df[col].isna().sum())
        missing_rate = missing / max(1, len(df))
        unique = int(df[col].nunique(dropna=True))
        lines.append(f"- {col}: dtype={df[col].dtype}, missing={missing} ({missing_rate:.1%}), unique={unique}")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        lines.append("")
        lines.append("Numeric summary:")
        desc = df[numeric_cols].describe().T[["mean", "std", "min", "50%", "max"]]
        for col, row in desc.iterrows():
            lines.append(
                f"- {col}: mean={row['mean']:.4g}, std={row['std']:.4g}, "
                f"min={row['min']:.4g}, median={row['50%']:.4g}, max={row['max']:.4g}"
            )

    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    if categorical_cols:
        lines.append("")
        lines.append("Categorical/date-like top values:")
        for col in categorical_cols[:12]:
            vc = df[col].astype(str).value_counts(dropna=True).head(5)
            top = "; ".join(f"{idx}: {val}" for idx, val in vc.items())
            lines.append(f"- {col}: {top}")

    lines.append("")
    lines.append("Suggested tool actions:")
    lines.append("- Validate missing values and duplicated rows before modeling.")
    lines.append("- Identify target variable, time column, entity key, and leakage-prone columns.")
    lines.append("- Use SQL/Python aggregation for EDA, then choose a baseline statistical or ML model.")
    return "\n".join(lines)


def main() -> None:
    try:
        import gradio as gr
        import pandas as pd
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
        from threading import Thread
    except ImportError as exc:
        raise SystemExit(
            "Missing demo dependencies. Install them first:\n"
            "  conda activate agent\n"
            "  pip install gradio pandas torch transformers accelerate peft\n"
            f"Original import error: {exc}"
        )

    args = parse_args()
    adapter_path = args.checkpoint_path or args.adapter_path
    print(f"[demo] loading tokenizer from {args.base_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    print(f"[demo] loading base model from {args.base_model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.float16 if not args.cpu_only else torch.float32,
        device_map="cpu" if args.cpu_only else "auto",
    )
    if adapter_path and Path(adapter_path).exists():
        print(f"[demo] loading LoRA adapter from {adapter_path}", flush=True)
        model = PeftModel.from_pretrained(model, adapter_path)
    elif adapter_path:
        print(f"[demo] LoRA adapter not found at {adapter_path}; using base model only", flush=True)
    model.eval()
    print("[demo] model ready", flush=True)

    def profile_upload(file_obj):
        try:
            df, status = read_table(file_obj)
            if df is None:
                return pd.DataFrame(), status, ""
            profile = profile_dataframe(df)
            preview = df.head(20)
            return preview, f"{status}\n\n```text\n{profile}\n```", profile
        except Exception as exc:
            return pd.DataFrame(), f"Failed to profile file: {exc}", ""

    def respond(message, history, dataset_profile):
        history = history or []
        profile_block = ""
        if dataset_profile:
            profile_block = "\n\nUploaded dataset profile:\n" + str(dataset_profile)[:6000]
        conversation = [{"role": "system", "content": SYSTEM_PROMPT + profile_block}]
        for item in history:
            if isinstance(item, dict) and item.get("role") in {"user", "assistant"}:
                conversation.append({"role": item["role"], "content": item.get("content", "")})
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                conversation.append({"role": "user", "content": item[0] or ""})
                conversation.append({"role": "assistant", "content": item[1] or ""})
        conversation.append({"role": "user", "content": message})

        text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        thread = Thread(target=model.generate, kwargs={**inputs, "streamer": streamer, "max_new_tokens": 1536})
        thread.start()

        updated = history + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
        for piece in streamer:
            updated[-1]["content"] += piece
            yield updated, ""

    with gr.Blocks(title="Data Agent Demo") as demo:
        gr.Markdown("# Data Agent Demo")
        with gr.Row():
            with gr.Column(scale=4):
                upload = gr.File(label="Upload CSV / Excel", file_types=[".csv", ".xlsx", ".xls"])
                profile_button = gr.Button("Profile Dataset", variant="primary")
                preview = gr.Dataframe(label="Data Preview", interactive=False)
                profile_md = gr.Markdown(label="Dataset Profile")
            with gr.Column(scale=5):
                chatbot = gr.Chatbot(label="Data Agent", height=520)
                message = gr.Textbox(
                    label="Ask a data-analysis question",
                    placeholder="Example: Find revenue drop drivers and design a validation plan.",
                    lines=3,
                )
                with gr.Row():
                    send = gr.Button("Run Analysis", variant="primary")
                    clear = gr.Button("Clear")
        dataset_state = gr.State("")
        profile_button.click(profile_upload, inputs=[upload], outputs=[preview, profile_md, dataset_state])
        send.click(respond, inputs=[message, chatbot, dataset_state], outputs=[chatbot, message])
        message.submit(respond, inputs=[message, chatbot, dataset_state], outputs=[chatbot, message])
        clear.click(lambda: ([], ""), outputs=[chatbot, message])

    print(f"[demo] launching Gradio on {args.server_name}:{args.server_port}", flush=True)
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
