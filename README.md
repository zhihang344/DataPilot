# Project 2: Data Agent System and Startup Plan

This repository contains a complete scaffold for Project 2:

- DataMind-12K trajectory selection and Qwen SFT data preparation.
- GPU Qwen3.5-0.8B LoRA SFT training entry point.
- Data-analysis oriented Qwen web demo with CSV/Excel profiling and LoRA adapter loading.
- Report, architecture diagram, and presentation source files.

## Environment

Use the requested conda environment:

```bash
conda activate agent
pip install -r requirements.txt
```

The data preparation script only needs Python standard library, so it can run even before installing the ML stack.

## Q1 Data Preparation

Place `datamind_12k.json` at `data/datamind_12k.json`, then run:

```bash
conda run -n agent python scripts/prepare_datamind_qwen.py \
  --input data/datamind_12k.json \
  --out-dir outputs/data
```

If HuggingFace is reachable, `--download` can be used instead of manually placing the file.

Outputs:

- `outputs/data/train_qwen.jsonl`
- `outputs/data/val_qwen.jsonl`
- `outputs/data/selection_stats.json`

Each line follows Qwen chat SFT style:

```json
{"id": "...", "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "metadata": {...}}
```

## Q1 GPU LoRA Training

```bash
conda run -n agent python scripts/train_sft_gpu.py \
  --model models/Qwen3.5-0.8B \
  --train-file outputs/data/train_qwen.jsonl \
  --val-file outputs/data/val_qwen.jsonl \
  --max-steps 500 \
  --max-length 1024 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --fp16 \
  --lora-r 8 \
  --lora-alpha 16 \
  --output-dir outputs/checkpoints/qwen35_08b_data_agent_lora
```

If HuggingFace is not reachable, download the Qwen3.5-0.8B checkpoint manually and pass the local directory to `--model`. LoRA saves adapter weights only, so keep the base model directory for demo and inference.

## Q1 Demo

```bash
conda run -n agent python demo/web_demo.py \
  --base-model models/Qwen3.5-0.8B \
  --adapter-path outputs/checkpoints/qwen35_08b_data_agent_lora/final
```

Open the printed local URL, usually `http://127.0.0.1:8000`. The demo supports uploading CSV/Excel files, profiling columns and missing values with pandas, and injecting the dataset profile into the agent context before generating analysis plans, SQL/Python sketches, modeling choices, and validation checks.

## Documents

- Report draft: `docs/report.md`
- Architecture diagram: `docs/architecture.mmd`
- Roadshow slides source: `docs/presentation.md`

Optional conversion:

```bash
pandoc docs/report.md -o studentID_Name.pdf
```

