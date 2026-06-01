#!/usr/bin/env python
"""Prepare selected DataMind-12K trajectories for Qwen SFT."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_agent.selection import dump_jsonl, load_json_records, select_records, to_qwen_example


DEFAULT_URL = "https://huggingface.co/datasets/zjunlp/DataMind-12K/resolve/main/datamind_12k.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/datamind_12k.json"))
    parser.add_argument("--download", action="store_true", help="Download the JSON file when --input is absent.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/data"))
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.92)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        if not args.download:
            raise SystemExit(f"{args.input} does not exist. Re-run with --download or pass --input.")
        args.input.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {args.url} -> {args.input}")
        urllib.request.urlretrieve(args.url, args.input)

    records = load_json_records(args.input)
    train, val, stats = select_records(
        records,
        train_size=args.train_size,
        val_size=args.val_size,
        near_duplicate_threshold=args.near_duplicate_threshold,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(args.out_dir / "train_qwen.jsonl", (to_qwen_example(x) for x in train))
    dump_jsonl(args.out_dir / "val_qwen.jsonl", (to_qwen_example(x) for x in val))
    (args.out_dir / "selection_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

