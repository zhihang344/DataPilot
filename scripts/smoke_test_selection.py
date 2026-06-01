#!/usr/bin/env python
"""Tiny smoke test for the selection pipeline."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_agent.selection import load_json_records, select_records, to_qwen_example


def main() -> None:
    records = []
    for i in range(12):
        records.append(
            {
                "data_source": "darl/sql",
                "db_id": f"db_{i % 4}",
                "task_id": f"task_{i}",
                "trajectory": [
                    {"role": "system", "content": "You are a data analyst."},
                    {"role": "user", "content": f"Write SQL and validate revenue trend for segment {i}."},
                    {
                        "role": "assistant",
                        "content": "Plan: inspect schema, aggregate monthly revenue, compute growth, save result.csv. "
                        "SQL: SELECT month, SUM(revenue) FROM sales GROUP BY month ORDER BY month. "
                        "Validation: compare row counts and null rates. Final answer saved to result.csv.",
                    },
                ],
                "reward_model": {"score": 0.9},
            }
        )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.json"
        path.write_text(json.dumps(records), encoding="utf-8")
        loaded = load_json_records(path)
        train, val, stats = select_records(loaded, train_size=8, val_size=2)
        assert len(train) == 8, len(train)
        assert len(val) == 2, len(val)
        example = to_qwen_example(train[0])
        assert "messages" in example and example["messages"]
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

