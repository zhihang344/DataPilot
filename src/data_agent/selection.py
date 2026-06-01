"""Trajectory selection utilities for DataMind-style agent data.

The implementation is dependency-light on purpose: it can process the 12K
records with only Python's standard library inside a fresh conda environment.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


ROLE_MAP = {
    "human": "user",
    "question": "user",
    "gpt": "assistant",
    "bot": "assistant",
    "model": "assistant",
    "tool_response": "tool",
}

ANALYTIC_KEYWORDS = {
    "sql",
    "select",
    "join",
    "group by",
    "order by",
    "where",
    "python",
    "pandas",
    "sklearn",
    "regression",
    "classification",
    "correlation",
    "hypothesis",
    "confidence interval",
    "p-value",
    "optimize",
    "forecast",
    "anomaly",
    "feature",
    "result.csv",
}

BAD_PATTERNS = {
    "i cannot",
    "i can't",
    "not enough information",
    "as an ai language model",
    "sorry",
    "traceback",
    "syntaxerror",
}


@dataclass(order=True)
class ScoredRecord:
    sort_key: tuple[float, float, str] = field(init=False, repr=False)
    score: float
    quality: float
    complexity: float
    diversity: float
    record_id: str
    source: str
    db_id: str
    messages: list[dict[str, str]]
    raw: dict[str, Any]

    def __post_init__(self) -> None:
        self.sort_key = (self.score, self.quality, self.record_id)


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON/JSONL dataset and return a flat list of dict records."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    data = json.loads(text)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("train", "data", "records", "examples"):
            value = data.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        records: list[dict[str, Any]] = []
        for value in data.values():
            if isinstance(value, list):
                records.extend(x for x in value if isinstance(x, dict))
        if records:
            return records
    raise ValueError(f"Unsupported dataset shape in {path}")


def dump_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    """Extract a Qwen-style messages list from a DataMind record."""
    candidates = [record.get("trajectory"), record.get("messages"), record.get("prompt")]
    for candidate in candidates:
        messages = _coerce_messages(candidate)
        if messages:
            return messages

    extra = record.get("extra_info") if isinstance(record.get("extra_info"), dict) else {}
    question = record.get("question") or extra.get("question") or record.get("instruction")
    answer = record.get("answer") or record.get("response") or _ground_truth(record)
    messages = []
    if question:
        messages.append({"role": "user", "content": str(question)})
    if answer:
        messages.append({"role": "assistant", "content": str(answer)})
    return messages


def to_qwen_example(scored: ScoredRecord) -> dict[str, Any]:
    return {
        "id": scored.record_id,
        "messages": scored.messages,
        "metadata": {
            "source": scored.source,
            "db_id": scored.db_id,
            "score": round(scored.score, 6),
            "quality": round(scored.quality, 6),
            "complexity": round(scored.complexity, 6),
            "diversity": round(scored.diversity, 6),
            "task_id": scored.raw.get("task_id") or scored.raw.get("id"),
        },
    }


def score_records(records: list[dict[str, Any]]) -> list[ScoredRecord]:
    source_counts = Counter(str(r.get("data_source") or r.get("source") or "unknown") for r in records)
    db_counts = Counter(str(r.get("db_id") or _extra(r).get("db_id") or "unknown") for r in records)
    scored: list[ScoredRecord] = []

    for idx, record in enumerate(records):
        messages = normalize_messages(record)
        if len(messages) < 2:
            continue
        text = messages_text(messages)
        if len(text) < 80:
            continue

        source = str(record.get("data_source") or record.get("source") or "unknown")
        db_id = str(record.get("db_id") or _extra(record).get("db_id") or "unknown")
        record_id = str(record.get("task_id") or record.get("id") or _extra(record).get("index") or idx)

        complexity = complexity_score(messages)
        quality = quality_score(record, messages)
        diversity = diversity_score(source_counts[source], db_counts[db_id], len(records))
        score = 0.45 * quality + 0.35 * complexity + 0.20 * diversity
        scored.append(
            ScoredRecord(
                score=score,
                quality=quality,
                complexity=complexity,
                diversity=diversity,
                record_id=record_id,
                source=source,
                db_id=db_id,
                messages=messages,
                raw=record,
            )
        )
    return scored


def select_records(
    records: list[dict[str, Any]],
    train_size: int = 2000,
    val_size: int = 500,
    near_duplicate_threshold: float = 0.92,
) -> tuple[list[ScoredRecord], list[ScoredRecord], dict[str, Any]]:
    """Select high-value train/validation records without random sampling."""
    target = train_size + val_size
    scored = sorted(score_records(records), reverse=True)
    deduped = deduplicate(scored, near_duplicate_threshold=near_duplicate_threshold)
    chosen = balanced_take(deduped, target)

    val: list[ScoredRecord] = []
    train: list[ScoredRecord] = []
    used_train_dbs: set[str] = set()
    for item in chosen:
        if len(val) < val_size and item.db_id not in used_train_dbs:
            val.append(item)
        else:
            train.append(item)
            used_train_dbs.add(item.db_id)

    if len(val) < val_size:
        need = val_size - len(val)
        val.extend(train[-need:])
        train = train[:-need]
    if len(train) < train_size:
        train.extend(x for x in deduped if x not in train and x not in val)
        train = train[:train_size]

    stats = {
        "input_records": len(records),
        "scored_records": len(scored),
        "deduplicated_records": len(deduped),
        "train_records": len(train[:train_size]),
        "validation_records": len(val[:val_size]),
        "avg_train_score": _avg(x.score for x in train[:train_size]),
        "avg_validation_score": _avg(x.score for x in val[:val_size]),
        "selection_policy": "quality_reward_heuristic + complexity_filter + trajectory_dedup + db/source_balancing",
    }
    return train[:train_size], val[:val_size], stats


def deduplicate(scored: list[ScoredRecord], near_duplicate_threshold: float = 0.92) -> list[ScoredRecord]:
    exact_seen: set[str] = set()
    buckets: dict[str, list[str]] = defaultdict(list)
    result: list[ScoredRecord] = []
    for item in scored:
        prompt = user_prompt(item.messages)
        exact = stable_hash(normalize_text(prompt))
        if exact in exact_seen:
            continue
        exact_seen.add(exact)

        bucket_key = stable_hash(" ".join(normalize_text(prompt).split()[:12]))[:8]
        candidates = buckets[bucket_key]
        if any(SequenceMatcher(None, normalize_text(prompt), old).ratio() >= near_duplicate_threshold for old in candidates):
            continue
        candidates.append(normalize_text(prompt))
        result.append(item)
    return result


def balanced_take(scored: list[ScoredRecord], target: int) -> list[ScoredRecord]:
    groups: dict[tuple[str, str], list[ScoredRecord]] = defaultdict(list)
    for item in scored:
        groups[(item.source, item.db_id)].append(item)

    chosen: list[ScoredRecord] = []
    group_lists = sorted(groups.values(), key=lambda xs: xs[0].score, reverse=True)
    while len(chosen) < target and group_lists:
        next_groups = []
        for group in group_lists:
            if group and len(chosen) < target:
                chosen.append(group.pop(0))
            if group:
                next_groups.append(group)
        group_lists = sorted(next_groups, key=lambda xs: xs[0].score, reverse=True)
    return chosen


def complexity_score(messages: list[dict[str, str]]) -> float:
    text = messages_text(messages).lower()
    token_count = len(text.split())
    turns = len(messages)
    keyword_hits = sum(1 for kw in ANALYTIC_KEYWORDS if kw in text)
    code_blocks = text.count("```")
    numeric_density = len(re.findall(r"\b\d+(?:\.\d+)?\b", text)) / max(1, token_count)
    value = (
        0.30 * min(1.0, math.log1p(token_count) / math.log(1800))
        + 0.25 * min(1.0, turns / 12)
        + 0.25 * min(1.0, keyword_hits / 8)
        + 0.10 * min(1.0, code_blocks / 3)
        + 0.10 * min(1.0, numeric_density * 20)
    )
    return clamp(value)


def quality_score(record: dict[str, Any], messages: list[dict[str, str]]) -> float:
    text = messages_text(messages).lower()
    assistant_text = "\n".join(m["content"] for m in messages if m["role"] == "assistant")
    reward = numeric_reward(record.get("reward_model"))
    reward_part = 0.25 + 0.35 * reward if reward is not None else 0.40
    answer_part = 0.20 if len(assistant_text.split()) > 80 else 0.10
    gt_part = 0.15 if _ground_truth(record) else 0.0
    completion_part = 0.10 if ("result.csv" in text or "final answer" in text or "saved" in text) else 0.04
    penalty = 0.08 * sum(1 for pattern in BAD_PATTERNS if pattern in text)
    return clamp(reward_part + answer_part + gt_part + completion_part - penalty)


def diversity_score(source_count: int, db_count: int, total: int) -> float:
    source_rarity = 1.0 - math.log1p(source_count) / math.log1p(total)
    db_rarity = 1.0 - math.log1p(db_count) / math.log1p(total)
    return clamp(0.35 + 0.30 * source_rarity + 0.35 * db_rarity)


def numeric_reward(value: Any) -> float | None:
    numbers: list[float] = []

    def walk(x: Any) -> None:
        if isinstance(x, bool):
            numbers.append(1.0 if x else 0.0)
        elif isinstance(x, (int, float)):
            numbers.append(float(x))
        elif isinstance(x, dict):
            for item in x.values():
                walk(item)
        elif isinstance(x, list):
            for item in x:
                walk(item)

    walk(value)
    if not numbers:
        return None
    avg = sum(numbers) / len(numbers)
    if avg > 1.0:
        avg = avg / 100.0 if avg <= 100.0 else 1.0
    return clamp(avg)


def messages_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in messages)


def user_prompt(messages: list[dict[str, str]]) -> str:
    users = [m["content"] for m in messages if m["role"] == "user"]
    return users[-1] if users else messages_text(messages)[:1000]


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9_\u4e00-\u9fff ]+", "", text)
    return text.strip()


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _avg(values: Iterable[float]) -> float:
    values = list(values)
    return round(sum(values) / len(values), 6) if values else 0.0


def _coerce_messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    messages: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, dict):
            role = str(item.get("role") or item.get("from") or item.get("speaker") or "").lower()
            content = item.get("content") or item.get("value") or item.get("text")
            if content is None:
                continue
            role = ROLE_MAP.get(role, role)
            if role not in {"system", "user", "assistant", "tool"}:
                role = "assistant" if messages and messages[-1]["role"] == "user" else "user"
            messages.append({"role": role, "content": str(content).strip()})
        elif isinstance(item, str) and item.strip():
            role = "assistant" if messages and messages[-1]["role"] == "user" else "user"
            messages.append({"role": role, "content": item.strip()})
    return [m for m in messages if m["content"]]


def _extra(record: dict[str, Any]) -> dict[str, Any]:
    return record.get("extra_info") if isinstance(record.get("extra_info"), dict) else {}


def _ground_truth(record: dict[str, Any]) -> str:
    reward = record.get("reward_model")
    if isinstance(reward, dict):
        gt = reward.get("ground_truth")
        if isinstance(gt, dict):
            return str(gt.get("ground_truth") or gt.get("answer") or "")
        if gt:
            return str(gt)
    extra = _extra(record)
    return str(extra.get("ground_truth") or "")

