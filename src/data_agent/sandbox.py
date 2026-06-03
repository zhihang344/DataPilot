"""Lightweight local Python sandbox for the Data Agent demo.

This is a teaching/demo sandbox, not a hardened container. It runs user-reviewed
code in a separate subprocess with timeout, static checks, isolated working
directory, and fixed DATA_PATH / OUTPUT_DIR variables.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


MAX_CAPTURE_CHARS = 12000
DEFAULT_TIMEOUT_SECONDS = 20
RUNS_ROOT = Path("outputs/sandbox_runs")

FORBIDDEN_PATTERNS = [
    (re.compile(r"\bimport\s+os\b"), "import os is disabled in the demo sandbox."),
    (re.compile(r"\bfrom\s+os\s+import\b"), "os module imports are disabled in the demo sandbox."),
    (re.compile(r"\bimport\s+subprocess\b"), "subprocess is disabled in the demo sandbox."),
    (re.compile(r"\bfrom\s+subprocess\s+import\b"), "subprocess is disabled in the demo sandbox."),
    (re.compile(r"\bimport\s+socket\b"), "network/socket access is disabled in the demo sandbox."),
    (re.compile(r"\bimport\s+requests\b"), "network requests are disabled in the demo sandbox."),
    (re.compile(r"\bimport\s+urllib\b"), "network access is disabled in the demo sandbox."),
    (re.compile(r"\bshutil\.rmtree\b"), "recursive deletion is disabled in the demo sandbox."),
    (re.compile(r"\beval\s*\("), "eval() is disabled in the demo sandbox."),
    (re.compile(r"\bexec\s*\("), "exec() is disabled in the demo sandbox."),
    (re.compile(r"__import__\s*\("), "dynamic imports are disabled in the demo sandbox."),
    (re.compile(r"open\s*\(\s*[rRuUbBfF]*['\"]\/"), "absolute file paths are disabled; use DATA_PATH and OUTPUT_DIR."),
    (re.compile(r"Path\s*\(\s*[rRuUbBfF]*['\"]\/"), "absolute file paths are disabled; use DATA_PATH and OUTPUT_DIR."),
]


@dataclass
class SandboxResult:
    ok: bool
    run_dir: str
    input_path: str | None
    output_dir: str
    stdout: str
    stderr: str
    exit_code: int | None
    timeout: bool
    generated_files: list[str]
    message: str


def rewrite_input_file_reads(code: str, input_path: Path | None) -> tuple[str, list[str]]:
    """Rewrite common hard-coded uploaded filenames to DATA_PATH.

    This fixes frequent LLM outputs such as pd.read_csv("sales.csv") while still
    leaving other code editable and visible in user_code.py.
    """
    if input_path is None:
        return code, []
    notes: list[str] = []

    def replace_csv(match: re.Match[str]) -> str:
        filename = match.group(2)
        if filename.startswith(("http://", "https://", "/")):
            return match.group(0)
        notes.append(f"Rewrote pd.read_csv({filename!r}) to pd.read_csv(DATA_PATH).")
        return f"{match.group(1)}DATA_PATH{match.group(3)}"

    def replace_excel(match: re.Match[str]) -> str:
        filename = match.group(2)
        if filename.startswith(("http://", "https://", "/")):
            return match.group(0)
        notes.append(f"Rewrote pd.read_excel({filename!r}) to pd.read_excel(DATA_PATH).")
        return f"{match.group(1)}DATA_PATH{match.group(3)}"

    code = re.sub(r"(pd\.read_csv\(\s*)[rRuUbBfF]*['\"]([^'\"]+\.csv)['\"](\s*[,\)])", replace_csv, code)
    code = re.sub(r"(pd\.read_excel\(\s*)[rRuUbBfF]*['\"]([^'\"]+\.xlsx?)['\"](\s*[,\)])", replace_excel, code)
    return code, notes


def _notes_suffix(notes: list[str]) -> str:
    return "" if not notes else " Notes: " + " ".join(notes)


def validate_code(code: str) -> list[str]:
    """Return policy violations found by lightweight static checks."""
    violations: list[str] = []
    for pattern, message in FORBIDDEN_PATTERNS:
        if pattern.search(code):
            violations.append(message)
    return violations


def run_python_in_sandbox(
    code: str,
    dataset_path: str | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    runs_root: str | Path = RUNS_ROOT,
) -> SandboxResult:
    """Run user-reviewed Python code in an isolated run directory."""
    code = (code or "").strip()
    run_dir = _new_run_dir(Path(runs_root))
    artifacts_dir = run_dir / "artifacts"
    original_dir = run_dir / "original"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    original_dir.mkdir(parents=True, exist_ok=True)

    input_path = _prepare_input_file(dataset_path, run_dir, original_dir)
    code, rewrite_notes = rewrite_input_file_reads(code, input_path)
    violations = validate_code(code)
    if not code:
        return _write_result(
            SandboxResult(False, str(run_dir), _str(input_path), str(artifacts_dir), "", "", None, False, [], "No code to run."),
            run_dir,
        )
    if violations:
        return _write_result(
            SandboxResult(
                False,
                str(run_dir),
                _str(input_path),
                str(artifacts_dir),
                "",
                "\n".join(violations),
                None,
                False,
                [],
                "Code rejected by sandbox policy." + _notes_suffix(rewrite_notes),
            ),
            run_dir,
        )

    user_code_path = run_dir / "user_code.py"
    user_code_path.write_text(_wrap_code(code, input_path, artifacts_dir), encoding="utf-8")

    stdout = ""
    stderr = ""
    exit_code: int | None = None
    timed_out = False
    try:
        completed = subprocess.run(
            [sys.executable, str(user_code_path.resolve())],
            cwd=run_dir,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        stdout = _truncate(completed.stdout)
        stderr = _truncate(completed.stderr)
        exit_code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = _truncate(exc.stdout or "")
        stderr = _truncate((exc.stderr or "") + f"\nExecution timed out after {timeout_seconds} seconds.")

    generated = _list_generated_files(artifacts_dir)
    result = SandboxResult(
        ok=(exit_code == 0 and not timed_out),
        run_dir=str(run_dir),
        input_path=_str(input_path),
        output_dir=str(artifacts_dir),
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        timeout=timed_out,
        generated_files=generated,
        message=("Execution finished." if exit_code == 0 and not timed_out else "Execution failed or timed out.") + _notes_suffix(rewrite_notes),
    )
    return _write_result(result, run_dir)


def format_result_markdown(result: SandboxResult) -> str:
    files = "\n".join(f"- `{path}`" for path in result.generated_files) or "No generated files."
    status = "success" if result.ok else "failed"
    return f"""### Sandbox {status}

Run directory: `{result.run_dir}`  
Input path: `{result.input_path or 'None'}`  
Output directory: `{result.output_dir}`  
Exit code: `{result.exit_code}`  
Timeout: `{result.timeout}`

**stdout**

```text
{result.stdout or '(empty)'}
```

**stderr**

```text
{result.stderr or '(empty)'}
```

**Generated files**

{files}
"""


def _new_run_dir(runs_root: Path) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _prepare_input_file(dataset_path: str | None, run_dir: Path, original_dir: Path) -> Path | None:
    if not dataset_path:
        return None
    source = Path(dataset_path)
    if not source.exists():
        return None
    original_copy = original_dir / source.name
    shutil.copy2(source, original_copy)
    suffix = source.suffix.lower()
    alias_name = "input.xlsx" if suffix in {".xlsx", ".xls"} else "input.csv"
    alias = run_dir / alias_name
    shutil.copy2(source, alias)
    root_original_alias = run_dir / source.name
    if root_original_alias.name != alias.name:
        shutil.copy2(source, root_original_alias)
    return alias


def _wrap_code(code: str, input_path: Path | None, artifacts_dir: Path) -> str:
    data_path = str(input_path.resolve()) if input_path else ""
    output_dir = str(artifacts_dir.resolve())
    return f"""from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use(\"Agg\")
import matplotlib.pyplot as plt

DATA_PATH = r\"{data_path}\"
OUTPUT_DIR = r\"{output_dir}\"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# User-reviewed code starts here.
{code}
"""


def _list_generated_files(artifacts_dir: Path) -> list[str]:
    if not artifacts_dir.exists():
        return []
    return sorted(str(path) for path in artifacts_dir.rglob("*") if path.is_file())


def _write_result(result: SandboxResult, run_dir: Path) -> SandboxResult:
    (run_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (run_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _truncate(text: str) -> str:
    if len(text) <= MAX_CAPTURE_CHARS:
        return text
    return text[:MAX_CAPTURE_CHARS] + "\n... [truncated]"


def _str(path: Path | None) -> str | None:
    return str(path) if path else None
