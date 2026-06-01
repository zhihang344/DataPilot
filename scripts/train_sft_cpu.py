#!/usr/bin/env python
"""Compatibility wrapper for the GPU SFT script.

Prefer ``scripts/train_sft_gpu.py`` for this project. This wrapper keeps older
commands working and forwards all arguments to the new entry point.
"""

from __future__ import annotations

from train_sft_gpu import main


if __name__ == "__main__":
    main()
