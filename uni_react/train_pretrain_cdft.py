#!/usr/bin/env python3
"""CDFT reactivity pretraining entry-point."""
from __future__ import annotations

from uni_react.tasks.cdft import run_cdft_entry


def main() -> None:
    run_cdft_entry()


if __name__ == "__main__":
    main()
