#!/usr/bin/env python3
"""Electron-density pretraining entry-point."""
from __future__ import annotations

from uni_react.training.density_runner import run_density_entry


def main() -> None:
    run_density_entry()


if __name__ == "__main__":
    main()
