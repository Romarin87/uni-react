"""Reaction task training task specification."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReactionTaskSpec:
    name: str = "reaction"
    train_mode: str = "reaction"


def resolve_reaction_task_spec(cfg) -> ReactionTaskSpec:
    return ReactionTaskSpec()
