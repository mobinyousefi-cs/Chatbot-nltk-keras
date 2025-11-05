#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Chatbot Project – NLTK & Keras
File: config.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-05
Updated: 2025-11-05
License: MIT License (see LICENSE file for details)
=
Description:
Central configuration and path helpers for the chatbot project.
============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent if (PACKAGE_DIR.parent.parent / "pyproject.toml").exists() else PACKAGE_DIR

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR
INTENTS_PATH = PACKAGE_DIR / "data" / "intents.json"


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 200
    batch_size: int = 8
    learning_rate: float = 0.001
    validation_split: float = 0.1
    verbose: int = 1


@dataclass(frozen=True)
class ModelConfig:
    hidden_units1: int = 128
    hidden_units2: int = 64
    dropout_rate: float = 0.5


def ensure_artifact_dirs() -> None:
    """Create required directories on disk if they do not exist."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
