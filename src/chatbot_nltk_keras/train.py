#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Chatbot Project – NLTK & Keras
File: train.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-05
Updated: 2025-11-05
License: MIT License (see LICENSE file for details)
=
Description:
Command-line entry point for training the chatbot model.
============================================================================
"""

from __future__ import annotations

import argparse
from typing import Any, Dict

from rich.console import Console

from .config import TrainingConfig, ModelConfig, ensure_artifact_dirs
from .model import save_model, train_model
from .preprocessing import PreprocessedData, load_intents, preprocess_intents, save_metadata


console = Console()


def run_training(extra_args: Dict[str, Any] | None = None) -> None:
    """Train the chatbot model and save all artifacts."""
    console.rule("[bold cyan]Chatbot Training[/bold cyan]")
    ensure_artifact_dirs()

    intents = load_intents()
    data: PreprocessedData = preprocess_intents(intents)

    console.print(f"[green]Loaded[/green] {len(data.documents_x)} patterns "
                  f"across {len(data.labels)} intents.")
    console.print(f"Vocabulary size: [yellow]{len(data.words)}[/yellow]")

    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()

    model, history = train_model(data.training, data.output, model_cfg, train_cfg)

    model_path = save_model(model)
    metadata_path = save_metadata(data)

    console.print(f"\n[bold green]Training complete![/bold green]")
    console.print(f"Model saved to: [cyan]{model_path}[/cyan]")
    console.print(f"Metadata saved to: [cyan]{metadata_path}[/cyan]")

    if "val_accuracy" in history:
        last_acc = history["accuracy"][-1]
        last_val_acc = history["val_accuracy"][-1]
        console.print(
            f"Final accuracy: [green]{last_acc:.3f}[/green], "
            f"val_accuracy: [green]{last_val_acc:.3f}[/green]"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the retrieval-based chatbot model using NLTK & Keras."
    )
    # You can add CLI overrides for epochs, batch_size, etc., in the future.
    return parser.parse_args()


def main() -> None:
    _ = parse_args()
    run_training()


if __name__ == "__main__":
    main()
