#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Chatbot Project – NLTK & Keras
File: chatbot.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-05
Updated: 2025-11-05
License: MIT License (see LICENSE file for details)
=
Description:
Runtime chatbot interface: classify user input and generate responses.
============================================================================
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from rich.console import Console
from rich.prompt import Prompt

from .config import INTENTS_PATH, ensure_artifact_dirs
from .model import load_trained_model
from .preprocessing import bag_of_words, load_metadata, tokenize


console = Console()


class Chatbot:
    """Retrieval-based chatbot using trained Keras model and intents.json."""

    def __init__(
        self,
        model_path: Path | None = None,
        metadata_path: Path | None = None,
        intents_path: Path | None = None,
    ) -> None:
        ensure_artifact_dirs()
        self.model = load_trained_model(model_path)
        metadata = load_metadata(metadata_path)
        self.words: List[str] = metadata["words"]
        self.labels: List[str] = metadata["labels"]
        self.intents = self._load_intents(intents_path or INTENTS_PATH)

    @staticmethod
    def _load_intents(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _predict_intent(self, sentence: str, threshold: float = 0.25) -> List[Tuple[str, float]]:
        """Return list of (intent_tag, probability) above threshold."""
        bow = bag_of_words(tokenize(sentence), self.words)
        bow = np.expand_dims(bow, axis=0)
        results = self.model.predict(bow, verbose=0)[0]
        results = [(self.labels[i], prob) for i, prob in enumerate(results) if prob > threshold]
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def get_response(self, sentence: str) -> str:
        """Generate a response for the user sentence."""
        intents_ranked = self._predict_intent(sentence)
        if not intents_ranked:
            tag = "noanswer"
        else:
            tag = intents_ranked[0][0]

        for intent in self.intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])

        # Fallback if tag not found
        return "I'm not sure I understand, but I'm learning more every day!"

    def chat(self) -> None:
        """Interactive command-line chat loop."""
        console.rule("[bold cyan]Chatbot[/bold cyan]")
        console.print("Type [bold magenta]'quit'[/bold magenta] to exit.\n")

        while True:
            user_input = Prompt.ask("[bold green]You[/bold green]")
            if user_input.strip().lower() in {"quit", "exit", "bye"}:
                console.print("[bold blue]Bot[/bold blue]: Goodbye! It was nice talking to you.")
                break

            response = self.get_response(user_input)
            console.print(f"[bold blue]Bot[/bold blue]: {response}")


def main() -> None:
    bot = Chatbot()
    bot.chat()


if __name__ == "__main__":
    main()
