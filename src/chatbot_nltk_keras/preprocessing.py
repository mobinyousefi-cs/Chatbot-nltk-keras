#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Chatbot Project – NLTK & Keras
File: preprocessing.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-05
Updated: 2025-11-05
License: MIT License (see LICENSE file for details)
=
Description:
Text preprocessing utilities: loading intents, tokenization, stemming, bag-of-words construction.
============================================================================
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import nltk
from nltk.stem import PorterStemmer

from .config import INTENTS_PATH, MODELS_DIR


Stemmer = PorterStemmer()


@dataclass
class PreprocessedData:
    words: List[str]
    labels: List[str]
    training: np.ndarray
    output: np.ndarray
    documents_x: List[List[str]]
    documents_y: List[str]


def ensure_nltk_data() -> None:
    """Ensure required NLTK resources are available."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")


def load_intents(path: Path | None = None) -> Dict[str, Any]:
    """Load intents JSON from disk."""
    path = path or INTENTS_PATH
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def tokenize(sentence: str) -> List[str]:
    """Tokenize a sentence into lowercase tokens."""
    ensure_nltk_data()
    return [token.lower() for token in nltk.word_tokenize(sentence)]


def stem(word: str) -> str:
    """Return stemmed version of a word."""
    return Stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence: List[str], vocabulary: List[str]) -> np.ndarray:
    """Create a bag-of-words vector from tokenized sentence and vocabulary."""
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(vocabulary), dtype=np.float32)

    for idx, w in enumerate(vocabulary):
        if w in sentence_words:
            bag[idx] = 1.0
    return bag


def preprocess_intents(intents: Dict[str, Any]) -> PreprocessedData:
    """Convert intents JSON into training data (bag-of-words + one-hot labels)."""
    words: List[str] = []
    labels: List[str] = []
    documents_x: List[List[str]] = []
    documents_y: List[str] = []

    for intent in intents["intents"]:
        tag = intent["tag"]
        if tag not in labels:
            labels.append(tag)

        for pattern in intent.get("patterns", []):
            w = tokenize(pattern)
            words.extend(w)
            documents_x.append(w)
            documents_y.append(tag)

    words = [stem(w) for w in words if w.isalpha()]
    words = sorted(sorted(set(words)))
    labels = sorted(labels)

    training: List[List[float]] = []
    output: List[List[float]] = []

    out_empty = [0.0] * len(labels)

    for idx, doc in enumerate(documents_x):
        bag = bag_of_words(doc, words)
        row = [0.0] * len(labels)
        row[labels.index(documents_y[idx])] = 1.0
        training.append(bag)
        output.append(row)

    return PreprocessedData(
        words=words,
        labels=labels,
        training=np.array(training, dtype=np.float32),
        output=np.array(output, dtype=np.float32),
        documents_x=documents_x,
        documents_y=documents_y,
    )


def save_metadata(data: PreprocessedData, path: Path | None = None) -> Path:
    """Persist vocabulary and labels for later use."""
    path = path or (MODELS_DIR / "metadata.pkl")
    payload = {"words": data.words, "labels": data.labels}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)
    return path


def load_metadata(path: Path | None = None) -> Dict[str, Any]:
    """Load vocabulary and labels from disk."""
    path = path or (MODELS_DIR / "metadata.pkl")
    with path.open("rb") as f:
        return pickle.load(f)
