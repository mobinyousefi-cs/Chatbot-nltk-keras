#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Chatbot Project – NLTK & Keras
File: model.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-05
Updated: 2025-11-05
License: MIT License (see LICENSE file for details)
=
Description:
Keras model definition and training helpers for the chatbot intent classifier.
============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from tensorflow.keras.models import load_model  # type: ignore

from .config import ModelConfig, TrainingConfig, MODELS_DIR


def build_model(input_dim: int, output_dim: int, cfg: ModelConfig | None = None) -> Sequential:
    """Create and compile the Keras model."""
    cfg = cfg or ModelConfig()

    model = Sequential()
    model.add(Dense(cfg.hidden_units1, input_shape=(input_dim,), activation="relu"))
    model.add(Dropout(cfg.dropout_rate))
    model.add(Dense(cfg.hidden_units2, activation="relu"))
    model.add(Dropout(cfg.dropout_rate))
    model.add(Dense(output_dim, activation="softmax"))

    optimizer = Adam()
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


def train_model(
    training: np.ndarray,
    output: np.ndarray,
    model_cfg: ModelConfig | None = None,
    train_cfg: TrainingConfig | None = None,
) -> Tuple[Sequential, dict]:
    """Train the intent classification model."""
    model_cfg = model_cfg or ModelConfig()
    train_cfg = train_cfg or TrainingConfig()

    model = build_model(training.shape[1], output.shape[1], model_cfg)

    history = model.fit(
        training,
        output,
        epochs=train_cfg.epochs,
        batch_size=train_cfg.batch_size,
        validation_split=train_cfg.validation_split,
        verbose=train_cfg.verbose,
    )

    return model, history.history


def save_model(model: Sequential, path: Path | None = None) -> Path:
    """Save the trained model to disk."""
    path = path or (MODELS_DIR / "chatbot_model.h5")
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    return path


def load_trained_model(path: Path | None = None) -> Sequential:
    """Load a trained model from disk."""
    path = path or (MODELS_DIR / "chatbot_model.h5")
    return load_model(path)
