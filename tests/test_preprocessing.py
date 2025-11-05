#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Chatbot Project – NLTK & Keras
File: test_preprocessing.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-05
Updated: 2025-11-05
License: MIT License (see LICENSE file for details)
=
Description:
Basic unit tests for preprocessing helpers.
============================================================================
"""

from __future__ import annotations

from chatbot_nltk_keras.preprocessing import (
    bag_of_words,
    load_intents,
    preprocess_intents,
    tokenize,
)


def test_load_and_preprocess_intents() -> None:
    intents = load_intents()
    data = preprocess_intents(intents)

    assert len(data.words) > 0
    assert len(data.labels) > 0
    assert data.training.shape[0] == data.output.shape[0]
    assert data.training.shape[1] == len(data.words)


def test_tokenize_and_bow_consistency() -> None:
    vocab = ["hi", "bye", "thank", "you"]
    sentence = "Hi, thank you!"
    tokens = tokenize(sentence)
    bow = bag_of_words(tokens, vocab)

    # hi, thank, you should appear
    assert bow.sum() >= 2.0
