"""Phoneme-based length prediction for Irodori-TTS.

Estimates the synthesis duration (seconds) for a Japanese text by counting
phonemes from pyopenjtalk's grapheme-to-phoneme converter.
"""
from __future__ import annotations

import pyopenjtalk

PHONEMES_PER_SECOND = 11.0
TAIL_PADDING_SEC = 0.6
MIN_SECONDS = 2.0


def expected_seconds(text: str) -> float:
    """Estimate the synthesis duration (in seconds) for `text`.

    Phoneme count comes from pyopenjtalk.g2p; conversion factor and tail
    padding were tuned empirically against Irodori-TTS-v2 outputs.
    """
    phonemes = pyopenjtalk.g2p(text, kana=False).split()
    return max(MIN_SECONDS, round(len(phonemes) / PHONEMES_PER_SECOND + TAIL_PADDING_SEC, 2))
