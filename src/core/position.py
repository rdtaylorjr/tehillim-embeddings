"""Normalized half-verse position and equal-width binning, shared by every domain's zoning."""

from __future__ import annotations

import numpy as np


def half_verse_positions(n: int) -> np.ndarray:
    """Continuity-corrected normalized position t_i = (i - 0.5) / n, i = 1..n."""
    return (np.arange(1, n + 1) - 0.5) / n


def bin_index(t: np.ndarray, k: int) -> np.ndarray:
    """Which of k equal-width [0, 1) regions each normalized position falls into."""
    return np.asarray(np.minimum((t * k).astype(int), k - 1))
