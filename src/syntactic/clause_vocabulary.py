"""Frozen BHSA clause vocabularies, verified against the raw `.tf` files, not their headers."""

from __future__ import annotations

__all__ = ["KIND_VOCABULARY"]

#: BHSA's three coarse clause classes: verbal, nominal, and without predication.
KIND_VOCABULARY: tuple[str, ...] = ("NC", "VC", "WP")
