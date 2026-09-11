"""Psalm-scale phrase-signature deployment: uniform inventory plus mean position."""

from __future__ import annotations

import numpy as np

from core.deploy import psalm_deploy_vectors
from core.support import collapsed_sequences
from syntactic.corpus import PhrasePsalm
from syntactic.signature import psalm_signatures


def signature_deploy_vectors(
    psalms: list[PhrasePsalm],
    vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
    order_by_psalm: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """`psalm_deploy_vectors` over the RARE-collapsed phrase-signature vocabulary (H5.9)."""

    def collapsed(psalm: PhrasePsalm) -> tuple[tuple[str, ...], ...]:
        """One half-verse's signatures with sub-threshold ones collapsed to RARE."""
        return collapsed_sequences(psalm_signatures(psalm), external_counts, k)

    return psalm_deploy_vectors(psalms, vocabulary, collapsed, order_by_psalm)
