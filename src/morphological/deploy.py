"""Psalm-scale grammatical deployment: uniform-weight inventory concatenated with mean position."""

from __future__ import annotations

import numpy as np

from core.deploy import psalm_deploy_vectors
from morphological.corpus import MorphologicalPsalm
from morphological.suffix import SUFFIX_VOCABULARY, psalm_suffix_signatures


def suffix_deploy_vectors(
    psalms: list[MorphologicalPsalm], order_by_psalm: dict[int, np.ndarray] | None = None
) -> dict[int, np.ndarray]:
    """`psalm_deploy_vectors` over the pronominal-suffix vocabulary, the 4B-4D-strongest signal."""
    return psalm_deploy_vectors(psalms, SUFFIX_VOCABULARY, psalm_suffix_signatures, order_by_psalm)
