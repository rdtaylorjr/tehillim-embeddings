"""Full (typ:function:det) phrase-signature inventory histogram: H5.8's S+det representation."""

from __future__ import annotations

import numpy as np

from morphological.ngram import pooled_ngram_psalm_vectors, unigram_histogram
from phrase.corpus import PhrasePsalm
from phrase.signature import psalm_full_signatures
from phrase.signature_support import collapse_rare


def _collapsed_full_signatures(
    psalm: PhrasePsalm, external_counts: dict[str, int], k: int
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(collapse_rare(signature, external_counts, k) for signature in colon)
        for colon in psalm_full_signatures(psalm)
    )


def phrase_full_signature_vectors(
    psalms: list[PhrasePsalm], vocabulary: tuple[str, ...], external_counts: dict[str, int], k: int
) -> dict[int, np.ndarray]:
    """Full typ:function:det signature inventory histogram, RARE-collapsed at the unigram level."""
    index_of = {value: i for i, value in enumerate(vocabulary)}
    dim = len(vocabulary)
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        collapsed = _collapsed_full_signatures(psalm, external_counts, k)
        for node, colon_sigs in zip(psalm.half_verse_nodes, collapsed, strict=True):
            vectors[node] = unigram_histogram(colon_sigs, index_of, dim)
    return vectors


def phrase_full_signature_psalm_vectors(
    psalms: list[PhrasePsalm], vocabulary: tuple[str, ...], external_counts: dict[str, int], k: int
) -> dict[int, np.ndarray]:
    """Psalm-broadcast full-signature inventory histogram, atom-count-weighted pooling."""
    index_of = {value: i for i, value in enumerate(vocabulary)}
    dim = len(vocabulary)
    columns = [
        (psalm.half_verse_nodes, _collapsed_full_signatures(psalm, external_counts, k))
        for psalm in psalms
    ]
    return pooled_ngram_psalm_vectors(columns, (1,), index_of, dim, order_by_node=None)
