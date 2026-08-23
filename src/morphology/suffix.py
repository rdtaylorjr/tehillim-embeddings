"""Pronominal-suffix signatures and the host-signature-plus-suffix representation (Phase 4D)."""

from __future__ import annotations

import numpy as np

from morphology.corpus import MorphologicalPsalm
from morphology.ngram import PsalmColumns, pooled_ngram_psalm_vectors, unigram_histogram
from morphology.signature import psalm_signatures
from morphology.signature_support import collapse_rare
from morphology.vocabulary import PRS_GN_VOCABULARY, PRS_NU_VOCABULARY, PRS_PS_VOCABULARY

NONE_SUFFIX_TOKEN = "<NONE>"

_SUFFIX_FIELD_ORDER: tuple[str, ...] = ("prs_ps", "prs_gn", "prs_nu")


def build_suffix_signature(*, prs_gn: str, prs_nu: str, prs_ps: str) -> str:
    """`prs_ps|prs_gn|prs_nu`-style signature, NA fields dropped, `<NONE>` when all three are NA."""
    values_by_field = {"prs_gn": prs_gn, "prs_nu": prs_nu, "prs_ps": prs_ps}
    parts = [values_by_field[f] for f in _SUFFIX_FIELD_ORDER if values_by_field[f] != "NA"]
    return "|".join(parts) if parts else NONE_SUFFIX_TOKEN


def _all_suffix_signatures() -> tuple[str, ...]:
    signatures = {
        build_suffix_signature(prs_gn=gn, prs_nu=nu, prs_ps=ps)
        for ps in PRS_PS_VOCABULARY
        for gn in PRS_GN_VOCABULARY
        for nu in PRS_NU_VOCABULARY
    }
    return tuple(sorted(signatures))


SUFFIX_VOCABULARY: tuple[str, ...] = _all_suffix_signatures()


def colon_suffix_signatures(
    *, prs_gn: tuple[str, ...], prs_nu: tuple[str, ...], prs_ps: tuple[str, ...]
) -> tuple[str, ...]:
    """One suffix signature per word, aligned across the three per-word suffix feature sequences."""
    return tuple(
        build_suffix_signature(prs_gn=w_gn, prs_nu=w_nu, prs_ps=w_ps)
        for w_gn, w_nu, w_ps in zip(prs_gn, prs_nu, prs_ps, strict=True)
    )


def psalm_suffix_signatures(psalm: MorphologicalPsalm) -> tuple[tuple[str, ...], ...]:
    """One suffix signature sequence per colon of `psalm`."""
    return tuple(
        colon_suffix_signatures(prs_gn=gn, prs_nu=nu, prs_ps=ps)
        for gn, nu, ps in zip(
            psalm.colon_prs_gn, psalm.colon_prs_nu, psalm.colon_prs_ps, strict=True
        )
    )


def suffix_inventory_vectors(psalms: list[MorphologicalPsalm]) -> dict[int, np.ndarray]:
    """`M_S`: normalized suffix-signature proportions per colon node, over `SUFFIX_VOCABULARY`."""
    index_of = {value: i for i, value in enumerate(SUFFIX_VOCABULARY)}
    dim = len(SUFFIX_VOCABULARY)
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        for node, colon_sigs in zip(psalm.colon_nodes, psalm_suffix_signatures(psalm), strict=True):
            vectors[node] = unigram_histogram(colon_sigs, index_of, dim)
    return vectors


def suffix_inventory_psalm_vectors(psalms: list[MorphologicalPsalm]) -> dict[int, np.ndarray]:
    """Psalm-broadcast `suffix_inventory_vectors`, word-count-weighted pooling."""
    index_of = {value: i for i, value in enumerate(SUFFIX_VOCABULARY)}
    dim = len(SUFFIX_VOCABULARY)
    columns: list[PsalmColumns] = [
        (psalm.colon_nodes, psalm_suffix_signatures(psalm)) for psalm in psalms
    ]
    return pooled_ngram_psalm_vectors(columns, (1,), index_of, dim, order_by_node=None)


def _collapsed_signatures(
    psalm: MorphologicalPsalm, external_counts: dict[str, int], k: int
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(collapse_rare(signature, external_counts, k) for signature in colon)
        for colon in psalm_signatures(psalm)
    )


def host_plus_suffix_vectors(
    psalms: list[MorphologicalPsalm],
    signature_vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
) -> dict[int, np.ndarray]:
    """`[M_G; M_S]` per colon node: RARE-collapsed host signature, then suffix, inventory blocks."""
    host_index_of = {value: i for i, value in enumerate(signature_vocabulary)}
    host_dim = len(signature_vocabulary)
    suffix_index_of = {value: i for i, value in enumerate(SUFFIX_VOCABULARY)}
    suffix_dim = len(SUFFIX_VOCABULARY)
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        collapsed_host = _collapsed_signatures(psalm, external_counts, k)
        suffix_sigs = psalm_suffix_signatures(psalm)
        for node, host_colon, suffix_colon in zip(
            psalm.colon_nodes, collapsed_host, suffix_sigs, strict=True
        ):
            vectors[node] = np.concatenate(
                [
                    unigram_histogram(host_colon, host_index_of, host_dim),
                    unigram_histogram(suffix_colon, suffix_index_of, suffix_dim),
                ]
            )
    return vectors


def host_plus_suffix_psalm_vectors(
    psalms: list[MorphologicalPsalm],
    signature_vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
) -> dict[int, np.ndarray]:
    """Psalm-broadcast `host_plus_suffix_vectors`, each half word-count-weighted independently."""
    host_index_of = {value: i for i, value in enumerate(signature_vocabulary)}
    host_dim = len(signature_vocabulary)
    host_columns: list[PsalmColumns] = [
        (psalm.colon_nodes, _collapsed_signatures(psalm, external_counts, k)) for psalm in psalms
    ]
    host_vectors = pooled_ngram_psalm_vectors(
        host_columns, (1,), host_index_of, host_dim, order_by_node=None
    )
    suffix_vectors = suffix_inventory_psalm_vectors(psalms)
    return {
        node: np.concatenate([host_vectors[node], suffix_vectors[node]]) for node in host_vectors
    }
