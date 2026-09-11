"""Grammatical-signature histograms: bundle inventory, and cumulative bigram/trigram sequences."""

from __future__ import annotations

from functools import partial

from core.ngram import (
    signature_psalm_vectors,
    signature_vectors,
    sparse_signature_psalm_vectors,
    sparse_signature_vectors,
)
from core.vocabulary import index_map
from morphological.atomic import (
    FeatureKey,
    concatenated_feature_psalm_vectors,
    concatenated_feature_vectors,
)
from morphological.signature import psalm_signatures
from morphological.vocabulary import SP_VOCABULARY

_SP_INDEX = index_map(SP_VOCABULARY)

_CORE_FEATURE_ORDER: tuple[FeatureKey, ...] = ("gn", "nu", "ps", "st", "vs", "vt")


morph_atomic_vectors = partial(concatenated_feature_vectors, features=_CORE_FEATURE_ORDER)
morph_atomic_psalm_vectors = partial(
    concatenated_feature_psalm_vectors, features=_CORE_FEATURE_ORDER
)


#: Each construction is the same n-gram build over RARE-collapsed signature sequences.
morph_signature_vectors = partial(signature_vectors, psalm_signatures, (1,))
morph_signature_psalm_vectors = partial(signature_psalm_vectors, psalm_signatures, (1,))
morph_signature_1_2gram_vectors = partial(signature_vectors, psalm_signatures, (1, 2))
morph_signature_1_2gram_psalm_vectors = partial(signature_psalm_vectors, psalm_signatures, (1, 2))
morph_signature_1_2_3gram_vectors = partial(signature_vectors, psalm_signatures, (1, 2, 3))
morph_signature_1_2_3gram_psalm_vectors = partial(
    signature_psalm_vectors, psalm_signatures, (1, 2, 3)
)
morph_signature_1_2_3gram_sparse_vectors = partial(sparse_signature_vectors, psalm_signatures)
morph_signature_1_2_3gram_psalm_sparse_vectors = partial(
    sparse_signature_psalm_vectors, psalm_signatures
)


DENSE_BUILDERS = {
    "1gram": morph_signature_vectors,
    "1gram_psalm": morph_signature_psalm_vectors,
    "1_2gram": morph_signature_1_2gram_vectors,
    "1_2gram_psalm": morph_signature_1_2gram_psalm_vectors,
}

#: The trigram block is overwhelmingly zero at this dimension, so these are stored sparsely.
SPARSE_BUILDERS = {
    "1_2_3gram": morph_signature_1_2_3gram_sparse_vectors,
    "1_2_3gram_psalm": morph_signature_1_2_3gram_psalm_sparse_vectors,
}

#: The inventory histograms count without reading order, so a half-verse shuffle cannot move them.
ORDER_INVARIANT = ("1gram", "1gram_psalm")

ORDERED_DENSE_BUILDERS = {
    "1_2gram": morph_signature_1_2gram_vectors,
    "1_2gram_psalm": morph_signature_1_2gram_psalm_vectors,
}
