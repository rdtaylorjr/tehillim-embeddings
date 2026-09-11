"""Full (typ:function:det) phrase-signature histograms: 1gram, 1_2gram and sparse 1_2_3gram."""

from __future__ import annotations

from functools import partial

from core.ngram import (
    signature_psalm_vectors,
    signature_vectors,
    sparse_signature_psalm_vectors,
    sparse_signature_vectors,
)
from syntactic.signature import psalm_full_signatures

#: Each construction is the same n-gram build over RARE-collapsed full-signature sequences.
phrase_full_signature_vectors = partial(signature_vectors, psalm_full_signatures, (1,))
phrase_full_signature_psalm_vectors = partial(signature_psalm_vectors, psalm_full_signatures, (1,))
phrase_full_signature_1_2gram_vectors = partial(signature_vectors, psalm_full_signatures, (1, 2))
phrase_full_signature_1_2gram_psalm_vectors = partial(
    signature_psalm_vectors, psalm_full_signatures, (1, 2)
)
phrase_full_signature_1_2_3gram_sparse_vectors = partial(
    sparse_signature_vectors, psalm_full_signatures
)
phrase_full_signature_1_2_3gram_psalm_sparse_vectors = partial(
    sparse_signature_psalm_vectors, psalm_full_signatures
)

DENSE_BUILDERS = {
    "1gram": phrase_full_signature_vectors,
    "1gram_psalm": phrase_full_signature_psalm_vectors,
    "1_2gram": phrase_full_signature_1_2gram_vectors,
    "1_2gram_psalm": phrase_full_signature_1_2gram_psalm_vectors,
}

#: The trigram block is overwhelmingly zero at this dimension, so these are stored sparsely.
SPARSE_BUILDERS = {
    "1_2_3gram": phrase_full_signature_1_2_3gram_sparse_vectors,
    "1_2_3gram_psalm": phrase_full_signature_1_2_3gram_psalm_sparse_vectors,
}
