from __future__ import annotations

import numpy as np

from phrase.corpus import PhrasePsalm
from phrase.signature_vectorize import (
    phrase_signature_1_2_3gram_psalm_vectors,
    phrase_signature_1_2_3gram_vectors,
    phrase_signature_1_2gram_psalm_vectors,
    phrase_signature_1_2gram_vectors,
    phrase_signature_psalm_vectors,
    phrase_signature_vectors,
)


def _psalm(*, number, nodes, typ, function):
    return PhrasePsalm(
        number=number, half_verse_nodes=nodes, half_verse_typ=typ, half_verse_function=function
    )


def _one_atom_psalm(number, node):
    return _psalm(number=number, nodes=(node,), typ=(("NP",),), function=(("Subj",),))


class TestPhraseSignatureVectors:
    def test_rare_below_threshold_signatures_collapse_before_histogramming(self):
        psalms = [_one_atom_psalm(1, 100)]
        # "NP:Subj" is not in the external counts at all, so it collapses to <RARE>
        # regardless of k, and the whole unigram mass lands on the <RARE> bin.
        external_counts = {"NP:Subj": 1}
        vocabulary = ("NP:Subj", "<RARE>")
        vector = phrase_signature_vectors(psalms, vocabulary, external_counts, k=100)[100]
        assert np.isclose(vector[vocabulary.index("<RARE>")], 1.0)

    def test_keeps_a_signature_at_or_above_k(self):
        psalms = [_one_atom_psalm(1, 100)]
        external_counts = {"NP:Subj": 500}
        vocabulary = ("NP:Subj", "<RARE>")
        vector = phrase_signature_vectors(psalms, vocabulary, external_counts, k=100)[100]
        assert np.isclose(vector[vocabulary.index("NP:Subj")], 1.0)

    def test_psalm_variant_broadcasts_the_same_vector(self):
        psalms = [
            _psalm(
                number=1,
                nodes=(100, 101),
                typ=(("NP",), ("NP",)),
                function=(("Subj",), ("Subj",)),
            )
        ]
        external_counts = {"NP:Subj": 500}
        vocabulary = ("NP:Subj", "<RARE>")
        vectors = phrase_signature_psalm_vectors(psalms, vocabulary, external_counts, k=100)
        assert np.allclose(vectors[100], vectors[101])


class TestPhraseSignatureNgramVectors:
    def test_1_2gram_dimension_is_unigram_plus_bigram_over_the_collapsed_vocabulary(self):
        psalms = [_one_atom_psalm(1, 100)]
        vocabulary = ("NP:Subj", "<RARE>")
        external_counts = {"NP:Subj": 500}
        vector = phrase_signature_1_2gram_vectors(psalms, vocabulary, external_counts, k=100)[100]
        assert vector.shape == (2 + 4,)

    def test_1_2_3gram_dimension_adds_the_trigram_block(self):
        psalms = [_one_atom_psalm(1, 100)]
        vocabulary = ("NP:Subj", "<RARE>")
        external_counts = {"NP:Subj": 500}
        vector = phrase_signature_1_2_3gram_vectors(psalms, vocabulary, external_counts, k=100)[100]
        assert vector.shape == (2 + 4 + 8,)

    def test_psalm_variants_have_matching_dimensions(self):
        psalms = [_one_atom_psalm(1, 100)]
        vocabulary = ("NP:Subj", "<RARE>")
        external_counts = {"NP:Subj": 500}
        bigram_vec = phrase_signature_1_2gram_psalm_vectors(
            psalms, vocabulary, external_counts, k=100
        )[100]
        trigram_vec = phrase_signature_1_2_3gram_psalm_vectors(
            psalms, vocabulary, external_counts, k=100
        )[100]
        assert bigram_vec.shape == (2 + 4,)
        assert trigram_vec.shape == (2 + 4 + 8,)
