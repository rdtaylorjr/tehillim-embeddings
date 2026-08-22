from __future__ import annotations

import numpy as np

from morphological.corpus import MorphologicalPsalm
from morphological.signature_vectorize import (
    morph_atomic_psalm_vectors,
    morph_atomic_vectors,
    morph_signature_1_2_3gram_psalm_sparse_vectors,
    morph_signature_1_2_3gram_psalm_vectors,
    morph_signature_1_2_3gram_sparse_vectors,
    morph_signature_1_2_3gram_vectors,
    morph_signature_1_2gram_psalm_vectors,
    morph_signature_1_2gram_vectors,
    morph_signature_psalm_vectors,
    morph_signature_vectors,
)
from morphological.vocabulary import (
    GN_VOCABULARY,
    NU_VOCABULARY,
    PS_VOCABULARY,
    SP_VOCABULARY,
    ST_VOCABULARY,
    VS_VOCABULARY,
    VT_VOCABULARY,
)


def _psalm(*, number, nodes, **feature_columns):
    return MorphologicalPsalm(
        number=number,
        half_verse_nodes=nodes,
        **{f"half_verse_{feature}": values for feature, values in feature_columns.items()},
    )


def _one_word_psalm(number, node):
    return _psalm(
        number=number,
        nodes=(node,),
        sp=(("subs",),),
        gn=(("m",),),
        nu=(("sg",),),
        ps=(("NA",),),
        st=(("a",),),
        vs=(("NA",),),
        vt=(("NA",),),
        prs_gn=(("NA",),),
        prs_nu=(("NA",),),
        prs_ps=(("NA",),),
    )


class TestMorphAtomicVectors:
    def test_dimension_is_sp_plus_the_six_core_feature_vocabularies_excluding_suffixes(self):
        psalms = [_one_word_psalm(1, 100)]
        vector = morph_atomic_vectors(psalms)[100]
        expected_dim = (
            len(SP_VOCABULARY)
            + len(GN_VOCABULARY)
            + len(NU_VOCABULARY)
            + len(PS_VOCABULARY)
            + len(ST_VOCABULARY)
            + len(VS_VOCABULARY)
            + len(VT_VOCABULARY)
        )
        assert vector.shape == (expected_dim,)
        assert expected_dim == 66

    def test_psalm_variant_broadcasts_and_has_the_same_dimension(self):
        psalms = [
            _psalm(
                number=1,
                nodes=(100, 101),
                **{
                    "sp": (("subs",), ("verb",)),
                    "gn": (("m",), ("m",)),
                    "nu": (("sg",), ("sg",)),
                    "ps": (("NA",), ("p3",)),
                    "st": (("a",), ("NA",)),
                    "vs": (("NA",), ("qal",)),
                    "vt": (("NA",), ("perf",)),
                    "prs_gn": (("NA",), ("NA",)),
                    "prs_nu": (("NA",), ("NA",)),
                    "prs_ps": (("NA",), ("NA",)),
                },
            )
        ]
        vectors = morph_atomic_psalm_vectors(psalms)
        assert np.allclose(vectors[100], vectors[101])
        assert vectors[100].shape == (66,)


class TestMorphSignatureVectors:
    def test_rare_below_threshold_signatures_collapse_before_histogramming(self):
        psalms = [_one_word_psalm(1, 100)]
        # "subs|m|sg|a" is not in the external counts at all, so it collapses to <RARE>
        # regardless of k, and the whole unigram mass lands on the <RARE> bin.
        external_counts = {"subs|m|sg|a": 1}
        vocabulary = ("subs|m|sg|a", "<RARE>")
        vector = morph_signature_vectors(psalms, vocabulary, external_counts, k=100)[100]
        assert np.isclose(vector[vocabulary.index("<RARE>")], 1.0)

    def test_keeps_a_signature_at_or_above_k(self):
        psalms = [_one_word_psalm(1, 100)]
        external_counts = {"subs|m|sg|a": 500}
        vocabulary = ("subs|m|sg|a", "<RARE>")
        vector = morph_signature_vectors(psalms, vocabulary, external_counts, k=100)[100]
        assert np.isclose(vector[vocabulary.index("subs|m|sg|a")], 1.0)

    def test_psalm_variant_broadcasts_the_same_vector(self):
        psalms = [
            _psalm(
                number=1,
                nodes=(100, 101),
                **{
                    "sp": (("subs",), ("subs",)),
                    "gn": (("m",), ("m",)),
                    "nu": (("sg",), ("sg",)),
                    "ps": (("NA",), ("NA",)),
                    "st": (("a",), ("a",)),
                    "vs": (("NA",), ("NA",)),
                    "vt": (("NA",), ("NA",)),
                    "prs_gn": (("NA",), ("NA",)),
                    "prs_nu": (("NA",), ("NA",)),
                    "prs_ps": (("NA",), ("NA",)),
                },
            )
        ]
        external_counts = {"subs|m|sg|a": 500}
        vocabulary = ("subs|m|sg|a", "<RARE>")
        vectors = morph_signature_psalm_vectors(psalms, vocabulary, external_counts, k=100)
        assert np.allclose(vectors[100], vectors[101])


class TestMorphSignatureNgramVectors:
    def test_1_2gram_dimension_is_unigram_plus_bigram_over_the_collapsed_vocabulary(self):
        psalms = [_one_word_psalm(1, 100)]
        vocabulary = ("subs|m|sg|a", "<RARE>")
        external_counts = {"subs|m|sg|a": 500}
        vector = morph_signature_1_2gram_vectors(psalms, vocabulary, external_counts, k=100)[100]
        assert vector.shape == (2 + 4,)

    def test_1_2_3gram_dimension_adds_the_trigram_block(self):
        psalms = [_one_word_psalm(1, 100)]
        vocabulary = ("subs|m|sg|a", "<RARE>")
        external_counts = {"subs|m|sg|a": 500}
        vector = morph_signature_1_2_3gram_vectors(psalms, vocabulary, external_counts, k=100)[100]
        assert vector.shape == (2 + 4 + 8,)

    def test_psalm_variants_have_matching_dimensions(self):
        psalms = [_one_word_psalm(1, 100)]
        vocabulary = ("subs|m|sg|a", "<RARE>")
        external_counts = {"subs|m|sg|a": 500}
        bigram_vec = morph_signature_1_2gram_psalm_vectors(
            psalms, vocabulary, external_counts, k=100
        )[100]
        trigram_vec = morph_signature_1_2_3gram_psalm_vectors(
            psalms, vocabulary, external_counts, k=100
        )[100]
        assert bigram_vec.shape == (6,)
        assert trigram_vec.shape == (14,)


def _multi_word_psalm(number, nodes):
    return _psalm(
        number=number,
        nodes=nodes,
        sp=(("subs", "verb", "subs", "verb", "prep"), ("verb", "subs")),
        gn=(("m", "NA", "f", "NA", "NA"), ("NA", "m")),
        nu=(("sg", "NA", "sg", "NA", "NA"), ("NA", "pl")),
        ps=(("NA", "p3", "NA", "p1", "NA"), ("p2", "NA")),
        st=(("a", "NA", "c", "NA", "NA"), ("NA", "a")),
        vs=(("NA", "qal", "NA", "piel", "NA"), ("hif", "NA")),
        vt=(("NA", "perf", "NA", "impf", "NA"), ("perf", "NA")),
        prs_gn=(("NA",) * 5, ("NA", "NA")),
        prs_nu=(("NA",) * 5, ("NA", "NA")),
        prs_ps=(("NA",) * 5, ("NA", "NA")),
    )


class TestMorphSignatureSparseVectors:
    def test_colon_level_matches_the_dense_computation_exactly(self):
        psalms = [_multi_word_psalm(1, (100, 101))]
        vocabulary = ("subs|m|sg|a", "verb|qal|perf|p3", "verb|piel|impf|p1", "<RARE>")
        external_counts = {"subs|m|sg|a": 5000, "verb|qal|perf|p3": 5000, "verb|piel|impf|p1": 5000}

        sparse = morph_signature_1_2_3gram_sparse_vectors(
            psalms, vocabulary, external_counts, k=1000
        )
        dense = morph_signature_1_2_3gram_vectors(psalms, vocabulary, external_counts, k=1000)

        for node in (100, 101):
            idx, val = sparse[node]
            reconstructed = np.zeros(dense[node].shape, dtype=np.float32)
            reconstructed[idx] = val
            assert np.array_equal(reconstructed, dense[node])

    def test_psalm_broadcast_matches_the_dense_computation_exactly(self):
        psalms = [_multi_word_psalm(1, (100, 101))]
        vocabulary = ("subs|m|sg|a", "verb|qal|perf|p3", "verb|piel|impf|p1", "<RARE>")
        external_counts = {"subs|m|sg|a": 5000, "verb|qal|perf|p3": 5000, "verb|piel|impf|p1": 5000}

        sparse = morph_signature_1_2_3gram_psalm_sparse_vectors(
            psalms, vocabulary, external_counts, k=1000
        )
        dense = morph_signature_1_2_3gram_psalm_vectors(psalms, vocabulary, external_counts, k=1000)

        idx, val = sparse[100]
        reconstructed = np.zeros(dense[100].shape, dtype=np.float32)
        reconstructed[idx] = val
        assert np.array_equal(reconstructed, dense[100])
        assert np.array_equal(sparse[100][0], sparse[101][0])
        assert np.array_equal(sparse[100][1], sparse[101][1])

    def test_colon_level_respects_order_by_node(self):
        psalms = [_multi_word_psalm(1, (100, 101))]
        vocabulary = ("subs|m|sg|a", "verb|qal|perf|p3", "verb|piel|impf|p1", "<RARE>")
        external_counts = {"subs|m|sg|a": 5000, "verb|qal|perf|p3": 5000, "verb|piel|impf|p1": 5000}
        order = {100: np.array([4, 3, 2, 1, 0])}

        unshuffled = morph_signature_1_2_3gram_sparse_vectors(
            psalms, vocabulary, external_counts, k=1000
        )
        shuffled = morph_signature_1_2_3gram_sparse_vectors(
            psalms, vocabulary, external_counts, k=1000, order_by_node=order
        )

        assert not (
            np.array_equal(unshuffled[100][0], shuffled[100][0])
            and np.array_equal(unshuffled[100][1], shuffled[100][1])
        )
