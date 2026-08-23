from __future__ import annotations

import numpy as np

from morphology.atomic import (
    atomic_histogram,
    atomic_psalm_vectors,
    atomic_vectors,
    full_morphology_psalm_vectors,
    full_morphology_vectors,
    sp_plus_feature_psalm_vectors,
    sp_plus_feature_vectors,
)
from morphology.corpus import MorphologicalPsalm
from morphology.vocabulary import (
    GN_VOCABULARY,
    NU_VOCABULARY,
    PRS_GN_VOCABULARY,
    PRS_NU_VOCABULARY,
    PRS_PS_VOCABULARY,
    PS_VOCABULARY,
    SP_VOCABULARY,
    ST_VOCABULARY,
    VS_VOCABULARY,
    VT_VOCABULARY,
)

_FEATURE_DIMS = {
    "gn": len(GN_VOCABULARY),
    "nu": len(NU_VOCABULARY),
    "ps": len(PS_VOCABULARY),
    "st": len(ST_VOCABULARY),
    "vs": len(VS_VOCABULARY),
    "vt": len(VT_VOCABULARY),
    "prs_gn": len(PRS_GN_VOCABULARY),
    "prs_nu": len(PRS_NU_VOCABULARY),
    "prs_ps": len(PRS_PS_VOCABULARY),
}


def _psalm(*, number, nodes, **feature_columns):
    return MorphologicalPsalm(
        number=number,
        half_verse_nodes=nodes,
        **{f"half_verse_{feature}": values for feature, values in feature_columns.items()},
    )


class TestAtomicHistogram:
    def test_sums_to_one_for_a_non_empty_colon(self):
        histogram = atomic_histogram(("NA", "NA", "qal"), VS_VOCABULARY)
        assert np.isclose(histogram.sum(), 1.0)

    def test_na_counts_as_part_of_the_same_distribution(self):
        # An all-noun colon: vs is NA for every word, so the NA bin should hold the full mass,
        # exposing the applicability-rate confound rather than hiding it.
        histogram = atomic_histogram(("NA", "NA", "NA"), VS_VOCABULARY)
        na_index = VS_VOCABULARY.index("NA")
        assert np.isclose(histogram[na_index], 1.0)
        assert np.isclose(histogram.sum(), 1.0)

    def test_unknown_bin_is_distinct_from_na_bin(self):
        histogram = atomic_histogram(("NA", "unknown", "f"), GN_VOCABULARY)
        na_index = GN_VOCABULARY.index("NA")
        unknown_index = GN_VOCABULARY.index("unknown")
        f_index = GN_VOCABULARY.index("f")
        assert np.isclose(histogram[na_index], 1 / 3)
        assert np.isclose(histogram[unknown_index], 1 / 3)
        assert np.isclose(histogram[f_index], 1 / 3)


class TestAtomicVectors:
    def test_dimension_matches_the_features_vocabulary(self):
        psalms = [_psalm(number=1, nodes=(100,), vs=(("qal", "NA"),))]
        vectors = atomic_vectors(psalms, "vs")
        assert vectors[100].shape == (_FEATURE_DIMS["vs"],)

    def test_a_colon_of_all_nouns_is_entirely_na_for_a_verb_only_feature(self):
        psalms = [_psalm(number=1, nodes=(100,), vt=(("NA", "NA", "NA"),))]
        vector = atomic_vectors(psalms, "vt")[100]
        na_index = VT_VOCABULARY.index("NA")
        assert np.isclose(vector[na_index], 1.0)


class TestAtomicPsalmVectors:
    def test_broadcasts_the_identical_vector_to_every_colon_node(self):
        psalms = [_psalm(number=1, nodes=(200, 201), gn=(("m",), ("f",)))]
        vectors = atomic_psalm_vectors(psalms, "gn")
        assert np.allclose(vectors[200], vectors[201])

    def test_pools_raw_word_counts_across_colons_before_normalizing_once(self):
        psalms = [_psalm(number=1, nodes=(300, 301), gn=(("m",), ("f", "f", "NA")))]
        vector = atomic_psalm_vectors(psalms, "gn")[300]
        m_index, f_index, na_index = (GN_VOCABULARY.index(v) for v in ("m", "f", "NA"))
        assert np.isclose(vector[m_index], 0.25)
        assert np.isclose(vector[f_index], 0.5)
        assert np.isclose(vector[na_index], 0.25)


class TestSpPlusFeatureVectors:
    def test_concatenates_sp_unigram_and_the_atomic_feature_histogram(self):
        psalms = [
            _psalm(
                number=1,
                nodes=(400,),
                sp=(("subs", "verb"),),
                vs=(("NA", "qal"),),
            )
        ]
        vector = sp_plus_feature_vectors(psalms, "vs")[400]
        assert vector.shape == (len(SP_VOCABULARY) + _FEATURE_DIMS["vs"],)

    def test_psalm_variant_broadcasts_the_same_vector(self):
        psalms = [
            _psalm(
                number=1,
                nodes=(500, 501),
                sp=(("subs",), ("verb",)),
                vs=(("NA",), ("qal",)),
            )
        ]
        vectors = sp_plus_feature_psalm_vectors(psalms, "vs")
        assert np.allclose(vectors[500], vectors[501])


class TestFullMorphologyVectors:
    def test_dimension_is_the_sum_of_sp_plus_all_nine_atomic_vocabularies(self):
        psalms = [
            _psalm(
                number=1,
                nodes=(600,),
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
        ]
        vector = full_morphology_vectors(psalms)[600]
        expected_dim = len(SP_VOCABULARY) + sum(_FEATURE_DIMS.values())
        assert vector.shape == (expected_dim,)
        assert expected_dim == 77

    def test_psalm_variant_has_the_same_dimension(self):
        psalms = [
            _psalm(
                number=1,
                nodes=(700,),
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
        ]
        vector = full_morphology_psalm_vectors(psalms)[700]
        assert vector.shape == (77,)
