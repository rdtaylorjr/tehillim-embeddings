from __future__ import annotations

import numpy as np

from syntax.complexity import (
    colon_complexity_features,
    phrase_complexity_psalm_vectors,
    phrase_complexity_vectors,
)
from syntax.corpus import PhrasePsalm


class TestColonComplexityFeatures:
    def test_is_all_zero_for_an_empty_colon(self):
        features = colon_complexity_features((), (), ())
        assert np.array_equal(features, np.zeros(4))

    def test_counts_atoms_and_distinct_phrases(self):
        # Two atoms belonging to the same (discontinuous) phrase: 2 atoms, 1 distinct phrase.
        features = colon_complexity_features((2, 3), (900, 900), (2, 2))
        assert features[0] == 2
        assert features[1] == 1

    def test_mean_words_per_atom(self):
        features = colon_complexity_features((2, 4), (900, 901), (1, 1))
        assert np.isclose(features[2], 3.0)

    def test_proportion_multi_atom_phrases(self):
        # Phrase 900 has 2 atoms total (multi-atom), phrase 901 has 1 (single-atom).
        features = colon_complexity_features((2, 3, 1), (900, 900, 901), (2, 2, 1))
        assert np.isclose(features[3], 0.5)

    def test_a_colon_with_only_single_atom_phrases_has_zero_proportion_multi_atom(self):
        features = colon_complexity_features((2, 3), (900, 901), (1, 1))
        assert features[3] == 0.0


def _psalm(*, number, nodes, n_words, phrase_id, phrase_atom_count):
    return PhrasePsalm(
        number=number,
        half_verse_nodes=nodes,
        half_verse_n_words=n_words,
        half_verse_phrase_id=phrase_id,
        half_verse_phrase_atom_count=phrase_atom_count,
    )


class TestPhraseComplexityVectors:
    def test_keys_vectors_by_colon_node_id(self):
        psalms = [
            _psalm(
                number=1,
                nodes=(100, 101),
                n_words=((2,), (3,)),
                phrase_id=((900,), (901,)),
                phrase_atom_count=((1,), (1,)),
            )
        ]
        vectors = phrase_complexity_vectors(psalms)
        assert set(vectors) == {100, 101}
        assert vectors[100].shape == (4,)


class TestPhraseComplexityPsalmVectors:
    def test_broadcasts_the_unweighted_mean_across_colons(self):
        psalms = [
            _psalm(
                number=1,
                nodes=(100, 101),
                n_words=((2, 2), (4,)),
                phrase_id=((900, 901), (902,)),
                phrase_atom_count=((1, 1), (1,)),
            )
        ]
        vectors = phrase_complexity_psalm_vectors(psalms)
        assert np.allclose(vectors[100], vectors[101])
        # colon 100: n_atoms=2; colon 101: n_atoms=1. Mean = 1.5.
        assert np.isclose(vectors[100][0], 1.5)
