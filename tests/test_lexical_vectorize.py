from __future__ import annotations

import numpy as np

from lexical.corpus import LexicalPsalm
from lexical.vectorize import binary_presence_vectors


def _psalm(*, number, lexemes, forms, nodes):
    return LexicalPsalm(
        number=number,
        half_verse_lexemes=lexemes,
        half_verse_forms=forms,
        half_verse_nodes=nodes,
    )


class TestBinaryPresenceVectors:
    def test_marks_present_lexemes_as_one_and_absent_as_zero(self):
        vocabulary = ("A", "B", "C")
        psalms = [_psalm(number=1, lexemes=(("A", "C"),), forms=((),), nodes=(100,))]

        vectors = binary_presence_vectors(psalms, vocabulary, key="lex")

        assert np.array_equal(vectors[100], [1.0, 0.0, 1.0])

    def test_a_lexeme_repeated_within_a_colon_still_scores_one_not_two(self):
        vocabulary = ("A", "B")
        psalms = [_psalm(number=1, lexemes=(("A", "A", "A"),), forms=((),), nodes=(100,))]

        vectors = binary_presence_vectors(psalms, vocabulary, key="lex")

        assert np.array_equal(vectors[100], [1.0, 0.0])

    def test_every_vector_has_length_equal_to_the_vocabulary_size(self):
        vocabulary = ("A", "B", "C", "D", "E")
        psalms = [_psalm(number=1, lexemes=(("A",), ("B", "C")), forms=((), ()), nodes=(100, 101))]

        vectors = binary_presence_vectors(psalms, vocabulary, key="lex")

        assert all(len(v) == 5 for v in vectors.values())

    def test_covers_every_half_verse_node_across_multiple_psalms(self):
        vocabulary = ("A", "B")
        psalms = [
            _psalm(number=1, lexemes=(("A",),), forms=((),), nodes=(100,)),
            _psalm(number=2, lexemes=(("B",), ("A", "B")), forms=((), ()), nodes=(200, 201)),
        ]

        vectors = binary_presence_vectors(psalms, vocabulary, key="lex")

        assert set(vectors) == {100, 200, 201}
        assert np.array_equal(vectors[201], [1.0, 1.0])

    def test_uses_lex0_forms_when_key_is_lex0(self):
        vocabulary = ("A0", "B0")
        psalms = [_psalm(number=1, lexemes=(("A", "B"),), forms=(("A0",),), nodes=(100,))]

        vectors = binary_presence_vectors(psalms, vocabulary, key="lex0")

        assert np.array_equal(vectors[100], [1.0, 0.0])

    def test_ignores_values_absent_from_the_given_vocabulary(self):
        vocabulary = ("A",)
        psalms = [_psalm(number=1, lexemes=(("A", "OUT_OF_VOCAB"),), forms=((),), nodes=(100,))]

        vectors = binary_presence_vectors(psalms, vocabulary, key="lex")

        assert np.array_equal(vectors[100], [1.0])

    def test_vectors_are_float32(self):
        vocabulary = ("A",)
        psalms = [_psalm(number=1, lexemes=(("A",),), forms=((),), nodes=(100,))]

        vectors = binary_presence_vectors(psalms, vocabulary, key="lex")

        assert vectors[100].dtype == np.float32
