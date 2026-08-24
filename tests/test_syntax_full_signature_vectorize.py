from __future__ import annotations

import numpy as np

from syntax.corpus import PhrasePsalm
from syntax.full_signature_vectorize import (
    phrase_full_signature_psalm_vectors,
    phrase_full_signature_vectors,
)


def _psalm(*, number, nodes, typ, function, det):
    return PhrasePsalm(
        number=number,
        colon_nodes=nodes,
        colon_typ=typ,
        colon_function=function,
        colon_det=det,
    )


def _one_atom_psalm(number, node, det="det"):
    return _psalm(
        number=number, nodes=(node,), typ=(("NP",),), function=(("Subj",),), det=((det,),)
    )


class TestPhraseFullSignatureVectors:
    def test_rare_below_threshold_signatures_collapse_before_histogramming(self):
        psalms = [_one_atom_psalm(1, 100)]
        external_counts = {"NP:Subj:det": 1}
        vocabulary = ("NP:Subj:det", "<RARE>")
        vector = phrase_full_signature_vectors(psalms, vocabulary, external_counts, k=100)[100]
        assert np.isclose(vector[vocabulary.index("<RARE>")], 1.0)

    def test_keeps_a_phrase_signature_at_or_above_k(self):
        psalms = [_one_atom_psalm(1, 100)]
        external_counts = {"NP:Subj:det": 500}
        vocabulary = ("NP:Subj:det", "<RARE>")
        vector = phrase_full_signature_vectors(psalms, vocabulary, external_counts, k=100)[100]
        assert np.isclose(vector[vocabulary.index("NP:Subj:det")], 1.0)

    def test_na_phrase_det_omits_the_phrase_det_field_from_the_signature(self):
        psalms = [_one_atom_psalm(1, 100, det="NA")]
        external_counts = {"NP:Subj": 500}
        vocabulary = ("NP:Subj", "<RARE>")
        vector = phrase_full_signature_vectors(psalms, vocabulary, external_counts, k=100)[100]
        assert np.isclose(vector[vocabulary.index("NP:Subj")], 1.0)


class TestPhraseFullSignaturePsalmVectors:
    def test_broadcasts_the_same_vector_to_every_colon(self):
        psalms = [
            _psalm(
                number=1,
                nodes=(100, 101),
                typ=(("NP",), ("NP",)),
                function=(("Subj",), ("Subj",)),
                det=(("det",), ("det",)),
            )
        ]
        external_counts = {"NP:Subj:det": 500}
        vocabulary = ("NP:Subj:det", "<RARE>")
        vectors = phrase_full_signature_psalm_vectors(psalms, vocabulary, external_counts, k=100)
        assert np.allclose(vectors[100], vectors[101])
