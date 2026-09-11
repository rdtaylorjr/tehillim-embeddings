from __future__ import annotations

import numpy as np
import pytest

from core.support import RARE_TOKEN
from syntactic.assignment import Assignment
from syntactic.clause_rela_vectorize import (
    clause_rela_1gram_vectors,
    clause_rela_columns,
    clause_signature_1gram_vectors,
    clause_signature_columns,
    safe_clause_mask,
)
from syntactic.corpus import ClausePsalm

COUNTS = {"NA": 50000, "Coor": 5000, "Resu": 5000, "ReVo": 5000, "NmCl:NA": 5000}
K = 1000


def _psalm(nodes, typ, rela, node_index, unit_index, weight):
    return ClausePsalm(
        number=1,
        half_verse_nodes=nodes,
        clause_nodes=tuple(range(len(rela))),
        clause_typ=typ,
        clause_rela=rela,
        clause_assignment=Assignment(
            node_index=np.array(node_index),
            unit_index=np.array(unit_index),
            weight=np.array(weight),
        ),
    )


class TestSafeClauseMask:
    def test_excludes_the_two_audited_resumption_relations(self):
        psalm = _psalm((10,), ("NmCl",) * 4, ("NA", "Resu", "Coor", "ReVo"), [0], [0], [1.0])

        assert safe_clause_mask(psalm).tolist() == [True, False, True, False]


class TestClauseRelaColumns:
    def test_a_firewalled_clause_never_reaches_a_colon_sequence(self):
        psalm = _psalm(
            (10,), ("NmCl",) * 3, ("NA", "Resu", "Coor"), [0, 1, 2], [0, 0, 0], [1.0, 1.0, 1.0]
        )

        assert clause_rela_columns(psalm, COUNTS, K) == (("NA", "Coor"),)

    def test_adding_resumption_clauses_cannot_change_the_surviving_sequence(self):
        """Removal, so a mass deficit cannot report the excluded clauses either."""
        plain = _psalm((10,), ("NmCl",) * 2, ("NA", "Coor"), [0, 1], [0, 0], [1.0, 1.0])
        padded = _psalm(
            (10,),
            ("NmCl",) * 4,
            ("NA", "Resu", "Coor", "ReVo"),
            [0, 1, 2, 3],
            [0, 0, 0, 0],
            [1.0] * 4,
        )

        assert clause_rela_columns(padded, COUNTS, K) == clause_rela_columns(plain, COUNTS, K)


class TestClauseSignatureColumns:
    def test_pairs_type_with_relation_before_collapsing(self):
        psalm = _psalm((10,), ("NmCl",), ("NA",), [0], [0], [1.0])

        assert clause_signature_columns(psalm, COUNTS, K) == (("NmCl:NA",),)

    def test_an_unsupported_signature_collapses_to_rare(self):
        psalm = _psalm((10,), ("Ptcp",), ("Objc",), [0], [0], [1.0])

        assert clause_signature_columns(psalm, COUNTS, K) == ((RARE_TOKEN,),)

    def test_the_firewall_applies_to_signatures_as_well_as_relations(self):
        psalm = _psalm((10,), ("NmCl", "NmCl"), ("NA", "Resu"), [0, 1], [0, 0], [1.0, 1.0])

        assert clause_signature_columns(psalm, COUNTS, K) == (("NmCl:NA",),)


class TestVectors:
    def test_a_relation_histogram_is_normalized_over_surviving_clauses_only(self):
        vocabulary = ("Coor", "NA", RARE_TOKEN)
        psalm = _psalm(
            (10,), ("NmCl",) * 3, ("NA", "Resu", "Coor"), [0, 1, 2], [0, 0, 0], [1.0, 1.0, 1.0]
        )

        vector = clause_rela_1gram_vectors([psalm], vocabulary, COUNTS, K)[10]

        assert vector.sum() == pytest.approx(1.0)
        assert vector[vocabulary.index("NA")] == pytest.approx(0.5)

    def test_a_colon_of_only_firewalled_clauses_is_the_zero_vector(self):
        vocabulary = ("Coor", "NA", RARE_TOKEN)
        psalm = _psalm((10,), ("NmCl", "NmCl"), ("Resu", "ReVo"), [0, 1], [0, 0], [1.0, 1.0])

        assert clause_rela_1gram_vectors([psalm], vocabulary, COUNTS, K)[10].sum() == 0.0

    def test_signature_vectors_cover_every_colon_node(self):
        vocabulary = ("NmCl:NA", RARE_TOKEN)
        psalm = _psalm((10, 11), ("NmCl",), ("NA",), [0], [0], [1.0])

        assert set(clause_signature_1gram_vectors([psalm], vocabulary, COUNTS, K)) == {10, 11}
