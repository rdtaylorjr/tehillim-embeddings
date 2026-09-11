from __future__ import annotations

import pytest

from syntactic.clause_code import Scope
from syntactic.clause_rela import (
    TARGET_ADJACENT_RELA,
    retained_rela_indices,
    safe_rela_vocabulary,
)

FULL = ("NA", "Coor", "Attr", "Adju", "ReVo", "Objc", "Resu", "RgRc", "Subj")


class TestTargetAdjacentRela:
    def test_holds_the_two_resumption_relations_the_audit_flagged(self):
        assert set(TARGET_ADJACENT_RELA) == {"Resu", "ReVo"}


class TestSafeRelaVocabulary:
    def test_parallelism_scope_drops_both_resumption_relations(self):
        assert safe_rela_vocabulary(FULL, Scope.PARALLELISM) == (
            "NA",
            "Coor",
            "Attr",
            "Adju",
            "Objc",
            "RgRc",
            "Subj",
        )

    def test_genre_scope_keeps_them_because_it_scores_a_different_target(self):
        assert safe_rela_vocabulary(FULL, Scope.GENRE) == FULL

    def test_the_surviving_order_is_the_input_order_so_dimensions_stay_stable(self):
        safe = safe_rela_vocabulary(FULL, Scope.PARALLELISM)

        assert list(safe) == [value for value in FULL if value in safe]


class TestRetainedRelaIndices:
    def test_parallelism_scope_drops_the_positions_carrying_an_excluded_relation(self):
        assert retained_rela_indices(("NA", "Resu", "Coor", "ReVo"), Scope.PARALLELISM) == (0, 2)

    def test_adding_excluded_clauses_cannot_change_which_relations_survive(self):
        """Dropping the dimension alone would leave a mass deficit that still signals presence."""

        def survivors(rela):
            return tuple(rela[i] for i in retained_rela_indices(rela, Scope.PARALLELISM))

        assert survivors(("NA", "Resu", "Coor", "ReVo", "Resu")) == survivors(("NA", "Coor"))

    def test_genre_scope_retains_every_position(self):
        rela = ("NA", "Resu", "ReVo")

        assert retained_rela_indices(rela, Scope.GENRE) == (0, 1, 2)

    def test_a_clause_sequence_of_only_excluded_relations_retains_nothing(self):
        assert retained_rela_indices(("Resu", "ReVo"), Scope.PARALLELISM) == ()


@pytest.mark.integration
def test_the_firewall_empties_the_measured_and_disclosed_colon_population():
    """Removal costs these colons, which population disclosure reports rather than hides."""
    from syntactic.corpus import clause_corpus

    emptied = 0
    for psalm in clause_corpus().psalms():
        assignment = psalm.clause_assignment
        kept = set(retained_rela_indices(psalm.clause_rela, Scope.PARALLELISM))
        for colon in range(len(psalm.half_verse_nodes)):
            touching = assignment.node_index[assignment.unit_index == colon]
            if touching.size and not (set(touching.tolist()) & kept):
                emptied += 1
    #: 0.42 percent, affordable where the code firewall's 9.07 percent was not.
    assert emptied == 22
