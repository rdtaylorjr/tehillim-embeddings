from __future__ import annotations

from syntactic.clause_signature import build_clause_signature, clause_signatures


class TestBuildClauseSignature:
    def test_joins_type_and_relation_into_one_token(self):
        assert build_clause_signature(typ="NmCl", rela="Subj") == "NmCl:Subj"

    def test_keeps_the_na_relation_because_it_is_the_modal_value_not_a_missing_one(self):
        """78 percent of Psalms clauses are rela=NA, so dropping it would erase most of the data."""
        assert build_clause_signature(typ="xQt0", rela="NA") == "xQt0:NA"

    def test_two_clauses_differing_only_in_relation_get_different_signatures(self):
        assert build_clause_signature(typ="NmCl", rela="Subj") != build_clause_signature(
            typ="NmCl", rela="Attr"
        )


class TestClauseSignatures:
    def test_pairs_each_type_with_the_relation_at_the_same_position(self):
        result = clause_signatures(typ=("NmCl", "xQt0"), rela=("Subj", "NA"))

        assert result == ("NmCl:Subj", "xQt0:NA")

    def test_empty_sequences_produce_no_signatures(self):
        assert clause_signatures(typ=(), rela=()) == ()
