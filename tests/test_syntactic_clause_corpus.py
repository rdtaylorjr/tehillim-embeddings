from __future__ import annotations

from functools import cache

import numpy as np
import pytest

from syntactic.assignment import containment_mask
from syntactic.corpus import CLAUSE, PHRASE, Corpus, clause_corpus


class TestClauseCorpusLoad:
    def test_requests_the_clause_features_and_not_the_phrase_only_ones(self, tmp_path):
        seen: dict[str, object] = {}

        def _loader(path, features):
            seen["path"], seen["features"] = path, features
            return "api"

        corpus = Corpus.load(CLAUSE, tmp_path / "missing", loader=_loader)

        requested = str(seen["features"]).split()
        assert {"otype", "typ", "kind", "rela", "code", "tab", "is_root"} <= set(requested)
        assert "function" not in requested
        assert corpus.api == "api"

    def test_sharing_reuses_a_loaded_api_so_two_levels_cost_one_bhsa_load(self):
        loads: list[int] = []

        def _loader(path, features):
            loads.append(1)
            return "api"

        phrase = Corpus.load(PHRASE, None, loader=_loader)
        clause = Corpus.sharing(phrase.api, CLAUSE)

        assert len(loads) == 1
        assert clause.api is phrase.api


@cache
def _clause_psalms():
    """Loads BHSA once for every integration test in this module."""
    return tuple(clause_corpus().psalms())


@pytest.mark.integration
class TestClauseExtractionAgainstBhsa:
    @property
    def psalms(self):
        return _clause_psalms()

    def test_extracts_all_150_psalms_in_canonical_order(self):
        psalms = self.psalms
        assert [p.number for p in psalms] == list(range(1, 151))

    def test_node_counts_match_the_measured_bhsa_psalms_totals(self):
        psalms = self.psalms
        assert sum(len(p.half_verse_nodes) for p in psalms) == 5203
        assert sum(len(p.clause_nodes) for p in psalms) == 7285
        assert sum(len(p.clause_atom_nodes) for p in psalms) == 7462

    def test_every_feature_sequence_aligns_with_its_node_sequence(self):
        psalms = self.psalms
        for psalm in psalms:
            n_atoms = len(psalm.clause_atom_nodes)
            for sequence in (
                psalm.clause_atom_typ,
                psalm.clause_atom_code,
                psalm.clause_atom_tab,
                psalm.clause_atom_is_root,
                psalm.clause_atom_mother,
            ):
                assert len(sequence) == n_atoms
            n_clauses = len(psalm.clause_nodes)
            for sequence in (
                psalm.clause_typ,
                psalm.clause_kind,
                psalm.clause_rela,
                psalm.clause_dist,
                psalm.clause_dist_unit,
            ):
                assert len(sequence) == n_clauses

    def test_clause_containment_matches_the_measured_spread(self):
        psalms = self.psalms
        spanned = [
            int((assignment.node_index == node).sum())
            for psalm in psalms
            for assignment in (psalm.clause_assignment,)
            for node in range(len(psalm.clause_nodes))
        ]
        assert sum(1 for s in spanned if s == 1) == 7031
        assert sum(1 for s in spanned if s == 2) == 250
        assert sum(1 for s in spanned if s == 3) == 4

    def test_clause_atom_containment_matches_the_measured_spread(self):
        psalms = self.psalms
        spanned = [
            int((assignment.node_index == node).sum())
            for psalm in psalms
            for assignment in (psalm.clause_atom_assignment,)
            for node in range(len(psalm.clause_atom_nodes))
        ]
        assert sum(1 for s in spanned if s == 1) == 7240
        assert sum(1 for s in spanned if s == 2) == 219
        assert sum(1 for s in spanned if s == 3) == 3

    def test_containment_only_leaves_the_measured_empty_colon_population(self):
        psalms = self.psalms

        def empty_colons(nodes_attr, assignment_attr):
            total = 0
            for psalm in psalms:
                assignment = getattr(psalm, assignment_attr)
                covered = set(assignment.unit_index[containment_mask(assignment)].tolist())
                total += len(psalm.half_verse_nodes) - len(covered)
            return total

        assert empty_colons("clause_nodes", "clause_assignment") == 334
        assert empty_colons("clause_atom_nodes", "clause_atom_assignment") == 310

    def test_weighted_assignment_leaves_no_empty_colon(self):
        psalms = self.psalms
        for psalm in psalms:
            for attr in ("clause_assignment", "clause_atom_assignment"):
                covered = set(getattr(psalm, attr).unit_index.tolist())
                assert covered == set(range(len(psalm.half_verse_nodes)))

    def test_weights_conserve_one_unit_of_mass_per_node(self):
        psalms = self.psalms
        for psalm in psalms:
            for nodes_attr, attr in (
                ("clause_nodes", "clause_assignment"),
                ("clause_atom_nodes", "clause_atom_assignment"),
            ):
                assignment = getattr(psalm, attr)
                mass = np.bincount(
                    assignment.node_index,
                    weights=assignment.weight,
                    minlength=len(getattr(psalm, nodes_attr)),
                )
                assert mass == pytest.approx(np.ones_like(mass))

    def test_clause_atom_mother_coverage_matches_the_measured_rate(self):
        psalms = self.psalms
        mothers = [m for p in psalms for m in p.clause_atom_mother]
        assert sum(1 for m in mothers if m is not None) == 7184

    def test_a_real_half_verses_clause_atoms_match_a_manual_tf_query(self):
        corpus = clause_corpus()
        psalm_1 = next(p for p in corpus.psalms() if p.number == 1)
        api = corpus.api
        first_colon = psalm_1.half_verse_nodes[0]

        manual = tuple(api.L.d(first_colon, otype="clause_atom"))
        assignment = psalm_1.clause_atom_assignment
        selected = assignment.node_index[
            containment_mask(assignment) & (assignment.unit_index == 0)
        ]
        extracted = tuple(psalm_1.clause_atom_nodes[i] for i in selected.tolist())

        assert extracted == manual

    def test_clause_kind_values_are_the_three_bhsa_classes(self):
        psalms = self.psalms
        observed = {value for p in psalms for value in p.clause_kind}
        assert observed == {"VC", "NC", "WP"}

    def test_clause_rela_never_carries_a_phrase_level_parallelism_marker(self):
        psalms = self.psalms
        observed = {value for p in psalms for value in p.clause_rela}
        assert "Para" not in observed
        assert observed <= {
            "Adju",
            "Attr",
            "Cmpl",
            "Coor",
            "NA",
            "Objc",
            "PrAd",
            "PreC",
            "ReVo",
            "Resu",
            "RgRc",
            "Spec",
            "Subj",
        }

    def test_the_parallel_clause_atom_codes_are_present_and_therefore_need_the_firewall(self):
        psalms = self.psalms
        codes = [c for p in psalms for c in p.clause_atom_code]
        assert sum(1 for c in codes if c == 200) == 782
        assert sum(1 for c in codes if c == 201) == 166
