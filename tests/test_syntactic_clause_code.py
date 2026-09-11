from __future__ import annotations

from itertools import pairwise

import pytest

from syntactic.clause_code import (
    CODE_BANDS,
    FORBIDDEN_FOR_PARALLELISM,
    SHIPPABLE_SCOPES,
    TARGET_ADJACENT,
    Scope,
    band_of,
    band_vocabulary,
    retained_codes,
)


class TestBandOf:
    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            (0, "no_relation"),
            (10, "relative"),
            (16, "relative"),
            (50, "infinitive_construct"),
            (74, "infinitive_construct"),
            (100, "asyndetic"),
            (167, "asyndetic"),
            (200, "parallel"),
            (201, "parallel"),
            (220, "defective"),
            (223, "defective"),
            (300, "conjunctive_adverb"),
            (400, "coordinate"),
            (500, "postulational"),
            (600, "conditional"),
            (700, "temporal"),
            (800, "final"),
            (900, "causal"),
            (999, "direct_speech"),
        ],
    )
    def test_maps_each_documented_bhsa_range_to_its_band(self, code, expected):
        assert band_of(code) == expected

    def test_a_code_outside_every_documented_range_is_rejected_rather_than_guessed(self):
        with pytest.raises(ValueError, match="no documented band"):
            band_of(1234)

    def test_the_bands_cover_disjoint_ranges(self):
        for earlier, later in pairwise(CODE_BANDS):
            assert earlier.high < later.low


class TestProvenanceBands:
    def test_the_parallel_band_is_the_forbidden_one(self):
        assert set(FORBIDDEN_FOR_PARALLELISM) == {"parallel"}

    def test_the_defective_band_is_target_adjacent_because_gapping_tracks_parallelism(self):
        assert set(TARGET_ADJACENT) == {"defective"}

    def test_the_two_provenance_sets_do_not_overlap(self):
        assert not FORBIDDEN_FOR_PARALLELISM & TARGET_ADJACENT


class TestBandVocabulary:
    def test_the_parallelism_vocabulary_omits_the_forbidden_and_target_adjacent_bands(self):
        vocabulary = band_vocabulary(Scope.PARALLELISM)

        assert "parallel" not in vocabulary
        assert "defective" not in vocabulary

    def test_the_parallelism_vocabulary_keeps_every_other_documented_band(self):
        vocabulary = set(band_vocabulary(Scope.PARALLELISM))
        every = {band.name for band in CODE_BANDS}

        assert vocabulary == every - FORBIDDEN_FOR_PARALLELISM - TARGET_ADJACENT

    def test_the_genre_vocabulary_keeps_every_band_because_it_scores_a_different_target(self):
        assert set(band_vocabulary(Scope.GENRE)) == {band.name for band in CODE_BANDS}

    def test_a_vocabulary_is_sorted_so_a_vector_dimension_is_stable_across_runs(self):
        for scope in Scope:
            vocabulary = band_vocabulary(scope)
            assert list(vocabulary) == sorted(vocabulary)


class TestRetainedCodes:
    def test_parallelism_scope_drops_the_parallel_and_defective_atoms_entirely(self):
        codes = (0, 200, 201, 220, 400)

        assert retained_codes(codes, Scope.PARALLELISM) == (0, 400)

    def test_adding_parallel_atoms_cannot_change_what_parallelism_scope_retains(self):
        """Dropping the dimension is not enough: a mass deficit would still signal presence."""
        without = retained_codes((0, 400, 999), Scope.PARALLELISM)
        with_parallel = retained_codes((0, 200, 400, 201, 999, 200), Scope.PARALLELISM)

        assert without == with_parallel

    def test_genre_scope_retains_everything(self):
        codes = (0, 200, 220, 400)

        assert retained_codes(codes, Scope.GENRE) == codes


class TestShippableScopes:
    def test_the_code_family_ships_for_genre_only(self):
        """Excluding the target construct costs 9.07 percent of colons, so parallelism is out."""
        assert SHIPPABLE_SCOPES == (Scope.GENRE,)

    def test_parallelism_is_deliberately_absent_rather_than_merely_unlisted(self):
        assert Scope.PARALLELISM not in SHIPPABLE_SCOPES


@pytest.mark.integration
def test_the_measured_cost_that_makes_the_code_family_genre_only_still_holds():
    """If BHSA changes this cost, the genre-only decision above needs revisiting."""
    from syntactic.corpus import clause_corpus

    psalms = clause_corpus().psalms()
    emptied = 0
    for psalm in psalms:
        assignment = psalm.clause_atom_assignment
        kept = {
            index
            for index, code in enumerate(psalm.clause_atom_code)
            if band_of(code) not in FORBIDDEN_FOR_PARALLELISM | TARGET_ADJACENT
        }
        for colon in range(len(psalm.half_verse_nodes)):
            touching = set(assignment.node_index[assignment.unit_index == colon].tolist())
            if touching and not touching & kept:
                emptied += 1

    assert emptied == 472
    assert sum(1 for p in psalms for c in p.clause_atom_code if band_of(c) == "parallel") == 948
