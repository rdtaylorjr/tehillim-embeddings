from __future__ import annotations

import pathlib

import pytest

from core.support import RARE_TOKEN, build_signature_vocabulary, load_external_signature_counts
from syntactic.clause_support import (
    MIN_EXTERNAL_SUPPORT_K_CLAUSE_ATOM_TYP,
    MIN_EXTERNAL_SUPPORT_K_CLAUSE_RELA,
    MIN_EXTERNAL_SUPPORT_K_CLAUSE_SIGNATURE,
    MIN_EXTERNAL_SUPPORT_K_CLAUSE_TYP,
)

CONFIG = pathlib.Path(__file__).resolve().parent.parent / "config"

FROZEN = (
    ("clause_atom_typ", MIN_EXTERNAL_SUPPORT_K_CLAUSE_ATOM_TYP, 43, 26),
    ("clause_typ", MIN_EXTERNAL_SUPPORT_K_CLAUSE_TYP, 42, 28),
    ("clause_rela", MIN_EXTERNAL_SUPPORT_K_CLAUSE_RELA, 13, 6),
    ("clause_signature", MIN_EXTERNAL_SUPPORT_K_CLAUSE_SIGNATURE, 234, 70),
)
IDS = [name for name, *_ in FROZEN]


@pytest.mark.parametrize(("name", "k", "n_external", "n_kept"), FROZEN, ids=IDS)
def test_the_committed_support_table_is_the_one_the_threshold_was_frozen_against(
    name, k, n_external, n_kept
):
    """A regenerated table that changed these counts invalidates the frozen K above it."""
    counts = load_external_signature_counts(CONFIG / f"{name}_external_support.csv")

    assert len(counts) == n_external
    assert sum(1 for count in counts.values() if count >= k) == n_kept


@pytest.mark.parametrize(("name", "k", "n_external", "n_kept"), FROZEN, ids=IDS)
def test_each_vocabulary_is_its_survivors_plus_one_rare_bin(name, k, n_external, n_kept):
    counts = load_external_signature_counts(CONFIG / f"{name}_external_support.csv")

    vocabulary = build_signature_vocabulary(counts, k)

    assert len(vocabulary) == n_kept + 1
    assert vocabulary[-1] == RARE_TOKEN
    assert len(set(vocabulary)) == len(vocabulary)


@pytest.mark.parametrize(("name", "k", "n_external", "n_kept"), FROZEN, ids=IDS)
def test_no_support_table_counts_anything_inside_psalms(name, k, n_external, n_kept):
    """The tables are the label-blind base, so a Psalms-only value must be absent from them."""
    counts = load_external_signature_counts(CONFIG / f"{name}_external_support.csv")

    assert all(count > 0 for count in counts.values())


def test_the_joint_signature_threshold_departs_from_the_inherited_thousand_deliberately():
    """A joint vocabulary at 1000 would put more mass in `<RARE>` than in any real signature."""
    counts = load_external_signature_counts(CONFIG / "clause_signature_external_support.csv")

    assert MIN_EXTERNAL_SUPPORT_K_CLAUSE_SIGNATURE < 1000
    assert sum(1 for c in counts.values() if c >= 1000) < len(
        build_signature_vocabulary(counts, MIN_EXTERNAL_SUPPORT_K_CLAUSE_SIGNATURE)
    )


def test_the_inherited_thresholds_match_the_phrase_and_morphology_families():
    assert MIN_EXTERNAL_SUPPORT_K_CLAUSE_ATOM_TYP == 1000
    assert MIN_EXTERNAL_SUPPORT_K_CLAUSE_RELA == 1000


def test_clause_level_typ_departs_because_it_is_a_different_distribution_from_clause_atom_typ():
    """1000 holds 80.5% of clause-level tokens against the clause-atom table's 91.5%."""
    counts = load_external_signature_counts(CONFIG / "clause_typ_external_support.csv")

    assert MIN_EXTERNAL_SUPPORT_K_CLAUSE_TYP == 500
    assert sum(1 for c in counts.values() if c >= 1000) < sum(
        1 for c in counts.values() if c >= MIN_EXTERNAL_SUPPORT_K_CLAUSE_TYP
    )
