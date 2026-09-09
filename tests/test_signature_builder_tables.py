"""Holds the production and shuffle-null builder tables to one another so neither drifts."""

from __future__ import annotations

import pytest

from morphological import signature_vectorize as morphological_vectorize
from syntactic import signature_vectorize as syntactic_vectorize

MODULES = [morphological_vectorize, syntactic_vectorize]
IDS = ["morphological", "syntactic"]


@pytest.mark.parametrize("module", MODULES, ids=IDS)
def test_the_ordered_subset_covers_every_order_sensitive_dense_construction(module) -> None:
    """A construction the shuffle control cannot permute would go untested against its null."""
    expected = set(module.DENSE_BUILDERS) - set(module.ORDER_INVARIANT)

    assert set(module.ORDERED_DENSE_BUILDERS) == expected


@pytest.mark.parametrize("module", MODULES, ids=IDS)
def test_the_ordered_table_holds_the_same_builder_the_production_table_does(module) -> None:
    """A shuffle null must permute the very function production runs, not a parallel copy."""
    for construction, builder in module.ORDERED_DENSE_BUILDERS.items():
        assert builder is module.DENSE_BUILDERS[construction]


@pytest.mark.parametrize("module", MODULES, ids=IDS)
def test_every_order_invariant_construction_is_a_real_production_construction(module) -> None:
    assert set(module.ORDER_INVARIANT) <= set(module.DENSE_BUILDERS)


@pytest.mark.parametrize("module", MODULES, ids=IDS)
def test_the_sparse_table_is_disjoint_from_the_dense_one(module) -> None:
    """A construction in both tables would be written twice, in two layouts, to one path."""
    assert set(module.DENSE_BUILDERS) & set(module.SPARSE_BUILDERS) == set()
