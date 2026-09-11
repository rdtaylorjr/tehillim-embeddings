"""A construction that reads order must have a shuffle-null family, or its null cannot be run."""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import morphological
import syntactic
from core.ngram_dataset import NgramDataset
from core.supported_dataset import SupportedDataset
from families.shuffle import FAMILIES


def _declarations() -> list[pytest.param]:
    """Every dataset declaration in the tree, found by walking the domain packages."""
    found = []
    for package in (morphological, syntactic):
        for module_info in pkgutil.iter_modules(package.__path__):
            module = importlib.import_module(f"{package.__name__}.{module_info.name}")
            dataset = getattr(module, "DATASET", None)
            if isinstance(dataset, (NgramDataset, SupportedDataset)):
                level = f"/{dataset.level}" if dataset.level else ""
                prefix = f"{dataset.domain}{level}/{dataset.unit}"
                found.append(pytest.param(dataset, prefix, id=dataset.unit))
    return found


DECLARATIONS = _declarations()


def _is_unigram(name: str) -> bool:
    """A supported family names its orders, so its unigram is the one construction so named."""
    return name.split("_psalm", maxsplit=1)[0] == "1gram"


def _reads_order(dataset: NgramDataset | SupportedDataset) -> set[str]:
    """Constructions concatenating a bigram or higher, which depend on within-colon order."""
    if isinstance(dataset, NgramDataset):
        return {name for name, orders in dataset.constructions.items() if max(orders) >= 2}
    return {name for name in (*dataset.dense, *dataset.sparse) if not _is_unigram(name)}


def _ignores_order(dataset: NgramDataset | SupportedDataset) -> set[str]:
    """Unigram histograms, which count without reading order."""
    if isinstance(dataset, NgramDataset):
        return {name for name, orders in dataset.constructions.items() if max(orders) == 1}
    return {name for name in (*dataset.dense, *dataset.sparse) if _is_unigram(name)}


def test_every_declaration_in_the_tree_is_found() -> None:
    """A walk that found nothing would pass the tests below vacuously."""
    assert len(DECLARATIONS) >= 7


@pytest.mark.parametrize(("dataset", "prefix"), DECLARATIONS)
def test_every_order_reading_construction_has_a_family(dataset, prefix):
    """A construction concatenating a bigram or higher depends on order, so it needs a null."""
    missing = {name for name in _reads_order(dataset) if f"{prefix}/{name}" not in FAMILIES}

    assert missing == set()


@pytest.mark.parametrize(("dataset", "prefix"), DECLARATIONS)
def test_no_order_invariant_construction_claims_a_family(dataset, prefix):
    """A unigram histogram ignores order, so a shuffle null over it would be degenerate."""
    claimed = {name for name in _ignores_order(dataset) if f"{prefix}/{name}" in FAMILIES}

    assert claimed == set()
