"""A shuffle-null must land beside the dataset it controls for, or it controls for nothing."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.export import dataset_path
from core.shuffle import shuffle_construction_name
from morphological import generate_deploy as morphological_production
from morphological.scripts import generate_deploy_shuffle_control as morphological_control
from syntactic import generate_deploy as syntactic_production
from syntactic.scripts import generate_deploy_shuffle_control as syntactic_control

#: Each domain names its own Hive unit, so the pair is compared through that domain's name.
PAIRS = [
    ("morphological", "SUFFIX_UNIT", morphological_production, morphological_control),
    ("syntactic", "SIGNATURE_UNIT", syntactic_production, syntactic_control),
]
IDS = ["morphological", "syntactic"]


@pytest.mark.parametrize(("domain", "unit_attribute", "production", "control"), PAIRS, ids=IDS)
def test_the_control_names_the_same_unit_as_the_dataset_it_controls_for(
    domain: str, unit_attribute: str, production: object, control: object
) -> None:
    assert getattr(control, unit_attribute) == getattr(production, unit_attribute)


@pytest.mark.parametrize(("domain", "unit_attribute", "production", "control"), PAIRS, ids=IDS)
def test_the_control_names_the_same_construction(
    domain: str, unit_attribute: str, production: object, control: object
) -> None:
    assert control.CONSTRUCTION == production.CONSTRUCTION


@pytest.mark.parametrize(("domain", "unit_attribute", "production", "control"), PAIRS, ids=IDS)
def test_the_control_writes_under_the_same_domain(
    domain: str, unit_attribute: str, production: object, control: object
) -> None:
    assert control.DATASET_TYPE == production.DATASET_TYPE == domain


@pytest.mark.parametrize(("domain", "unit_attribute", "production", "control"), PAIRS, ids=IDS)
def test_a_seeded_control_differs_from_the_real_dataset_only_by_its_construction(
    domain: str, unit_attribute: str, production: object, control: object
) -> None:
    """The two paths must share every partition tier but the construction the seed names."""
    root = Path("/out")
    real = dataset_path(
        root,
        getattr(production, unit_attribute),
        production.CONSTRUCTION,
        domain=production.DATASET_TYPE,
        unit_key="feature",
    )
    seeded = dataset_path(
        root,
        getattr(control, unit_attribute),
        shuffle_construction_name(control.CONSTRUCTION, 1),
        domain=control.DATASET_TYPE,
        unit_key="feature",
    )

    assert real.parent.parent == seeded.parent.parent
