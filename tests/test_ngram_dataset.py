"""One driver writes every n-gram feature's family, so each declaration is checked through it."""

from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq
import pytest

from core.export import dataset_path
from core.ngram import concatenated_1_2_3gram_dim
from core.ngram_dataset import order_sensitive_constructions
from morphological.corpus import MorphologicalPsalm
from morphological.generate_pos import DATASET as SP
from morphological.generate_pos import generate as generate_sp
from syntactic.corpus import PhrasePsalm
from syntactic.generate_function import DATASET as FUNCTION
from syntactic.generate_function import generate as generate_function
from syntactic.generate_typ import DATASET as TYP
from syntactic.generate_typ import generate as generate_typ

MORPHOLOGICAL = [
    MorphologicalPsalm(
        number=1, half_verse_nodes=(100, 101), half_verse_sp=(("subs", "verb"), ("prep",))
    ),
    MorphologicalPsalm(number=2, half_verse_nodes=(200,), half_verse_sp=(("conj",),)),
]
PHRASE = [
    PhrasePsalm(
        number=1,
        half_verse_nodes=(100, 101),
        half_verse_typ=(("NP", "VP"), ("PP",)),
        half_verse_function=(("Subj", "Pred"), ("Cmpl",)),
    ),
    PhrasePsalm(
        number=2,
        half_verse_nodes=(200,),
        half_verse_typ=(("CP",),),
        half_verse_function=(("Conj",),),
    ),
]

DATASETS = [
    pytest.param(SP, generate_sp, MORPHOLOGICAL, id="sp"),
    pytest.param(TYP, generate_typ, PHRASE, id="typ"),
    pytest.param(FUNCTION, generate_function, PHRASE, id="function"),
]


def _path(dataset, output_root, construction):
    return dataset_path(
        output_root,
        dataset.unit,
        construction,
        domain=dataset.domain,
        unit_key="feature",
        level=dataset.level,
    )


@pytest.mark.parametrize(("dataset", "generate", "psalms"), DATASETS)
def test_writes_the_declared_family_and_nothing_else(dataset, generate, psalms, tmp_path):
    written = generate(psalms, tmp_path)

    assert set(written) == {f"{dataset.unit}_{c}" for c in dataset.constructions}
    assert all(_path(dataset, tmp_path, c).exists() for c in dataset.constructions)


@pytest.mark.parametrize(("dataset", "generate", "psalms"), DATASETS)
def test_a_second_run_writes_nothing(dataset, generate, psalms, tmp_path):
    generate(psalms, tmp_path)

    assert generate(psalms, tmp_path) == []


@pytest.mark.parametrize(("dataset", "generate", "psalms"), DATASETS)
def test_a_psalm_construction_broadcasts_one_vector_across_its_nodes(
    dataset, generate, psalms, tmp_path
):
    generate(psalms, tmp_path)
    construction = next(c for c in dataset.constructions if c.endswith("_psalm"))

    table = pq.read_table(_path(dataset, tmp_path, construction))
    by_node = dict(zip(table["node_id"].to_pylist(), table["vector"].to_pylist(), strict=True))

    assert by_node[100] == by_node[101]


@pytest.mark.parametrize(("dataset", "generate", "psalms"), DATASETS)
def test_a_half_verse_construction_gives_each_node_its_own_vector(
    dataset, generate, psalms, tmp_path
):
    generate(psalms, tmp_path)
    construction = next(c for c in dataset.constructions if not c.endswith("_psalm"))

    table = pq.read_table(_path(dataset, tmp_path, construction))
    by_node = dict(zip(table["node_id"].to_pylist(), table["vector"].to_pylist(), strict=True))

    assert by_node[100] != by_node[101]


@pytest.mark.parametrize(("dataset", "generate", "psalms"), DATASETS)
def test_only_declared_constructions_are_written(dataset, generate, psalms, tmp_path):
    """The declaration is the whole set, so an undeclared name cannot reach the writer."""
    generate(psalms, tmp_path)

    written = {
        path.parent.name.removeprefix("construction=") for path in tmp_path.rglob("*.parquet")
    }

    assert written == set(dataset.constructions)


class TestOrderSensitiveConstructions:
    """The registry reads a feature's order-sensitive constructions off its declaration."""

    def test_only_constructions_reading_a_bigram_or_higher_are_returned(self) -> None:
        names = [name for name, _ in order_sensitive_constructions(SP)]

        assert names == ["1_2gram", "1_2_3gram", "1_2gram_psalm", "1_2_3gram_psalm"]

    def test_a_sparse_construction_carries_its_dense_width(self) -> None:
        by_name = dict(order_sensitive_constructions(TYP))

        assert by_name["1_2gram"].sparse_dim is None
        assert by_name["1_2_3gram"].sparse_dim == concatenated_1_2_3gram_dim(len(TYP.vocabulary))

    def test_each_builder_accepts_the_permutation_a_draw_passes(self) -> None:
        """A builder the registry cannot hand an order to would score a null against itself."""
        psalm = MorphologicalPsalm(
            number=1, half_verse_nodes=(10,), half_verse_sp=(("subs", "verb", "subs"),)
        )
        build = dict(order_sensitive_constructions(SP))["1_2gram"].build

        unpermuted = build([psalm], order_by_node=None)
        permuted = build([psalm], order_by_node={10: np.array([2, 0, 1])})

        assert not np.array_equal(unpermuted[10], permuted[10])
