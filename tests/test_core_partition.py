"""The tree's grammar: one order of keys, each domain's own keys, paths and parses as inverses."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from core.partition import (
    BHSA_HALF_VERSE,
    KEYS,
    Partition,
    PartitionError,
    Scope,
    family,
)

SCROLL = Scope("dss", "verse", witness="11Q5", reconstruction="none")


class TestScope:
    def test_a_single_text_corpus_takes_no_witness_or_reconstruction(self) -> None:
        with pytest.raises(PartitionError, match="one text"):
            Scope("bhsa", "half_verse", witness="L")

    def test_a_multi_witness_corpus_requires_both(self) -> None:
        with pytest.raises(PartitionError, match="witness and reconstruction required"):
            Scope("dss", "verse", witness="11Q5")

    def test_the_scope_directory_holds_the_keys_above_domain_in_order(self) -> None:
        assert BHSA_HALF_VERSE.directory == "corpus=bhsa/unit=half_verse"
        assert SCROLL.directory == "corpus=dss/witness=11Q5/reconstruction=none/unit=verse"


class TestPartition:
    def test_writes_its_segments_in_tree_order(self) -> None:
        partition = Partition(
            BHSA_HALF_VERSE, "syntactic", level="clause", feature="typ", construction="1gram"
        )
        assert partition.directory == (
            "corpus=bhsa/unit=half_verse/domain=syntactic/level=clause/feature=typ/"
            "construction=1gram"
        )

    def test_a_scroll_partition_carries_its_witness_and_reconstruction(self) -> None:
        partition = Partition(SCROLL, "semantic", model="berel", text="consonantal")
        assert partition.directory == (
            "corpus=dss/witness=11Q5/reconstruction=none/unit=verse/domain=semantic/"
            "model=berel/text=consonantal"
        )

    def test_the_file_sits_under_the_directory(self, tmp_path: Path) -> None:
        partition = Partition(BHSA_HALF_VERSE, "lexical", type="lexeme", construction="icf")
        assert partition.file(tmp_path) == tmp_path / partition.directory / "part-0.parquet"

    @pytest.mark.parametrize(
        ("domain", "keys", "message"),
        [
            ("lexical", {"feature": "sp", "construction": "icf"}, "missing \\['type'\\]"),
            ("morphological", {"type": "sp", "construction": "icf"}, "foreign \\['type'\\]"),
            ("syntactic", {"feature": "typ", "construction": "icf"}, "missing \\['level'\\]"),
            ("semantic", {"model": "berel"}, "missing \\['text'\\]"),
            ("semantic", {"model": "berel", "text": "x", "construction": "y"}, "foreign"),
            ("phonology", {"model": "berel", "text": "x"}, "not one of"),
        ],
    )
    def test_each_domain_takes_exactly_its_own_keys(
        self, domain: str, keys: dict[str, str], message: str
    ) -> None:
        with pytest.raises(PartitionError, match=message):
            Partition(BHSA_HALF_VERSE, domain, **keys)

    def test_the_lexical_text_tier_belongs_to_the_surface_type_only(self) -> None:
        with pytest.raises(PartitionError, match="type=word only"):
            Partition(BHSA_HALF_VERSE, "lexical", type="lexeme", text="vocalized", construction="c")
        with pytest.raises(PartitionError, match="type=word only"):
            Partition(BHSA_HALF_VERSE, "lexical", type="word", construction="c")

    def test_the_identifier_is_the_values_below_domain(self) -> None:
        partition = Partition(
            BHSA_HALF_VERSE, "lexical", type="word", text="vocalized", construction="icf"
        )
        assert partition.identifier == "word_vocalized_icf"
        assert Partition(SCROLL, "semantic", model="berel", text="consonantal").identifier == (
            "berel_consonantal"
        )

    def test_the_metadata_is_every_present_key(self) -> None:
        partition = Partition(SCROLL, "semantic", model="berel", text="consonantal")
        assert partition.metadata() == {
            "corpus": "dss",
            "witness": "11Q5",
            "reconstruction": "none",
            "unit": "verse",
            "domain": "semantic",
            "model": "berel",
            "text": "consonantal",
        }


class TestParse:
    @pytest.mark.parametrize(
        "partition",
        [
            Partition(BHSA_HALF_VERSE, "lexical", type="homograph", construction="binary"),
            Partition(
                BHSA_HALF_VERSE, "lexical", type="word", text="cantillation", construction="c"
            ),
            Partition(BHSA_HALF_VERSE, "morphological", feature="morph_gn", construction="1gram"),
            Partition(
                BHSA_HALF_VERSE, "syntactic", level="phrase", feature="typ", construction="c"
            ),
            Partition(BHSA_HALF_VERSE, "semantic", model="berel", text="consonantal"),
            Partition(Scope("bhsa", "verse"), "semantic", model="berel", text="consonantal"),
            Partition(SCROLL, "semantic", model="berel", text="consonantal"),
        ],
    )
    def test_parse_inverts_the_path_from_a_file_or_a_directory(
        self, partition: Partition, tmp_path: Path
    ) -> None:
        assert Partition.parse(partition.file(tmp_path)) == partition
        assert Partition.parse(partition.directory) == partition

    def test_parse_reads_a_file_named_otherwise_than_the_part(self) -> None:
        path = "corpus=bhsa/unit=half_verse/domain=semantic/model=berel/text=consonantal/x.parquet"
        assert Partition.parse(path).model == "berel"

    def test_refuses_a_path_with_no_partition(self) -> None:
        with pytest.raises(PartitionError, match="names no dataset"):
            Partition.parse("data/stray.parquet")

    def test_refuses_a_key_outside_the_tree(self) -> None:
        with pytest.raises(PartitionError, match="not a key of the tree"):
            Partition.parse("corpus=bhsa/unit=half_verse/domain=lexical/vocab=lexeme")

    def test_refuses_a_key_out_of_order(self) -> None:
        with pytest.raises(PartitionError, match="out of tree order"):
            Partition.parse("unit=half_verse/corpus=bhsa/domain=semantic/model=b/text=c")

    def test_refuses_a_repeated_key(self) -> None:
        with pytest.raises(PartitionError, match="appears twice"):
            Partition.parse("corpus=bhsa/corpus=bhsa/unit=half_verse/domain=semantic")

    def test_refuses_a_scope_without_its_corpus(self) -> None:
        with pytest.raises(PartitionError, match="missing 'corpus'"):
            Partition.parse("unit=half_verse/domain=semantic/model=b/text=c")

    def test_from_mapping_reads_metadata_back(self) -> None:
        partition = Partition(SCROLL, "semantic", model="berel", text="consonantal")
        assert Partition.from_mapping(partition.metadata()) == partition

    def test_from_mapping_refuses_an_unknown_key(self) -> None:
        with pytest.raises(PartitionError, match="unknown keys"):
            Partition.from_mapping({"corpus": "bhsa", "unit": "verse", "domain": "x", "k": "v"})


class TestFamily:
    def test_one_partition_per_construction_in_declared_order(self) -> None:
        partitions = family(
            BHSA_HALF_VERSE, "syntactic", ("1gram", "1gram_psalm"), level="phrase", feature="typ"
        )
        assert [p.construction for p in partitions] == ["1gram", "1gram_psalm"]
        assert {p.feature for p in partitions} == {"typ"}

    def test_a_domain_without_constructions_is_one_partition(self) -> None:
        (only,) = family(BHSA_HALF_VERSE, "semantic", model="berel", text="consonantal")
        assert only.directory.endswith("domain=semantic/model=berel/text=consonantal")

    def test_refuses_a_repeated_construction(self) -> None:
        with pytest.raises(PartitionError, match="duplicate"):
            family(BHSA_HALF_VERSE, "lexical", ("icf", "icf"), type="lexeme")


def test_every_key_appears_once_in_the_declared_order() -> None:
    assert len(set(KEYS)) == len(KEYS)
    assert KEYS[0] == "corpus"
    assert KEYS.index("unit") < KEYS.index("domain") < KEYS.index("construction")


def test_every_file_in_the_tree_parses_and_reproduces_its_path() -> None:
    """Parity: the on-disk tree is exactly what the grammar writes, file by file."""
    root = Path(__file__).resolve().parents[1] / "data"
    if not root.exists():
        pytest.skip("no generated tree in this checkout")
    for path in root.rglob("part-0.parquet"):
        relative = path.relative_to(root)
        assert Partition.parse(relative).file(root) == path


def test_every_file_in_the_tree_carries_the_keys_its_path_spells() -> None:
    """Parity: a file's metadata and its partition agree, so a moved file still names itself."""
    root = Path(__file__).resolve().parents[1] / "data"
    if not root.exists():
        pytest.skip("no generated tree in this checkout")
    for path in root.rglob("part-0.parquet"):
        partition = Partition.parse(path.relative_to(root))
        metadata = {k.decode(): v.decode() for k, v in pq.read_schema(path).metadata.items()}
        assert Partition.from_mapping({k: metadata[k] for k in partition.metadata()}) == partition
