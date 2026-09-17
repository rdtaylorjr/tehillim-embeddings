"""Unit tests for generate.py: every external dependency is injected, no network or model loads."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pyarrow.parquet as pq
import pytest

from core.partition import BHSA_HALF_VERSE, Scope
from semantic.export import partition
from semantic.generate import SPECS, encoder_for, generate, main, resource_for, tiers_for
from semantic.registry import MODEL_REGISTRY, variations_for_model
from semantic.sources import SOURCES, Source, source_for
from semantic.units import Unit

SCROLL = Scope("dss", "verse", witness="11Q5", reconstruction="none")
BHSA = source_for(BHSA_HALF_VERSE)
SCROLL_SOURCE = source_for(SCROLL)

UNITS = [
    Unit(100, {"consonantal": "a", "vocalized": "n", "cantillation": "A"}),
    Unit(101, {"consonantal": "b", "vocalized": "m", "cantillation": "B"}),
]
SCROLL_UNITS = [Unit(1857557, {"consonantal": "x"}), Unit(1857567, {"consonantal": "y"})]


def _recording_encoder(calls: list[tuple[str, list[str]]], width: int = 2):
    """An encoder factory that records which model saw which texts."""

    def open_encoder(slug: str):
        def encode(texts):
            calls.append((slug, list(texts)))
            return np.zeros((len(texts), width))

        return nullcontext(encode)

    return open_encoder


class TestGenerate:
    def test_writes_every_tier_of_a_diacritic_preserving_model(self, tmp_path):
        calls: list[tuple[str, list[str]]] = []

        written = generate(UNITS, tmp_path, BHSA, ["bge-m3"], encoder=_recording_encoder(calls))

        assert written == [
            "semantic_bge_m3_consonantal",
            "semantic_bge_m3_vocalized",
            "semantic_bge_m3_cantillation",
        ]
        assert calls == [("bge-m3", ["a", "b"]), ("bge-m3", ["n", "m"]), ("bge-m3", ["A", "B"])]
        for tier in ("consonantal", "vocalized", "cantillation"):
            assert partition("bge_m3", tier).file(tmp_path).exists()

    def test_writes_only_the_consonantal_tier_of_a_diacritic_stripping_model(self, tmp_path):
        calls: list[tuple[str, list[str]]] = []

        written = generate(UNITS, tmp_path, BHSA, ["miqrabert"], encoder=_recording_encoder(calls))

        assert written == ["semantic_miqrabert_consonantal"]
        assert calls == [("miqrabert", ["a", "b"])]

    def test_writes_a_scroll_source_under_its_own_scope_in_its_one_tier(self, tmp_path):
        calls: list[tuple[str, list[str]]] = []

        written = generate(
            SCROLL_UNITS, tmp_path, SCROLL_SOURCE, ["bge-m3"], encoder=_recording_encoder(calls)
        )

        assert written == ["semantic_bge_m3_consonantal"]
        assert calls == [("bge-m3", ["x", "y"])]
        path = partition("bge_m3", "consonantal", SCROLL).file(tmp_path)
        table = pq.read_table(path)
        assert table["node_id"].to_pylist() == [1857557, 1857567]
        assert table.schema.metadata[b"witness"] == b"11Q5"
        assert table.schema.metadata[b"reconstruction"] == b"none"

    def test_restricts_to_a_single_named_variation(self, tmp_path):
        calls: list[tuple[str, list[str]]] = []

        written = generate(
            UNITS,
            tmp_path,
            BHSA,
            ["bge-m3"],
            variation="vocalized",
            encoder=_recording_encoder(calls),
        )

        assert written == ["semantic_bge_m3_vocalized"]
        assert len(calls) == 1

    def test_opens_the_model_once_per_cell_and_not_at_all_when_nothing_is_missing(self, tmp_path):
        opened: list[str] = []

        def open_encoder(slug: str):
            opened.append(slug)
            return nullcontext(lambda texts: np.zeros((len(texts), 2)))

        generate(UNITS, tmp_path, BHSA, ["bge-m3"], encoder=open_encoder)
        assert opened == ["bge-m3"]

        written = generate(UNITS, tmp_path, BHSA, ["bge-m3"], encoder=open_encoder)
        assert written == []
        assert opened == ["bge-m3"]

    def test_a_tier_the_source_lacks_is_never_asked_of_the_encoder(self):
        """A scroll has consonants only, so a model's other tiers write nothing for it."""
        assert [tier for tier, _ in tiers_for("bge-m3", SCROLL_SOURCE)] == ["consonantal"]
        assert [tier for tier, _ in tiers_for("bge-m3", BHSA)] == [
            "consonantal",
            "vocalized",
            "cantillation",
        ]


class TestEncoderFor:
    def test_a_hosted_model_reads_its_key_and_binds_it_to_the_fetcher(self):
        with encoder_for("gemini", env={"TEHILLIM_OPENROUTER_API_KEY": "k"}) as encode:
            assert encode.keywords == {"api_key": "k"}

    def test_a_missing_key_names_the_exact_env_var_when_the_encoder_is_opened(self):
        with (
            pytest.raises(RuntimeError, match="TEHILLIM_OPENROUTER_API_KEY"),
            encoder_for("gemini", env={}),
        ):
            pass

    def test_cohere_uses_a_separate_api_key_env_var(self):
        with encoder_for("cohere", env={"TEHILLIM_COHERE_API_KEY": "c"}) as encode:
            assert encode.keywords == {"api_key": "c"}

    def test_a_local_model_opens_with_its_registered_dtype(self):
        seen: dict[str, object] = {}

        def fake_local(technical_name, *, device, torch_dtype):
            seen[technical_name] = (device, torch_dtype)
            return nullcontext(lambda texts: np.zeros((len(texts), 1)))

        with encoder_for("kalm-embedding", device="cuda", local=fake_local):
            pass
        with encoder_for("berel", local=fake_local):
            pass
        assert seen == {
            MODEL_REGISTRY["kalm-embedding"][0]: ("cuda", "bfloat16"),
            MODEL_REGISTRY["berel"][0]: (None, None),
        }


class TestSemanticSpecs:
    def test_one_spec_per_source_and_model_covering_the_tiers_they_share(self) -> None:
        """Each cell declares exactly the text partitions its model and source both have."""
        by_name = {spec.name: spec for spec in SPECS}
        assert set(by_name) == {
            f"semantic.generate.{source.slug}.{slug}"
            for source in SOURCES
            for slug in MODEL_REGISTRY
        }
        for source in SOURCES:
            for slug, (_, model_slug, _) in MODEL_REGISTRY.items():
                spec = by_name[f"semantic.generate.{source.slug}.{slug}"]
                expected = tuple(
                    f"{source.scope.directory}/domain=semantic/model={model_slug}/text={tier}"
                    for tier, _ in variations_for_model(slug)
                    if tier in source.tiers
                )
                assert tuple(p.directory for p in spec.partitions) == expected
                assert spec.args[-2:] == ("--model", slug)

    def test_a_scroll_cell_selects_its_scope_by_every_key_it_carries(self) -> None:
        spec = next(s for s in SPECS if s.name == "semantic.generate.dss-11Q5-none-verse.berel")
        assert spec.args == (
            "--corpus=dss",
            "--witness=11Q5",
            "--reconstruction=none",
            "--unit=verse",
            "--model",
            "berel",
        )

    def test_resource_classes_separate_hosted_large_and_small_models(self) -> None:
        """Hosted models need an API, Colab-only models a GPU, the rest run on the CPU."""
        assert resource_for("gemini") == "api"
        assert resource_for("kalm-embedding") == "gpu"
        assert resource_for("berel") is None
        assert {spec.resource for spec in SPECS} == {"api", "gpu", None}


class TestMain:
    def test_selects_the_source_by_its_scope_flags_and_one_model(self, tmp_path) -> None:
        calls: list[tuple[str, list[str]]] = []
        loaded: list[str] = []

        def source_factory(scope: Scope) -> Source:
            loaded.append(scope.directory)
            return Source(scope, ("consonantal",), lambda: SCROLL_UNITS)

        main(
            [
                "--output-root",
                str(tmp_path),
                "--corpus=dss",
                "--witness=11Q5",
                "--reconstruction=none",
                "--unit=verse",
                "--model",
                "berel",
            ],
            source_factory=source_factory,
            encoder=_recording_encoder(calls),
        )

        assert loaded == [SCROLL.directory]
        assert calls == [("berel", ["x", "y"])]
        assert partition("berel", "consonantal", SCROLL).file(tmp_path).exists()

    def test_without_a_model_flag_every_model_runs(self, tmp_path) -> None:
        calls: list[tuple[str, list[str]]] = []

        main(
            ["--output-root", str(tmp_path), "--corpus=bhsa", "--unit=half_verse"],
            source_factory=lambda scope: Source(scope, ("consonantal",), lambda: UNITS),
            encoder=_recording_encoder(calls),
        )

        assert [slug for slug, _ in calls] == list(MODEL_REGISTRY)
