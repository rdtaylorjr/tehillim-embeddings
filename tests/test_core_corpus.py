from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.corpus import (
    DEFAULT_BHSA_CLONE,
    PSALMS_BOOK_NAME,
    BaseCorpus,
    bhsa_clone_location,
    load_api,
    shared_api,
)


class _FakeFeature:
    def __init__(self, values: dict[int, object], nodes_by_type: dict[str, list[int]]) -> None:
        self._values = values
        self._nodes_by_type = nodes_by_type

    def v(self, node: int) -> object:
        return self._values.get(node)

    def s(self, otype: str) -> list[int]:
        return self._nodes_by_type.get(otype, [])


class _FakeF:
    def __init__(self, otype: _FakeFeature, book: _FakeFeature) -> None:
        self.otype = otype
        self.book = book


class _FakeApi:
    """Two psalms, numbered out of canonical order, each with two half-verse nodes."""

    def __init__(self, book_name: str = PSALMS_BOOK_NAME) -> None:
        self.F = _FakeF(
            otype=_FakeFeature({}, {"book": [1]}),
            book=_FakeFeature({1: book_name}, {}),
        )
        self.L = self
        self.T = self

    def d(self, node: int, otype: str) -> list[int]:
        if otype == "chapter":
            return [20, 10]
        if otype == "half_verse":
            return {10: [100, 101], 20: [200, 201]}[node]
        return []

    def sectionFromNode(self, node: int) -> tuple[str, int]:  # noqa: N802
        return ("Psalmi", {10: 1, 20: 2}[node])


class _CountingCorpus(BaseCorpus):
    """Minimal concrete corpus that records the half-verse nodes it was handed."""

    def _extract(self, number: int, half_verse_nodes: tuple[int, ...]):
        return (number, half_verse_nodes)


class TestBaseCorpus:
    def test_returns_psalms_in_canonical_number_order(self):
        corpus = _CountingCorpus(_FakeApi())

        assert [number for number, _ in corpus.psalms()] == [1, 2]

    def test_passes_each_chapters_half_verse_nodes_to_the_subclass(self):
        corpus = _CountingCorpus(_FakeApi())

        assert dict(corpus.psalms()) == {1: (100, 101), 2: (200, 201)}

    def test_exposes_the_underlying_api(self):
        api = _FakeApi()

        assert _CountingCorpus(api).api is api

    def test_raises_when_the_psalms_book_is_absent(self):
        corpus = _CountingCorpus(_FakeApi(book_name="Genesis"))

        with pytest.raises(RuntimeError, match="Psalmi"):
            corpus.psalms()


def _stub_api(features: str = "otype book verse") -> SimpleNamespace:
    """An object shaped enough like a Text-Fabric api, carrying the named features."""
    return SimpleNamespace(
        F=SimpleNamespace(**{name: object() for name in features.split()}),
        TF=SimpleNamespace(load=lambda *a, **k: None),
    )


class TestLoadApi:
    def test_falls_back_to_use_when_the_local_directory_is_missing(self, tmp_path):
        missing = tmp_path / "absent"
        calls: dict[str, object] = {}

        def _use(spec, checkout, silent):
            calls["spec"] = spec
            return SimpleNamespace(api=_stub_api())

        with pytest.warns(RuntimeWarning, match="falling back to use"):
            api = load_api(missing, "otype book", use_fn=_use)

        assert calls["spec"] == "etcbc/bhsa"
        assert api is not None

    def test_raises_only_when_both_the_local_clone_and_use_fail(self, tmp_path):
        def _fabric(locations, silent):
            class _Tf:
                def load(self, features, silent):
                    return None

            return _Tf()

        with pytest.raises(RuntimeError, match="failed to load BHSA"), pytest.warns(RuntimeWarning):
            load_api(tmp_path, "otype book", fabric=_fabric, use_fn=lambda *a, **k: None)

    def test_falls_back_to_use_when_the_local_clone_declines_to_load(self, tmp_path):
        def _fabric(locations, silent):
            class _Tf:
                def load(self, features, silent):
                    return None

            return _Tf()

        loaded = _stub_api()

        with pytest.warns(RuntimeWarning, match="falling back to use"):
            api = load_api(
                tmp_path,
                "otype book",
                fabric=_fabric,
                use_fn=lambda *a, **k: SimpleNamespace(api=loaded),
            )

        assert api is loaded

    def test_passes_the_requested_features_through_to_text_fabric(self, tmp_path):
        seen: dict[str, object] = {}

        def _fabric(locations, silent):
            seen["locations"] = locations

            class _Tf:
                def load(self, features, silent):
                    seen["features"] = features
                    return _stub_api()

            return _Tf()

        assert load_api(tmp_path, "otype book verse", fabric=_fabric) is not None
        assert seen["features"] == "otype book verse"
        assert seen["locations"] == [str(tmp_path)]

    def test_defaults_to_the_documented_bhsa_location(self):
        assert DEFAULT_BHSA_CLONE.name == "2021"
        assert DEFAULT_BHSA_CLONE.parent.name == "tf"


class TestBhsaCloneLocation:
    def test_prefers_the_environment_variable_when_it_is_set(self):
        assert bhsa_clone_location(env={"TEHILLIM_BHSA_PATH": "/custom/tf"}) == Path("/custom/tf")

    def test_falls_back_to_the_conventional_clone_path_when_unset(self):
        assert bhsa_clone_location(env={}) == DEFAULT_BHSA_CLONE

    def test_treats_an_empty_variable_as_unset(self):
        assert bhsa_clone_location(env={"TEHILLIM_BHSA_PATH": ""}) == DEFAULT_BHSA_CLONE

    def test_load_api_reads_the_environment_when_no_path_is_given(self, tmp_path):
        seen: dict[str, object] = {}

        def _fabric(locations, silent):
            seen["locations"] = locations

            class _Tf:
                def load(self, features, silent):
                    return _stub_api()

            return _Tf()

        result = load_api(
            None, "otype book", fabric=_fabric, env={"TEHILLIM_BHSA_PATH": str(tmp_path)}
        )

        assert result is not None
        assert seen["locations"] == [str(tmp_path)]


class TestLoadApiRejectsANonApiResult:
    def test_falls_back_when_the_local_load_returns_false(self, tmp_path):
        """Text-Fabric returns False (not None) when features are missing, and never raises."""

        def _fabric(locations, silent):
            class _Tf:
                def load(self, features, silent):
                    return False

            return _Tf()

        loaded = _stub_api()

        with pytest.warns(RuntimeWarning, match="falling back to use"):
            api = load_api(
                tmp_path,
                "otype book",
                fabric=_fabric,
                use_fn=lambda *a, **k: SimpleNamespace(api=loaded),
            )

        assert api is loaded

    def test_falls_back_when_the_local_load_returns_true_without_an_api(self, tmp_path):
        """load() can report success as True while producing no api object."""

        def _fabric(locations, silent):
            class _Tf:
                def load(self, features, silent):
                    return True

            return _Tf()

        loaded = _stub_api()

        with pytest.warns(RuntimeWarning, match="falling back to use"):
            api = load_api(
                tmp_path,
                "otype book",
                fabric=_fabric,
                use_fn=lambda *a, **k: SimpleNamespace(api=loaded),
            )

        assert api is loaded

    def test_raises_when_use_also_returns_something_that_is_not_an_api(self, tmp_path):
        def _fabric(locations, silent):
            class _Tf:
                def load(self, features, silent):
                    return False

            return _Tf()

        with (
            pytest.raises(RuntimeError, match="failed to load BHSA"),
            pytest.warns(RuntimeWarning),
        ):
            load_api(
                tmp_path,
                "otype book",
                fabric=_fabric,
                use_fn=lambda *a, **k: SimpleNamespace(api=False),
            )


class TestLoadApiVerifiesTheFeaturesItWasAskedFor:
    def test_raises_when_the_fallback_api_lacks_a_required_feature(self, tmp_path):
        """A fallback that silently omits a feature fails later with a confusing AttributeError."""
        partial = SimpleNamespace(
            F=SimpleNamespace(otype=object()), TF=SimpleNamespace(load=lambda *a, **k: None)
        )

        with (
            pytest.raises(RuntimeError, match="did not load required features"),
            pytest.warns(RuntimeWarning),
        ):
            load_api(
                tmp_path / "absent",
                "otype book",
                use_fn=lambda *a, **k: SimpleNamespace(api=partial),
            )

    def test_accepts_a_fallback_api_that_has_every_required_feature(self, tmp_path):
        complete = SimpleNamespace(
            F=SimpleNamespace(otype=object(), book=object()),
            TF=SimpleNamespace(load=lambda *a, **k: None),
        )

        with pytest.warns(RuntimeWarning):
            api = load_api(
                tmp_path / "absent",
                "otype book",
                use_fn=lambda *a, **k: SimpleNamespace(api=complete),
            )

        assert api is complete

    def test_accepts_a_required_edge_feature_that_loads_onto_e_rather_than_f(self, tmp_path):
        """Text-Fabric puts edge features such as `mother` on E, so demanding F would reject one."""
        complete = SimpleNamespace(
            F=SimpleNamespace(otype=object()),
            E=SimpleNamespace(mother=object()),
            TF=SimpleNamespace(load=lambda *a, **k: None),
        )

        with pytest.warns(RuntimeWarning):
            api = load_api(
                tmp_path / "absent",
                "otype mother",
                use_fn=lambda *a, **k: SimpleNamespace(api=complete),
            )

        assert api is complete

    def test_still_rejects_a_feature_present_on_neither_f_nor_e(self, tmp_path):
        partial = SimpleNamespace(
            F=SimpleNamespace(otype=object()),
            E=SimpleNamespace(mother=object()),
            TF=SimpleNamespace(load=lambda *a, **k: None),
        )

        with (
            pytest.raises(RuntimeError, match="did not load required features"),
            pytest.warns(RuntimeWarning),
        ):
            load_api(
                tmp_path / "absent",
                "otype mother book",
                use_fn=lambda *a, **k: SimpleNamespace(api=partial),
            )


class TestUseFallbackReportsWhyItFailed:
    """The local path names its reason, so the remote path must not lose one either."""

    @staticmethod
    def _declining_fabric(locations, silent):
        class _Tf:
            def load(self, features, silent):
                return None

        return _Tf()

    def test_names_the_error_use_raised(self, tmp_path):
        def _exploding_use(*_args, **_kwargs):
            raise ConnectionError("no route to the Text-Fabric host")

        with (
            pytest.raises(RuntimeError, match="no route to the Text-Fabric host"),
            pytest.warns(RuntimeWarning),
        ):
            load_api(tmp_path, "otype book", fabric=self._declining_fabric, use_fn=_exploding_use)

    def test_says_it_timed_out_when_use_never_returns(self, tmp_path):
        def _hanging_use(*_args, **_kwargs):
            time.sleep(5)

        with pytest.raises(RuntimeError, match="timed out"), pytest.warns(RuntimeWarning):
            load_api(
                tmp_path,
                "otype book",
                fabric=self._declining_fabric,
                use_fn=_hanging_use,
                timeout_seconds=0.05,
            )


class TestSharedApi:
    """One BHSA load per feature set, so a sweep over families pays for each corpus once."""

    def setup_method(self) -> None:
        shared_api.cache_clear()

    def teardown_method(self) -> None:
        shared_api.cache_clear()

    def test_a_second_request_for_the_same_features_reuses_the_first_load(self) -> None:
        loads: list[tuple[Path | None, str]] = []

        def _loader(path: Path | None, features: str) -> object:
            loads.append((path, features))
            return SimpleNamespace(name=len(loads))

        first = shared_api(None, "otype book", loader=_loader)
        second = shared_api(None, "otype book", loader=_loader)

        assert first is second
        assert loads == [(None, "otype book")]

    def test_a_different_feature_set_loads_separately(self) -> None:
        loads: list[str] = []

        def _loader(path: Path | None, features: str) -> object:
            loads.append(features)
            return SimpleNamespace(features=features)

        wide = shared_api(None, "otype book typ", loader=_loader)
        narrow = shared_api(None, "otype book", loader=_loader)

        assert wide is not narrow
        assert loads == ["otype book typ", "otype book"]

    def test_a_different_path_loads_separately(self) -> None:
        loads: list[Path | None] = []

        def _loader(path: Path | None, features: str) -> object:
            loads.append(path)
            return SimpleNamespace(path=path)

        shared_api(Path("/a"), "otype", loader=_loader)
        shared_api(Path("/b"), "otype", loader=_loader)

        assert loads == [Path("/a"), Path("/b")]

    def test_only_the_most_recent_feature_set_is_held(self) -> None:
        """Holding every loaded corpus swaps a long sweep to disk, costing more than a reload."""
        loads: list[str] = []

        def _loader(path: Path | None, features: str) -> object:
            loads.append(features)
            return SimpleNamespace(features=features)

        shared_api(None, "a", loader=_loader)
        shared_api(None, "b", loader=_loader)
        shared_api(None, "a", loader=_loader)

        assert loads == ["a", "b", "a"]

    def test_consecutive_requests_for_one_feature_set_still_load_once(self) -> None:
        """Callers group by corpus, so the bound holds while a group runs."""
        loads: list[str] = []

        def _loader(path: Path | None, features: str) -> object:
            loads.append(features)
            return SimpleNamespace(features=features)

        for _ in range(5):
            shared_api(None, "a", loader=_loader)

        assert loads == ["a"]

    def test_a_failed_load_is_not_cached(self) -> None:
        """Caching a failure would turn one bad load into a permanently broken process."""
        attempts: list[int] = []

        def _loader(path: Path | None, features: str) -> object:
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("cold clone")
            return SimpleNamespace(ok=True)

        with pytest.raises(RuntimeError, match="cold clone"):
            shared_api(None, "otype", loader=_loader)
        recovered = shared_api(None, "otype", loader=_loader)

        assert recovered.ok is True
        assert len(attempts) == 2
