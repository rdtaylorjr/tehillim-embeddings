from __future__ import annotations

from pathlib import Path

from embeddings.heavy import (
    HEAVY_MODELS,
    ensure_corpus_data,
    gpu_memory_summary,
    models_for_choice,
)


class TestModelsForChoice:
    def test_none_returns_every_heavy_model(self):
        assert models_for_choice(None) == HEAVY_MODELS

    def test_a_choice_returns_only_that_model(self):
        result = models_for_choice("kalm")
        assert len(result) == 1
        assert result[0][0] == "kalm-embedding"


class TestEnsureCorpusData:
    def test_returns_the_defaults_when_both_already_exist(self, tmp_path):
        bhsa = tmp_path / "bhsa"
        valence = tmp_path / "valence"
        bhsa.mkdir()
        valence.mkdir()

        def _must_not_be_called(url: str, destination: Path) -> None:
            raise AssertionError("clone must not be called when defaults already exist")

        result = ensure_corpus_data(
            bhsa_default=bhsa,
            valence_default=valence,
            data_dir=tmp_path / "data",
            clone=_must_not_be_called,
        )

        assert result == (bhsa, valence)

    def test_clones_both_when_neither_default_exists(self, tmp_path):
        clones: list[tuple[str, Path]] = []

        def _fake_clone(url: str, destination: Path) -> None:
            clones.append((url, destination))
            destination.mkdir(parents=True)

        data_dir = tmp_path / "data"
        result = ensure_corpus_data(
            bhsa_default=tmp_path / "missing-bhsa",
            valence_default=tmp_path / "missing-valence",
            data_dir=data_dir,
            clone=_fake_clone,
        )

        assert result == (data_dir / "bhsa" / "tf" / "2021", data_dir / "valence" / "tf" / "2021")
        cloned_urls = {url for url, _ in clones}
        assert cloned_urls == {
            "https://github.com/ETCBC/bhsa.git",
            "https://github.com/ETCBC/valence.git",
        }

    def test_skips_cloning_a_repo_already_present_in_data_dir(self, tmp_path):
        data_dir = tmp_path / "data"
        (data_dir / "bhsa").mkdir(parents=True)
        clones: list[str] = []

        def _fake_clone(url: str, destination: Path) -> None:
            clones.append(url)
            destination.mkdir(parents=True)

        ensure_corpus_data(
            bhsa_default=tmp_path / "missing-bhsa",
            valence_default=tmp_path / "missing-valence",
            data_dir=data_dir,
            clone=_fake_clone,
        )

        assert clones == ["https://github.com/ETCBC/valence.git"]


class _FakeCuda:
    def __init__(self, *, available: bool, allocated: float, reserved: float, total: float):
        self._available = available
        self._allocated = allocated
        self._reserved = reserved
        self._total = total

    def is_available(self) -> bool:
        return self._available

    def memory_allocated(self) -> float:
        return self._allocated

    def memory_reserved(self) -> float:
        return self._reserved

    def get_device_properties(self, index: int) -> object:
        return type("Props", (), {"total_memory": self._total})()


class _FakeTorch:
    def __init__(self, cuda: _FakeCuda) -> None:
        self.cuda = cuda


class TestGpuMemorySummary:
    def test_returns_none_when_cuda_unavailable(self):
        fake_torch = _FakeTorch(_FakeCuda(available=False, allocated=0, reserved=0, total=0))
        assert gpu_memory_summary(fake_torch) is None

    def test_formats_memory_in_gigabytes(self):
        fake_torch = _FakeTorch(
            _FakeCuda(available=True, allocated=1e9, reserved=2e9, total=4e10)
        )
        summary = gpu_memory_summary(fake_torch)
        assert summary == "allocated=1.00GB reserved=2.00GB total=40.00GB"
