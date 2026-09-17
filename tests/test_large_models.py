from __future__ import annotations

from pathlib import Path

from semantic.large_models import (
    CHECKOUTS,
    LARGE_MODELS,
    Checkout,
    ensure_checkout,
    ensure_corpus_data,
    gpu_memory_summary,
    models_for_choice,
)


class TestModelsForChoice:
    def test_none_returns_every_large_model(self):
        assert models_for_choice(None) == LARGE_MODELS

    def test_a_choice_returns_only_that_model(self):
        result = models_for_choice("kalm")
        assert len(result) == 1
        assert result[0][0] == "kalm-embedding"

    def test_harrier_and_f2llm_are_registered_large_models(self):
        slugs = {model[0] for model in LARGE_MODELS}
        assert "harrier-oss-v1" in slugs
        assert "f2llm-v2" in slugs

    def test_harrier_and_f2llm_choices_resolve(self):
        assert models_for_choice("harrier")[0][0] == "harrier-oss-v1"
        assert models_for_choice("f2llm")[0][0] == "f2llm-v2"


class TestEnsureCheckout:
    def test_returns_the_default_when_it_already_exists(self, tmp_path):
        bhsa = tmp_path / "bhsa"
        bhsa.mkdir()

        def _must_not_be_called(url: str, destination: Path) -> None:
            raise AssertionError("clone must not be called when the default already exists")

        checkout = Checkout("X", bhsa, "https://github.com/ETCBC/bhsa.git", "tf/2021")
        assert ensure_checkout(checkout, data_dir=tmp_path / "d", clone=_must_not_be_called) == bhsa

    def test_clones_when_the_default_does_not_exist(self, tmp_path):
        clones: list[tuple[str, Path]] = []

        def _fake_clone(url: str, destination: Path) -> None:
            clones.append((url, destination))
            destination.mkdir(parents=True)

        data_dir = tmp_path / "data"
        checkout = Checkout("X", tmp_path / "missing", "https://github.com/ETCBC/dss.git", "tf/2.0")
        result = ensure_checkout(checkout, data_dir=data_dir, clone=_fake_clone)

        assert result == data_dir / "dss" / "tf" / "2.0"
        assert clones == [("https://github.com/ETCBC/dss.git", data_dir / "dss")]

    def test_skips_cloning_a_repo_already_present_in_data_dir(self, tmp_path):
        data_dir = tmp_path / "data"
        (data_dir / "bhsa").mkdir(parents=True)
        clones: list[str] = []

        def _fake_clone(url: str, destination: Path) -> None:
            clones.append(url)

        checkout = Checkout(
            "X", tmp_path / "missing", "https://github.com/ETCBC/bhsa.git", "tf/2021"
        )
        ensure_checkout(checkout, data_dir=data_dir, clone=_fake_clone)

        assert clones == []


class TestEnsureCorpusData:
    def test_names_every_dataset_by_the_variable_its_loader_reads(self, tmp_path):
        env = ensure_corpus_data(
            data_dir=tmp_path, clone=lambda url, dest: dest.mkdir(parents=True)
        )

        assert set(env) == {"TEHILLIM_BHSA_PATH", "TEHILLIM_DSS_PATH", "TEHILLIM_SCRIBES_TF_PATH"}
        assert all(
            Path(path).parts[-2:] == ("tf", version)
            for path, version in zip(env.values(), ("2021", "2.0", "2.0"), strict=True)
        ) or all(Path(path).exists() for path in env.values())

    def test_the_checkouts_cover_the_three_datasets_the_sources_read(self):
        assert [c.repository.rsplit("/", 1)[1] for c in CHECKOUTS] == [
            "bhsa.git",
            "dss.git",
            "tehillim-scribes.git",
        ]


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
        fake_torch = _FakeTorch(_FakeCuda(available=True, allocated=1e9, reserved=2e9, total=4e10))
        summary = gpu_memory_summary(fake_torch)
        assert summary == "allocated=1.00GB reserved=2.00GB total=40.00GB"
