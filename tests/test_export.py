"""Where a semantic dataset is written, and when an existing one is skipped."""

from __future__ import annotations

from core.partition import Scope
from semantic.export import partition, path_to_write


def test_the_partition_defaults_to_the_masoretic_half_verse(tmp_path):
    path = partition("bge_m3", "vocalized").file(tmp_path)
    assert path == (
        tmp_path
        / "corpus=bhsa/unit=half_verse/domain=semantic/model=bge_m3/text=vocalized"
        / "part-0.parquet"
    )


def test_a_scroll_scope_places_the_same_model_under_its_witness(tmp_path):
    scope = Scope("dss", "verse", witness="11Q5", reconstruction="none")
    path = partition("bge_m3", "consonantal", scope).file(tmp_path)
    assert path == (
        tmp_path
        / "corpus=dss/witness=11Q5/reconstruction=none/unit=verse/domain=semantic/model=bge_m3"
        / "text=consonantal/part-0.parquet"
    )


def test_path_to_write_skips_a_written_dataset(tmp_path, capsys):
    path = path_to_write(tmp_path, "bge_m3", "vocalized")
    assert path == partition("bge_m3", "vocalized").file(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"")
    assert path_to_write(tmp_path, "bge_m3", "vocalized") is None
    assert "model=bge_m3/text=vocalized" in capsys.readouterr().err
