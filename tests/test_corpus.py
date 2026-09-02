from __future__ import annotations

from pathlib import Path

import pytest

from semantic.corpus import Corpus


class TestCorpusLoad:
    def test_delegates_loading_to_the_shared_bhsa_loader(self, tmp_path):
        seen: dict[str, object] = {}

        def _loader(path, features):
            seen["path"], seen["features"] = path, features
            return "api"

        corpus = Corpus.load(tmp_path / "missing", loader=_loader)

        assert seen["path"] == tmp_path / "missing"
        assert "otype" in seen["features"]
        assert corpus.api == "api"


@pytest.mark.integration
def test_extracts_all_150_psalms_with_three_text_variants_and_real_node_ids():
    corpus = Corpus.load()
    psalms = corpus.psalms()

    assert len(psalms) == 150
    assert [p.number for p in psalms] == list(range(1, 151))

    psalm_1 = next(p for p in psalms if p.number == 1)
    assert len(psalm_1.half_verses) == 14
    assert len(psalm_1.half_verses_unvocalized) == 14
    assert len(psalm_1.half_verses_niqqud_only) == 14
    assert len(psalm_1.half_verse_nodes) == 14

    # Vocalized text has niqqud. Unvocalized text (BHSA's g_cons_utf8) has none.
    assert any("ָ" in hv for hv in psalm_1.half_verses)
    assert not any("ָ" in hv for hv in psalm_1.half_verses_unvocalized)

    # Niqqud-only text keeps niqqud but drops the cantillation marks the vocalized text carries.
    assert any("ָ" in hv for hv in psalm_1.half_verses_niqqud_only)
    assert any("֑" in hv for hv in psalm_1.half_verses)
    assert not any(any("֑" <= ch <= "֯" for ch in hv) for hv in psalm_1.half_verses_niqqud_only)

    all_nodes = [node for p in psalms for node in p.half_verse_nodes]
    assert len(all_nodes) == len(set(all_nodes))
    assert all(isinstance(node, int) and node > 0 for node in all_nodes)


@pytest.mark.integration
def test_load_accepts_an_explicit_path():
    from semantic.corpus import DEFAULT_BHSA_CLONE

    corpus = Corpus.load(Path(DEFAULT_BHSA_CLONE))
    assert len(corpus.psalms()) == 150
