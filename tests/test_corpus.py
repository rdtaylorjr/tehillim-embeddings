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
        assert corpus.unit == "half_verse"

    def test_opens_at_the_verse_when_asked(self):
        assert Corpus.load(unit="verse", loader=lambda path, features: "api").unit == "verse"

    def test_renders_the_consonantal_tier_as_letters_and_spaces(self):
        class _Api:
            class L:
                @staticmethod
                def d(node, otype):
                    return [1]

            class T:
                @staticmethod
                def text(words, fmt=None):
                    return "אַ֥שְֽׁרֵי־הָאִ֗ישׁ\u05c3 " if fmt is None else "אשׁרי־האישׁ\u05c3 "

        (unit,) = Corpus(_Api())._extract(1, (100,))
        assert unit.node == 100
        assert unit.texts["consonantal"] == "אשרי האיש"
        assert unit.texts["cantillation"] == "אַ֥שְֽׁרֵי־הָאִ֗ישׁ\u05c3"
        assert "֥" not in unit.texts["vocalized"]


@pytest.mark.integration
def test_extracts_every_half_verse_as_a_unit_with_three_tiers_and_real_node_ids():
    corpus = Corpus.load()
    psalms = corpus.psalms()
    units = corpus.units()

    assert len(psalms) == 150
    assert len(psalms[0]) == 14
    assert len(units) == sum(len(psalm) for psalm in psalms)

    first = psalms[0]
    # Cantillated text has niqqud and accents, vocalized niqqud only, consonantal neither.
    assert any("ָ" in unit.texts["cantillation"] for unit in first)
    assert any("֑" in unit.texts["cantillation"] for unit in first)
    assert any("ָ" in unit.texts["vocalized"] for unit in first)
    assert not any(any("֑" <= ch <= "֯" for ch in unit.texts["vocalized"]) for unit in first)
    assert not any("ָ" in unit.texts["consonantal"] for unit in first)

    nodes = [unit.node for unit in units]
    assert len(nodes) == len(set(nodes))
    assert all(isinstance(node, int) and node > 0 for node in nodes)


@pytest.mark.integration
def test_load_accepts_an_explicit_path():
    from semantic.corpus import DEFAULT_BHSA_CLONE

    corpus = Corpus.load(Path(DEFAULT_BHSA_CLONE))
    assert len(corpus.psalms()) == 150
