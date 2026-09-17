from __future__ import annotations

import pytest

from lexical.surface_corpus import SurfaceCorpus


class TestSurfaceCorpusLoad:
    def test_delegates_loading_to_the_shared_bhsa_loader(self, tmp_path):
        seen: dict[str, object] = {}

        def _loader(path, features):
            seen["path"], seen["features"] = path, features
            return "api"

        corpus = SurfaceCorpus.load(tmp_path / "missing", loader=_loader)

        assert seen["path"] == tmp_path / "missing"
        assert "otype" in seen["features"]
        assert corpus.api == "api"

    def test_the_consonantal_forms_are_letters_only(self):
        class _Feature:
            def __init__(self, values):
                self._values = values

            def v(self, node):
                return self._values[node]

        class _Api:
            class F:
                g_cons_utf8 = _Feature({1: "אשׁרי", 2: "שׂם"})
                g_word_utf8 = _Feature({1: "אַ֥שְֽׁרֵי", 2: "שָׂם"})

            class L:
                @staticmethod
                def d(node, otype):
                    return [1, 2]

        psalm = SurfaceCorpus(_Api())._extract(1, (100,))
        assert psalm.half_verse_consonantal == (("אשרי", "שם"),)
        assert psalm.half_verse_cantillation == (("אַ֥שְֽׁרֵי", "שָׂם"),)


@pytest.mark.integration
def test_extracts_all_150_psalms_with_three_aligned_text_tiers():
    corpus = SurfaceCorpus.load()
    psalms = corpus.psalms()

    assert len(psalms) == 150
    assert [p.number for p in psalms] == list(range(1, 151))

    psalm_1 = next(p for p in psalms if p.number == 1)
    assert len(psalm_1.half_verse_nodes) == 14
    assert len(psalm_1.half_verse_consonantal) == 14
    assert len(psalm_1.half_verse_vocalized) == 14
    assert len(psalm_1.half_verse_cantillation) == 14

    for consonantal, vocalized, cantillation in zip(
        psalm_1.half_verse_consonantal,
        psalm_1.half_verse_vocalized,
        psalm_1.half_verse_cantillation,
        strict=True,
    ):
        assert len(consonantal) == len(vocalized) == len(cantillation)

    # Cantillation adds marks over niqqud, so it differs while the consonantal form stays shortest.
    first_word_cons = psalm_1.half_verse_consonantal[0][0]
    first_word_cant = psalm_1.half_verse_cantillation[0][0]
    assert len(first_word_cons) <= len(first_word_cant)
