"""A scroll read as verse units: labels, runs, kept letters, and the node each row is keyed to."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.partition import Scope
from semantic.scroll import (
    RECONSTRUCTIONS,
    ScrollCorpus,
    dss_location,
    kept_letters,
    label_of,
    load_api,
    runs,
    scribes_location,
    text_of,
)


class _Feature:
    def __init__(self, values):
        self._values = values

    def v(self, node):
        return self._values.get(node)


class _Otype(_Feature):
    def s(self, otype):
        return [node for node, value in self._values.items() if value == otype]


class _Locality:
    def __init__(self, words_by_scroll):
        self._words = words_by_scroll

    def d(self, node, otype):
        assert otype == "word"
        return self._words[node]


class _F:
    def __init__(self, **features):
        for name, values in features.items():
            setattr(self, name, _Otype(values) if name == "otype" else _Feature(values))


class _Api:
    def __init__(self, F, L):  # noqa: N803 -- Text-Fabric's names
        self.F, self.L = F, L


def _api():
    """Two scrolls: 11Q5 with a psalm verse split around a lacuna, a composition, and a plus."""
    words = {
        1: {
            "biblical": 1,
            "book": "Ps",
            "chapter": "1",
            "verse": "1",
            "gword": 1,
            "text": "אשׁרי",
            "grade": "cccc",
        },
        2: {
            "biblical": 1,
            "book": "Ps",
            "chapter": "1",
            "verse": "1",
            "gword": 2,
            "text": "האישׁ",
            "grade": "rrcc",
        },
        3: {
            "biblical": 1,
            "book": "Ps",
            "chapter": "1",
            "verse": "1",
            "gword": 2,
            "text": "האישׁ",
            "grade": "rrcc",
        },
        4: {"type": "punct"},
        5: {
            "biblical": 1,
            "book": "Ps",
            "chapter": "1",
            "verse": "2",
            "gword": 5,
            "text": "כי",
            "grade": "rr",
        },
        6: {
            "biblical": 1,
            "book": "2Sam",
            "chapter": "23",
            "verse": "7",
            "gword": None,
            "text": None,
            "grade": None,
        },
        7: {
            "composition": "Ps151A",
            "composition_verse": 0,
            "gword": 7,
            "text": "הללויה",
            "grade": "cccccc",
        },
        8: {
            "composition": "Ps151A",
            "composition_verse": 0,
            "gword": 7,
            "text": "הללויה",
            "grade": "cccccc",
        },
        9: {
            "composition": "Ps151A",
            "composition_verse": 1,
            "gword": 9,
            "text": "קטן",
            "grade": "cqx",
        },
        10: {
            "biblical": 1,
            "book": "Ps",
            "chapter": "1",
            "verse": "1",
            "gword": 10,
            "text": "שׂם",
            "grade": "cc",
        },
    }
    features = {
        "otype": {100: "scroll", 200: "scroll"},
        "scroll": {100: "11Q5", 200: "4Q83"},
    }
    for name in (
        "biblical",
        "book",
        "chapter",
        "verse",
        "type",
        "composition",
        "composition_verse",
        "gword",
        "grade",
    ):
        features[name] = {node: values.get(name) for node, values in words.items()}
    features["gw_cons_utf8"] = {node: values.get("text") for node, values in words.items()}
    return _Api(_F(**features), _Locality({100: list(words), 200: [11]}))


class TestLabelOf:
    def test_a_psalm_word_is_labelled_by_the_dss_chapter_and_verse(self):
        assert label_of(_api().F, 1) == ("Ps", "1", "1")

    def test_a_composition_word_is_labelled_by_the_scribes_composition_and_verse(self):
        assert label_of(_api().F, 7) == ("Ps151A", None, "0")

    def test_a_word_of_another_book_or_of_no_verse_has_no_label(self):
        assert label_of(_api().F, 6) is None
        assert label_of(_api().F, 4) is None


class TestKeptLetters:
    def test_keeps_the_letters_whose_grade_is_kept(self):
        assert kept_letters("האישׁ", "rrcc", RECONSTRUCTIONS["none"]) == "ישׁ"

    def test_a_shin_dot_rides_on_its_letter(self):
        assert kept_letters("שׂם", "rc", RECONSTRUCTIONS["none"]) == "ם"
        assert kept_letters("שׂם", "cr", RECONSTRUCTIONS["none"]) == "שׂ"

    def test_a_trace_is_not_a_letter(self):
        assert kept_letters("קטן", "cqx", RECONSTRUCTIONS["none"]) == "קט"

    def test_refuses_a_word_whose_letters_and_grades_disagree(self):
        with pytest.raises(ValueError, match="2 letters but 3 grades"):
            kept_letters("אב", "ccc", RECONSTRUCTIONS["none"])


class TestRuns:
    def test_cuts_maximal_runs_under_one_label_and_drops_unlabelled_words(self):
        found = list(runs(_api().F, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        assert [(run.label, run.nodes) for run in found] == [
            (("Ps", "1", "1"), (1, 2, 3)),
            (("Ps", "1", "2"), (5,)),
            (("Ps151A", None, "0"), (7, 8)),
            (("Ps151A", None, "1"), (9,)),
            (("Ps", "1", "1"), (10,)),
        ]

    def test_a_label_that_recurs_later_is_a_second_run(self):
        """Ps 118:29 stands twice in 11Q5, so a label is not a key and the node is."""
        found = list(runs(_api().F, [1, 7, 10]))
        assert [run.nodes for run in found] == [(1,), (7,), (10,)]


class TestTextOf:
    def test_emits_each_graphic_word_once_with_its_kept_letters(self):
        api = _api()
        (first, *_) = runs(api.F, [1, 2, 3])
        assert text_of(api.F, first, RECONSTRUCTIONS["none"]) == "אשרי יש"

    def test_a_word_with_no_kept_letters_drops_out(self):
        api = _api()
        (run,) = runs(api.F, [5])
        assert text_of(api.F, run, RECONSTRUCTIONS["none"]) == ""


class TestScrollCorpus:
    def test_words_are_the_scrolls_word_nodes_in_order_without_punctuation(self):
        assert ScrollCorpus(_api(), "11Q5").words() == [1, 2, 3, 5, 6, 7, 8, 9, 10]

    def test_an_unknown_scroll_is_refused(self):
        with pytest.raises(LookupError, match="4Q99 is not a scroll"):
            ScrollCorpus(_api(), "4Q99").words()

    def test_units_are_the_runs_with_text_keyed_by_their_first_node(self):
        units = ScrollCorpus(_api(), "11Q5").units()
        assert [(unit.node, unit.texts) for unit in units] == [
            (1, {"consonantal": "אשרי יש"}),
            (7, {"consonantal": "הללויה"}),
            (9, {"consonantal": "קט"}),
            (10, {"consonantal": "שם"}),
        ]

    def test_the_scope_names_the_witness_the_reconstruction_and_the_verse(self):
        assert ScrollCorpus(_api(), "11Q5").scope == Scope(
            "dss", "verse", witness="11Q5", reconstruction="none"
        )

    def test_load_opens_dss_with_the_scribes_module_through_the_injected_loader(self):
        api = _api()
        corpus = ScrollCorpus.load("11Q5", loader=lambda: api)
        assert corpus.api is api
        assert corpus.reconstruction == "none"


class TestLoadApi:
    def test_reads_both_locations_from_the_environment(self, tmp_path: Path):
        seen: dict[str, object] = {}

        def load(locations, features):
            seen["locations"], seen["features"] = locations, features
            return "api"

        env = {
            "TEHILLIM_DSS_PATH": str(tmp_path / "dss"),
            "TEHILLIM_SCRIBES_TF_PATH": str(tmp_path / "s"),
        }
        assert load_api(load=load, env=env) == "api"
        assert seen["locations"] == [tmp_path / "dss", tmp_path / "s"]
        assert "gw_cons_utf8" in seen["features"]
        assert "composition_verse" in seen["features"]

    def test_defaults_to_the_conventional_checkouts(self):
        assert dss_location({}).name == "2.0"
        assert scribes_location({}).parts[-3:] == ("tehillim-scribes", "tf", "2.0")

    def test_a_missing_dataset_is_an_error_naming_the_locations(self):
        with pytest.raises(RuntimeError, match="could not load dss"):
            load_api(load=lambda locations, features: None, env={})
