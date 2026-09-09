"""Covers each shuffle-control CLI: argparse wiring, shuffle-count default, dataset naming."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.export import dataset_path
from core.shuffle import DEFAULT_N_SHUFFLES, shuffle_construction_name
from lexical.corpus import LexicalPsalm
from lexical.scripts import generate_shuffle_control as lexical_psalm_script
from lexical.scripts import generate_shuffle_control_half_verse as lexical_half_verse_script
from lexical.scripts.shuffle_driver import SeedContext, write_seed
from morphological.corpus import MorphologicalPsalm
from morphological.scripts import generate_deploy_shuffle_control as morph_deploy_script
from morphological.scripts import generate_shuffle_control as morph_script
from syntactic.corpus import PhrasePsalm
from syntactic.scripts import generate_deploy_shuffle_control as syntactic_deploy_script
from syntactic.scripts import generate_shuffle_control as syntactic_script

CONFIG_ROOT = Path(__file__).resolve().parent.parent / "config"


class _FakeWordFeature:
    def __init__(self, values: dict[int, object]) -> None:
        self._values = values

    def v(self, node: int) -> object:
        return self._values.get(node)

    def s(self, otype: str) -> list[int]:
        return sorted(self._values) if otype == "word" else []


class _FakeLexicalApi:
    """Just enough Text-Fabric surface for the lexical ICF weighting to run."""

    def __init__(self) -> None:
        self.F = type(
            "_F",
            (),
            {
                "otype": _FakeWordFeature({1: "word", 2: "word"}),
                "lex": _FakeWordFeature({1: "A", 2: "B"}),
                "lex0": _FakeWordFeature({1: "A0", 2: "B0"}),
                "freq_lex": _FakeWordFeature({1: 10, 2: 20}),
            },
        )()


class _FakeLexicalCorpus:
    def __init__(self) -> None:
        self.api = _FakeLexicalApi()

    def psalms(self) -> list[LexicalPsalm]:
        return [
            LexicalPsalm(
                number=1,
                half_verse_lexemes=(("A", "B"), ("A",), ("B",)),
                half_verse_forms=(("A0", "B0"), ("A0",), ("B0",)),
                half_verse_nodes=(100, 101, 102),
            )
        ]


class _FakeMorphologyCorpus:
    def psalms(self) -> list[MorphologicalPsalm]:
        all_na = (("NA", "NA", "NA"), ("NA", "NA", "NA"))
        return [
            MorphologicalPsalm(
                number=1,
                half_verse_nodes=(100, 101),
                half_verse_sp=(("subs", "verb", "prep"), ("verb", "subs", "conj")),
                half_verse_gn=all_na,
                half_verse_nu=all_na,
                half_verse_ps=all_na,
                half_verse_st=all_na,
                half_verse_vs=all_na,
                half_verse_vt=all_na,
                half_verse_prs_gn=all_na,
                half_verse_prs_nu=all_na,
                half_verse_prs_ps=all_na,
            )
        ]


class _FakeSyntaxCorpus:
    def psalms(self) -> list[PhrasePsalm]:
        return [
            PhrasePsalm(
                number=1,
                half_verse_nodes=(100, 101),
                half_verse_typ=(("NP", "VP", "PP"), ("VP", "NP", "CP")),
                half_verse_function=(("Subj", "Pred", "Cmpl"), ("Pred", "Subj", "Conj")),
                half_verse_det=(("det", "NA", "und"), ("NA", "det", "NA")),
                half_verse_rela=(("NA", "NA", "NA"), ("NA", "NA", "NA")),
                half_verse_n_words=((1, 1, 1), (1, 1, 1)),
                half_verse_phrase_id=((10, 11, 12), (13, 14, 15)),
                half_verse_phrase_atom_count=((1, 1, 1), (1, 1, 1)),
                half_verse_subphrase_rela=(("NA",), ("NA",)),
            )
        ]


def _lexical_dataset(root: Path, construction: str) -> Path:
    return dataset_path(root, "homograph", construction, unit_key="unit")


def _seed_context(output_root: Path) -> SeedContext:
    """The shared driver's context for the psalm-broadcast lexical control."""
    return SeedContext(
        psalms=tuple(_FakeLexicalCorpus().psalms()),
        output_root=output_root,
        vocabulary=("A0", "B0"),
        icf_weights={"A0": 1.5, "B0": 2.0},
        construction=lexical_psalm_script._CONSTRUCTION,
        builder=lexical_psalm_script.build_vectors,
    )


_SCRIPTS = [
    lexical_psalm_script,
    lexical_half_verse_script,
    morph_script,
    morph_deploy_script,
    syntactic_script,
    syntactic_deploy_script,
]


#: The arguments each parser marks required, so `--n-shuffles` can be read off in isolation.
def _parser(module):
    """Each CLI's parser, whether it defines one itself or takes the shared lexical driver's."""
    try:
        return module.build_parser()
    except TypeError:
        return module.build_parser(module.__doc__)


_REQUIRED_ARGS = {
    lexical_psalm_script: [],
    lexical_half_verse_script: [],
    morph_script: ["--family", "pos", "--representation", "1_2gram"],
    morph_deploy_script: [],
    syntactic_script: ["--unit", "typ", "--representation", "1_2gram"],
    syntactic_deploy_script: [],
}


class TestShuffleCountDefault:
    @pytest.mark.parametrize("module", _SCRIPTS)
    def test_every_cli_defaults_to_the_shared_shuffle_count(self, module, tmp_path):
        argv = [*_REQUIRED_ARGS[module], "--output-root", str(tmp_path)]
        if "--config-root" in _parser(module).format_usage():
            argv += ["--config-root", str(CONFIG_ROOT)]

        args = _parser(module).parse_args(argv)

        assert args.n_shuffles == DEFAULT_N_SHUFFLES

    @pytest.mark.parametrize("module", _SCRIPTS)
    def test_every_cli_lets_the_shuffle_count_be_overridden(self, module, tmp_path):
        argv = [*_REQUIRED_ARGS[module], "--output-root", str(tmp_path), "--n-shuffles", "7"]
        if "--config-root" in _parser(module).format_usage():
            argv += ["--config-root", str(CONFIG_ROOT)]

        args = _parser(module).parse_args(argv)

        assert args.n_shuffles == 7


class TestLexicalPsalmMain:
    def test_writes_the_requested_number_of_four_digit_named_datasets(self, tmp_path):
        lexical_psalm_script.main(
            ["--output-root", str(tmp_path), "--n-shuffles", "2", "--max-workers", "1"],
            corpus_factory=_FakeLexicalCorpus,
        )

        for seed in (1, 2):
            name = shuffle_construction_name("icf_position_mean_psalm", seed)
            assert _lexical_dataset(tmp_path, name).exists()

    def test_names_carry_the_full_width_rather_than_two_digits(self, tmp_path):
        lexical_psalm_script.main(
            ["--output-root", str(tmp_path), "--n-shuffles", "1", "--max-workers", "1"],
            corpus_factory=_FakeLexicalCorpus,
        )

        assert not _lexical_dataset(tmp_path, "icf_position_mean_psalm_shuffle01").exists()
        assert _lexical_dataset(tmp_path, "icf_position_mean_psalm_shuffle0001").exists()


class TestLexicalHalfVerseMain:
    def test_writes_the_requested_number_of_datasets(self, tmp_path):
        lexical_half_verse_script.main(
            ["--output-root", str(tmp_path), "--n-shuffles", "2", "--max-workers", "1"],
            corpus_factory=_FakeLexicalCorpus,
        )

        for seed in (1, 2):
            name = shuffle_construction_name("icf_position4", seed)
            assert _lexical_dataset(tmp_path, name).exists()


class TestHighSeedNaming:
    """Seeds past the two- and three-digit boundaries, which a small run never reaches."""

    @pytest.mark.parametrize(
        ("seed", "expected"),
        [
            (9, "icf_position_mean_psalm_shuffle0009"),
            (99, "icf_position_mean_psalm_shuffle0099"),
            (100, "icf_position_mean_psalm_shuffle0100"),
            (999, "icf_position_mean_psalm_shuffle0999"),
            (DEFAULT_N_SHUFFLES, "icf_position_mean_psalm_shuffle1000"),
        ],
    )
    def test_a_single_high_seed_writes_under_its_padded_name(self, seed, expected, tmp_path):
        context = _seed_context(tmp_path)

        name = write_seed(context, seed)

        assert name == expected
        assert _lexical_dataset(tmp_path, expected).exists()

    def test_names_written_across_the_hundred_boundary_still_sort_in_seed_order(self, tmp_path):
        context = _seed_context(tmp_path)
        seeds = [1, 9, 10, 99, 100, 101, 999, 1000]

        names = [write_seed(context, seed) for seed in seeds]

        assert names == sorted(names)

    def test_a_seed_past_the_name_width_is_refused_before_anything_is_written(self, tmp_path):
        context = _seed_context(tmp_path)

        with pytest.raises(ValueError, match="exceeds"):
            write_seed(context, 10000)

        assert list(tmp_path.rglob("*.parquet")) == []


class TestMorphologyMains:
    def test_pos_family_writes_named_datasets(self, tmp_path):
        morph_script.main(
            [
                "--family",
                "pos",
                "--representation",
                "1_2gram",
                "--output-root",
                str(tmp_path),
                "--config-root",
                str(CONFIG_ROOT),
                "--n-shuffles",
                "2",
                "--max-workers",
                "1",
            ],
            corpus_factory=_FakeMorphologyCorpus,
        )

        name = shuffle_construction_name("1_2gram", 1)
        assert dataset_path(
            tmp_path, "sp", name, domain="morphological", unit_key="feature"
        ).exists()

    def test_signature_family_reads_its_support_table_and_writes_datasets(self, tmp_path):
        morph_script.main(
            [
                "--family",
                "signature",
                "--representation",
                "1_2gram",
                "--output-root",
                str(tmp_path),
                "--config-root",
                str(CONFIG_ROOT),
                "--n-shuffles",
                "1",
                "--max-workers",
                "1",
            ],
            corpus_factory=_FakeMorphologyCorpus,
        )

        name = shuffle_construction_name("1_2gram", 1)
        assert dataset_path(
            tmp_path, "morph_signature", name, domain="morphological", unit_key="feature"
        ).exists()

    def test_deploy_writes_named_datasets(self, tmp_path):
        morph_deploy_script.main(
            ["--output-root", str(tmp_path), "--n-shuffles", "2", "--max-workers", "1"],
            corpus_factory=_FakeMorphologyCorpus,
        )

        name = shuffle_construction_name("posmean", 2)
        assert dataset_path(
            tmp_path, "morph_suffix", name, domain="morphological", unit_key="feature"
        ).exists()


class TestSyntaxMains:
    def test_typ_unit_writes_named_datasets(self, tmp_path):
        syntactic_script.main(
            [
                "--unit",
                "typ",
                "--representation",
                "1_2gram",
                "--output-root",
                str(tmp_path),
                "--config-root",
                str(CONFIG_ROOT),
                "--n-shuffles",
                "2",
                "--max-workers",
                "1",
            ],
            corpus_factory=_FakeSyntaxCorpus,
        )

        name = shuffle_construction_name("1_2gram", 1)
        assert dataset_path(
            tmp_path, "typ", name, domain="syntactic", unit_key="feature", level="phrase"
        ).exists()

    def test_signature_unit_reads_its_support_table(self, tmp_path):
        syntactic_script.main(
            [
                "--unit",
                "signature",
                "--representation",
                "1_2gram",
                "--output-root",
                str(tmp_path),
                "--config-root",
                str(CONFIG_ROOT),
                "--n-shuffles",
                "1",
                "--max-workers",
                "1",
            ],
            corpus_factory=_FakeSyntaxCorpus,
        )

        name = shuffle_construction_name("1_2gram", 1)
        assert dataset_path(
            tmp_path, "signature", name, domain="syntactic", unit_key="feature", level="phrase"
        ).exists()

    def test_deploy_writes_named_datasets(self, tmp_path):
        syntactic_deploy_script.main(
            [
                "--output-root",
                str(tmp_path),
                "--config-root",
                str(CONFIG_ROOT),
                "--n-shuffles",
                "2",
                "--max-workers",
                "1",
            ],
            corpus_factory=_FakeSyntaxCorpus,
        )

        name = shuffle_construction_name("posmean", 1)
        assert dataset_path(
            tmp_path, "signature", name, domain="syntactic", unit_key="feature", level="phrase"
        ).exists()
