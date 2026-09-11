"""The registry a fused control draws from: one row per construction, bound to its builder."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pytest

from core.ngram import concatenated_1_2_3gram_dim, ngram_vectors, sparse_ngram_vectors
from core.support import build_signature_vocabulary, load_external_signature_counts
from families.shuffle import (
    CLAUSE_NGRAMS,
    FAMILIES,
    MORPH_SIGNATURE_DENSE,
    MORPH_SIGNATURE_K,
    MORPH_SIGNATURE_SPARSE,
    MORPH_SIGNATURE_SUPPORT,
    PHRASE_SIGNATURE_DENSE,
    TYP_WIDTH,
    Draws,
    build_clause,
    build_deploy,
    build_deploy_with_support,
    build_lexical,
    build_plain,
    build_with_support,
    by_psalm,
    draw,
    half_verse_sp,
    half_verse_typ,
    load_clause,
    load_draws,
    load_lexical,
    load_morphological_deploy,
    load_morphological_pos,
    load_morphological_signature,
    load_phrase_deploy,
    load_phrase_signature,
    load_phrase_unit,
    within_half_verse,
)
from lexical.corpus import LexicalPsalm
from lexical.positional import positional_icf_vectors
from morphological.corpus import MorphologicalPsalm
from morphological.vocabulary import SP_VOCABULARY, sp_columns
from syntactic.assignment import Assignment
from syntactic.clause_ordered import ORDERED_UNITS, resolve_family
from syntactic.corpus import ClausePsalm, PhrasePsalm
from syntactic.vocabulary import TYP_VOCABULARY, typ_columns

CONFIG = Path(__file__).resolve().parent.parent / "config"

#: Any seed exercises the same wiring, since the permutation reads it and the builder never does.
SEED = 1
EXPECTED_KEYS = (
    "lexical/homograph/icf_position4",
    "lexical/homograph/icf_position_mean_psalm",
    "morphological/morph_gn/1_2_3gram",
    "morphological/morph_gn/1_2_3gram_psalm",
    "morphological/morph_gn/1_2gram",
    "morphological/morph_gn/1_2gram_psalm",
    "morphological/morph_nu/1_2_3gram",
    "morphological/morph_nu/1_2_3gram_psalm",
    "morphological/morph_nu/1_2gram",
    "morphological/morph_nu/1_2gram_psalm",
    "morphological/morph_prs_gn/1_2_3gram",
    "morphological/morph_prs_gn/1_2_3gram_psalm",
    "morphological/morph_prs_gn/1_2gram",
    "morphological/morph_prs_gn/1_2gram_psalm",
    "morphological/morph_prs_nu/1_2_3gram",
    "morphological/morph_prs_nu/1_2_3gram_psalm",
    "morphological/morph_prs_nu/1_2gram",
    "morphological/morph_prs_nu/1_2gram_psalm",
    "morphological/morph_prs_ps/1_2_3gram",
    "morphological/morph_prs_ps/1_2_3gram_psalm",
    "morphological/morph_prs_ps/1_2gram",
    "morphological/morph_prs_ps/1_2gram_psalm",
    "morphological/morph_ps/1_2_3gram",
    "morphological/morph_ps/1_2_3gram_psalm",
    "morphological/morph_ps/1_2gram",
    "morphological/morph_ps/1_2gram_psalm",
    "morphological/morph_signature/1_2_3gram",
    "morphological/morph_signature/1_2_3gram_psalm",
    "morphological/morph_signature/1_2gram",
    "morphological/morph_signature/1_2gram_psalm",
    "morphological/morph_st/1_2_3gram",
    "morphological/morph_st/1_2_3gram_psalm",
    "morphological/morph_st/1_2gram",
    "morphological/morph_st/1_2gram_psalm",
    "morphological/morph_suffix/1_2_3gram",
    "morphological/morph_suffix/1_2_3gram_psalm",
    "morphological/morph_suffix/1_2gram",
    "morphological/morph_suffix/1_2gram_psalm",
    "morphological/morph_suffix/posmean",
    "morphological/morph_vs/1_2_3gram",
    "morphological/morph_vs/1_2_3gram_psalm",
    "morphological/morph_vs/1_2gram",
    "morphological/morph_vs/1_2gram_psalm",
    "morphological/morph_vt/1_2_3gram",
    "morphological/morph_vt/1_2_3gram_psalm",
    "morphological/morph_vt/1_2gram",
    "morphological/morph_vt/1_2gram_psalm",
    "morphological/sp/1_2_3gram",
    "morphological/sp/1_2_3gram_psalm",
    "morphological/sp/1_2gram",
    "morphological/sp/1_2gram_psalm",
    "syntactic/clause/kind/1_2_3gram",
    "syntactic/clause/kind/1_2_3gram_psalm",
    "syntactic/clause/kind/1_2gram",
    "syntactic/clause/kind/1_2gram_psalm",
    "syntactic/clause/rela/1_2_3gram",
    "syntactic/clause/rela/1_2_3gram_psalm",
    "syntactic/clause/rela/1_2gram",
    "syntactic/clause/rela/1_2gram_psalm",
    "syntactic/clause/signature/1_2_3gram",
    "syntactic/clause/signature/1_2_3gram_psalm",
    "syntactic/clause/signature/1_2gram",
    "syntactic/clause/signature/1_2gram_psalm",
    "syntactic/clause/tab/transition_psalm",
    "syntactic/clause/typ/1_2_3gram",
    "syntactic/clause/typ/1_2_3gram_psalm",
    "syntactic/clause/typ/1_2gram",
    "syntactic/clause/typ/1_2gram_psalm",
    "syntactic/phrase/det/1_2_3gram",
    "syntactic/phrase/det/1_2_3gram_psalm",
    "syntactic/phrase/det/1_2gram",
    "syntactic/phrase/det/1_2gram_psalm",
    "syntactic/phrase/function/1_2_3gram",
    "syntactic/phrase/function/1_2_3gram_psalm",
    "syntactic/phrase/function/1_2gram",
    "syntactic/phrase/function/1_2gram_psalm",
    "syntactic/phrase/rela/1_2_3gram",
    "syntactic/phrase/rela/1_2_3gram_psalm",
    "syntactic/phrase/rela/1_2gram",
    "syntactic/phrase/rela/1_2gram_psalm",
    "syntactic/phrase/signature/1_2_3gram",
    "syntactic/phrase/signature/1_2_3gram_psalm",
    "syntactic/phrase/signature/1_2gram",
    "syntactic/phrase/signature/1_2gram_psalm",
    "syntactic/phrase/full_signature/1_2_3gram",
    "syntactic/phrase/full_signature/1_2_3gram_psalm",
    "syntactic/phrase/full_signature/1_2gram",
    "syntactic/phrase/full_signature/1_2gram_psalm",
    "syntactic/phrase/signature/posmean",
    "syntactic/phrase/subphrase_rela/1_2_3gram",
    "syntactic/phrase/subphrase_rela/1_2_3gram_psalm",
    "syntactic/phrase/subphrase_rela/1_2gram",
    "syntactic/phrase/subphrase_rela/1_2gram_psalm",
    "syntactic/phrase/typ/1_2_3gram",
    "syntactic/phrase/typ/1_2_3gram_psalm",
    "syntactic/phrase/typ/1_2gram",
    "syntactic/phrase/typ/1_2gram_psalm",
)


def test_the_registry_holds_one_row_per_order_sensitive_construction() -> None:
    assert sorted(FAMILIES) == sorted(EXPECTED_KEYS)


def test_load_draws_refuses_a_key_the_registry_lacks() -> None:
    with pytest.raises(KeyError, match="not an order-sensitive construction"):
        load_draws("lexical/homograph/binary", CONFIG)


def test_load_draws_returns_what_the_registrys_row_loads(tmp_path: Path) -> None:
    loaded = Draws(key="k", psalms=(), permute=by_psalm, build=build_plain, sparse_width=None)

    draws = load_draws("k", tmp_path, families={"k": lambda _config_root: loaded})

    assert draws is loaded


def test_draw_builds_the_psalms_under_the_permutation_its_seed_fixes() -> None:
    seen: dict[str, object] = {}

    def permute(psalms, seed):
        seen["permuted"] = (psalms, seed)
        return {1: np.array([1, 0])}

    def build(psalms, order):
        seen["built"] = (psalms, order)
        return {10: np.zeros(2, dtype="<f4")}

    draws = Draws(key="k", psalms=("a", "b"), permute=permute, build=build, sparse_width=None)

    assert list(draw(draws, 7)) == [10]
    assert seen["permuted"] == (["a", "b"], 7)
    assert seen["built"][0] == ["a", "b"]  # type: ignore[index]


def test_by_psalm_permutes_each_psalms_half_verse_positions() -> None:
    psalms = [LexicalPsalm(number=3, half_verse_nodes=(10, 11, 12))]

    order = by_psalm(psalms, SEED)

    assert sorted(order[3].tolist()) == [0, 1, 2]


def test_within_half_verse_permutes_each_half_verses_elements_by_node() -> None:
    psalms = [
        MorphologicalPsalm(
            number=3,
            half_verse_nodes=(10, 11),
            half_verse_sp=(("subs", "verb"), ("conj", "verb", "subs")),
        )
    ]

    order = within_half_verse(half_verse_sp, psalms, SEED)

    assert sorted(order[10].tolist()) == [0, 1]
    assert sorted(order[11].tolist()) == [0, 1, 2]


def test_the_selectors_name_the_sequence_their_domain_permutes() -> None:
    morphological = MorphologicalPsalm(number=1, half_verse_sp=(("subs",),))
    phrase = PhrasePsalm(number=1, half_verse_typ=(("NP",),))

    assert half_verse_sp(morphological) == (("subs",),)
    assert half_verse_typ(phrase) == (("NP",),)


def test_build_lexical_projects_the_psalms_onto_the_column_view_its_builder_reads() -> None:
    seen: dict[str, object] = {}

    def builder(columns, vocabulary, weights, *, order_by_psalm):
        seen.update(columns=columns, vocabulary=vocabulary, weights=weights, order=order_by_psalm)
        return {10: np.zeros(1, dtype="<f4")}

    psalms = [
        LexicalPsalm(
            number=1,
            half_verse_nodes=(10,),
            half_verse_lexemes=(("A/",),),
            half_verse_forms=(("A/",),),
        )
    ]
    order = {1: np.array([0])}

    build_lexical(builder, ("A/",), {"A/": 1.0}, psalms, order)

    assert [columns.number for columns in seen["columns"]] == [1]  # type: ignore[attr-defined]
    assert seen["vocabulary"] == ("A/",)
    assert seen["weights"] == {"A/": 1.0}
    assert seen["order"] is order


def test_build_plain_passes_only_the_psalms_and_the_permutation() -> None:
    seen: dict[str, object] = {}

    def builder(psalms, order_by_node):
        seen.update(psalms=psalms, order=order_by_node)
        return {10: np.zeros(1, dtype="<f4")}

    build_plain(builder, ["p"], {10: np.array([0])})

    assert seen["psalms"] == ["p"]


def test_build_with_support_passes_the_frozen_vocabulary_and_its_counts() -> None:
    seen: dict[str, object] = {}

    def builder(psalms, vocabulary, counts, k, order):
        seen.update(psalms=psalms, vocabulary=vocabulary, counts=counts, k=k, order=order)
        return {10: np.zeros(1, dtype="<f4")}

    build_with_support(builder, ("a",), {"a": 5}, 3, ["p"], {10: np.array([0])})

    assert (seen["vocabulary"], seen["counts"], seen["k"]) == (("a",), {"a": 5}, 3)


def test_build_deploy_passes_the_permutation_by_keyword() -> None:
    seen: dict[str, object] = {}

    def builder(psalms, *, order_by_psalm):
        seen.update(psalms=psalms, order=order_by_psalm)
        return {10: np.zeros(1, dtype="<f4")}

    order = {1: np.array([0])}
    build_deploy(builder, ["p"], order)

    assert seen["order"] is order


def test_build_deploy_with_support_passes_the_vocabulary_and_the_keyword_permutation() -> None:
    seen: dict[str, object] = {}

    def builder(psalms, vocabulary, counts, k, *, order_by_psalm):
        seen.update(vocabulary=vocabulary, counts=counts, k=k, order=order_by_psalm)
        return {10: np.zeros(1, dtype="<f4")}

    order = {1: np.array([0])}
    build_deploy_with_support(builder, ("a",), {"a": 5}, 3, ["p"], order)

    assert (seen["vocabulary"], seen["k"], seen["order"]) == (("a",), 3, order)


def test_build_clause_routes_through_the_producer_its_loader_bound() -> None:
    seen: dict[str, object] = {}

    def produce(family, psalms, construction, order):
        seen.update(family=family, psalms=psalms, construction=construction, order=order)
        return {10: np.zeros(1, dtype="<f4")}

    order = {10: np.array([0])}
    build_clause(produce, "family", "1_2gram", ["p"], order)

    assert (seen["family"], seen["construction"], seen["order"]) == ("family", "1_2gram", order)


@pytest.mark.parametrize("unit", ORDERED_UNITS)
def test_the_registry_holds_every_ordered_clause_construction(unit: str) -> None:
    """A clause family declares its own constructions, so the registry must not drift from them."""
    family = resolve_family(unit, CONFIG)
    prefix = f"syntactic/clause/{unit}/"

    registered = {key.removeprefix(prefix) for key in FAMILIES if key.startswith(prefix)}

    assert registered == set(family.dense) | set(family.sparse)
    assert {name: name in family.sparse for name in registered} == {
        name: CLAUSE_NGRAMS.get(name, False) for name in registered
    }


class _FakeWordFeature:
    """One Text-Fabric feature over the fake corpus's two words."""

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
                half_verse_nodes=(100, 101, 102),
                half_verse_lexemes=(("A", "B"), ("A",), ("B",)),
                half_verse_forms=(("A0", "B0"), ("A0",), ("B0",)),
            )
        ]


class _FakeMorphologicalCorpus:
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


class _FakePhraseCorpus:
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


class _FakeClauseCorpus:
    def psalms(self) -> list[ClausePsalm]:
        return [
            ClausePsalm(
                number=1,
                half_verse_nodes=(100, 101),
                clause_atom_nodes=(0, 1, 2, 3),
                clause_atom_typ=("WayX", "XQtl", "WayX", "ZQtl"),
                clause_atom_assignment=Assignment(
                    node_index=np.array([0, 1, 2, 3]),
                    unit_index=np.array([0, 0, 1, 1]),
                    weight=np.array([1.0, 1.0, 1.0, 1.0]),
                ),
            )
        ]


class TestLoaders:
    """A loader binds a corpus to a builder, so each one is checked against a fake corpus."""

    def test_load_lexical_binds_the_psalm_order_permutation_and_a_dense_draw(self) -> None:
        draws = load_lexical(
            "lexical/homograph/icf_position4",
            partial(positional_icf_vectors, k=4),
            CONFIG,
            corpus_factory=_FakeLexicalCorpus,
        )

        assert draws.permute is by_psalm
        assert draws.sparse_width is None
        assert sorted(draw(draws, SEED)) == [100, 101, 102]

    def test_load_morphological_pos_permutes_inside_each_half_verse(self) -> None:
        draws = load_morphological_pos(
            "morphological/sp/1_2gram",
            partial(ngram_vectors, columns_of=sp_columns, vocabulary=SP_VOCABULARY, orders=(1, 2)),
            CONFIG,
            corpus_factory=_FakeMorphologicalCorpus,
        )

        assert sorted(draws.permute(list(draws.psalms), SEED)) == [100, 101]
        assert draws.sparse_width is None
        assert sorted(draw(draws, SEED)) == [100, 101]

    def test_load_morphological_signature_widens_only_when_it_stores_sparsely(self) -> None:
        """A sparse width is the concatenated n-gram dimension, not the vocabulary size."""
        counts = load_external_signature_counts(CONFIG / MORPH_SIGNATURE_SUPPORT)
        vocabulary = build_signature_vocabulary(counts, MORPH_SIGNATURE_K)
        dense = load_morphological_signature(
            "morphological/morph_signature/1_2gram",
            MORPH_SIGNATURE_DENSE["1_2gram"],
            CONFIG,
            sparse=False,
            corpus_factory=_FakeMorphologicalCorpus,
        )
        sparse = load_morphological_signature(
            "morphological/morph_signature/1_2_3gram",
            MORPH_SIGNATURE_SPARSE["1_2_3gram"],
            CONFIG,
            sparse=True,
            corpus_factory=_FakeMorphologicalCorpus,
        )

        assert dense.sparse_width is None
        assert sparse.sparse_width == concatenated_1_2_3gram_dim(len(vocabulary))

    def test_load_morphological_deploy_permutes_half_verse_order(self) -> None:
        draws = load_morphological_deploy(
            "morphological/morph_suffix/posmean",
            CONFIG,
            corpus_factory=_FakeMorphologicalCorpus,
        )

        assert draws.permute is by_psalm
        assert draws.sparse_width is None
        assert draw(draws, SEED)

    def test_load_phrase_unit_widens_a_trigram_and_leaves_a_bigram_dense(self) -> None:
        dense = load_phrase_unit(
            "syntactic/phrase/typ/1_2gram",
            partial(
                ngram_vectors, columns_of=typ_columns, vocabulary=TYP_VOCABULARY, orders=(1, 2)
            ),
            CONFIG,
            sparse_dim=None,
            corpus_factory=_FakePhraseCorpus,
        )
        sparse = load_phrase_unit(
            "syntactic/phrase/typ/1_2_3gram",
            partial(sparse_ngram_vectors, columns_of=typ_columns, vocabulary=TYP_VOCABULARY),
            CONFIG,
            sparse_dim=TYP_WIDTH,
            corpus_factory=_FakePhraseCorpus,
        )

        assert dense.sparse_width is None
        assert sparse.sparse_width == concatenated_1_2_3gram_dim(TYP_WIDTH)
        assert sorted(draw(dense, SEED)) == [100, 101]

    def test_load_phrase_signature_reads_the_frozen_support_table(self) -> None:
        draws = load_phrase_signature(
            "syntactic/phrase/signature/1_2gram",
            PHRASE_SIGNATURE_DENSE["1_2gram"],
            CONFIG,
            sparse=False,
            corpus_factory=_FakePhraseCorpus,
        )

        assert draws.sparse_width is None
        assert sorted(draw(draws, SEED)) == [100, 101]

    def test_load_phrase_deploy_permutes_half_verse_order(self) -> None:
        draws = load_phrase_deploy(
            "syntactic/phrase/signature/posmean", CONFIG, corpus_factory=_FakePhraseCorpus
        )

        assert draws.permute is by_psalm
        assert draws.sparse_width is None
        assert draw(draws, SEED)

    def test_load_clause_permutes_through_its_family_record(self) -> None:
        draws = load_clause(
            "syntactic/clause/typ/1_2gram",
            "typ",
            "1_2gram",
            CONFIG,
            sparse=False,
            corpus_factory=_FakeClauseCorpus,
        )

        assert draws.sparse_width is None
        assert sorted(draw(draws, SEED)) == [100, 101]
