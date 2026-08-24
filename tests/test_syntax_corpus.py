from __future__ import annotations

from pathlib import Path

import pytest

from syntax.corpus import Corpus


class TestCorpusLoad:
    def test_raises_a_clear_error_when_bhsa_path_does_not_exist(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="BHSA"):
            Corpus.load(tmp_path / "missing")


@pytest.mark.integration
def test_extracts_all_150_psalms_with_phrase_typ_aligned_to_colons():
    corpus = Corpus.load()
    psalms = corpus.psalms()

    assert len(psalms) == 150
    assert [p.number for p in psalms] == list(range(1, 151))

    psalm_1 = next(p for p in psalms if p.number == 1)
    assert len(psalm_1.colon_nodes) == 14
    assert len(psalm_1.colon_typ) == 14
    assert len(psalm_1.colon_function) == 14
    feature_sequences = (
        psalm_1.colon_typ,
        psalm_1.colon_function,
        psalm_1.colon_det,
        psalm_1.colon_rela,
        psalm_1.colon_n_words,
        psalm_1.colon_phrase_id,
        psalm_1.colon_phrase_atom_count,
    )
    for phrase_typ_colon, *rest in zip(*feature_sequences, strict=True):
        for colon in rest:
            assert len(colon) == len(phrase_typ_colon)

    # subphrase_rela is indexed by subphrase, not phrase atom, so it only aligns by colon count.
    assert len(psalm_1.colon_subphrase_rela) == 14

    all_nodes = [node for p in psalms for node in p.colon_nodes]
    assert len(all_nodes) == len(set(all_nodes))
    assert all(isinstance(node, int) and node > 0 for node in all_nodes)


@pytest.mark.integration
def test_a_real_colons_phrase_typ_sequence_matches_a_manual_tf_query():
    corpus = Corpus.load()
    psalm_1 = next(p for p in corpus.psalms() if p.number == 1)
    api = corpus.api
    colon_node = psalm_1.colon_nodes[0]

    manual = tuple(api.F.typ.v(pa) for pa in api.L.d(colon_node, otype="phrase_atom"))

    assert psalm_1.colon_typ[0] == manual


@pytest.mark.integration
def test_phrase_typ_values_match_the_verified_thirteen_value_bhsa_inventory():
    from syntax.vocabulary import TYP_VOCABULARY

    corpus = Corpus.load()
    observed = {value for p in corpus.psalms() for hv in p.colon_typ for value in hv}

    assert observed == set(TYP_VOCABULARY)


@pytest.mark.integration
def test_a_real_colons_phrase_function_sequence_matches_a_manual_tf_query():
    corpus = Corpus.load()
    psalm_1 = next(p for p in corpus.psalms() if p.number == 1)
    api = corpus.api
    colon_node = psalm_1.colon_nodes[0]

    manual = tuple(
        api.F.function.v(api.L.u(pa, otype="phrase")[0])
        for pa in api.L.d(colon_node, otype="phrase_atom")
    )

    assert psalm_1.colon_function[0] == manual


@pytest.mark.integration
def test_phrase_function_values_observed_in_psalms_are_a_subset_of_the_frozen_inventory():
    from syntax.vocabulary import FUNCTION_VOCABULARY

    corpus = Corpus.load()
    observed = {value for p in corpus.psalms() for hv in p.colon_function for value in hv}

    assert observed <= set(FUNCTION_VOCABULARY)


@pytest.mark.integration
def test_phrase_det_n_words_and_phrase_bookkeeping_match_a_manual_tf_query():
    corpus = Corpus.load()
    psalm_1 = next(p for p in corpus.psalms() if p.number == 1)
    api = corpus.api
    F, L = api.F, api.L  # noqa: N806
    colon_node = psalm_1.colon_nodes[0]
    atoms = L.d(colon_node, otype="phrase_atom")

    manual_det = tuple(F.det.v(pa) for pa in atoms)
    manual_n_words = tuple(len(L.d(pa, otype="word")) for pa in atoms)
    manual_phrase_id = tuple(L.u(pa, otype="phrase")[0] for pa in atoms)
    manual_atom_count = tuple(len(L.d(pid, otype="phrase_atom")) for pid in manual_phrase_id)

    assert psalm_1.colon_det[0] == manual_det
    assert psalm_1.colon_n_words[0] == manual_n_words
    assert psalm_1.colon_phrase_id[0] == manual_phrase_id
    assert psalm_1.colon_phrase_atom_count[0] == manual_atom_count


@pytest.mark.integration
def test_phrase_rela_matches_a_manual_tf_query_and_whole_bhsa_inventory_includes_para():
    corpus = Corpus.load()
    psalm_1 = next(p for p in corpus.psalms() if p.number == 1)
    api = corpus.api
    colon_node = psalm_1.colon_nodes[0]

    manual_rela = tuple(api.F.rela.v(pa) for pa in api.L.d(colon_node, otype="phrase_atom"))
    assert psalm_1.colon_rela[0] == manual_rela

    # Confirms the contamination risk this session's Phase 5E plan explicitly quarantines:
    # rela=Para exists in the raw whole-BHSA feature and must never reach a benchmarked vector.
    whole_bhsa_rela = {api.F.rela.v(pa) for pa in api.F.otype.s("phrase_atom")}
    assert "Para" in whole_bhsa_rela


@pytest.mark.integration
def test_subphrase_rela_matches_a_manual_tf_query_and_whole_bhsa_inventory_includes_par():
    corpus = Corpus.load()
    psalm_1 = next(p for p in corpus.psalms() if p.number == 1)
    api = corpus.api
    colon_node = psalm_1.colon_nodes[0]

    manual_rela = tuple(api.F.rela.v(sp) for sp in api.L.d(colon_node, otype="subphrase"))
    assert psalm_1.colon_subphrase_rela[0] == manual_rela

    # Same contamination risk as phrase_atom's rela=Para, but lowercase and far more common
    # at the subphrase level (417 occurrences in Psalms alone vs. phrase_atom's 15).
    whole_bhsa_subphrase_rela = {api.F.rela.v(sp) for sp in api.F.otype.s("subphrase")}
    assert "par" in whole_bhsa_subphrase_rela


@pytest.mark.integration
def test_load_accepts_an_explicit_path():
    from syntax.corpus import DEFAULT_BHSA_TF_PATH

    corpus = Corpus.load(Path(DEFAULT_BHSA_TF_PATH))
    assert len(corpus.psalms()) == 150


@pytest.mark.integration
def test_api_property_exposes_the_whole_corpus_not_just_psalms():
    corpus = Corpus.load()
    psalms_atom_count = sum(len(typ) for p in corpus.psalms() for typ in p.colon_typ)

    whole_bible_atom_count = len(corpus.api.F.otype.s("phrase_atom"))

    assert whole_bible_atom_count > psalms_atom_count * 5
