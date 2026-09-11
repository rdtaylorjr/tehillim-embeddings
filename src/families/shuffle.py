"""Every order-sensitive construction, declared once so a draw can be built without a file."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.export import dataset_path
from core.ngram import (
    concatenated_1_2_3gram_dim,
    ngram_psalm_vectors,
    ngram_vectors,
    sparse_ngram_psalm_vectors,
    sparse_ngram_vectors,
)
from core.ngram_dataset import NgramDataset, order_sensitive_constructions
from core.shuffle import (
    shuffle_construction_name,
    shuffled_order_by_psalm,
    shuffled_within_half_verse_order,
)
from core.support import build_signature_vocabulary, load_external_signature_counts
from lexical.corpus import Corpus as LexicalCorpus
from lexical.frequency import icf_weights as compute_icf_weights
from lexical.frequency import lex0_token_frequencies, total_token_count
from lexical.positional import positional_icf_vectors
from lexical.psalm_zoning import psalm_position_mean_vectors
from lexical.vocabulary import VocabularyKey, build_vocabulary, columns_for_key
from morphological.atomic import (
    SPARSE_TRIGRAM_FEATURES,
    FeatureKey,
    feature_1_2_3gram_psalm_vectors,
    feature_1_2_3gram_vectors,
    feature_1_2gram_psalm_vectors,
    feature_1_2gram_vectors,
    feature_sparse_trigram_psalm_vectors,
    feature_sparse_trigram_vectors,
    vocabulary_for_feature,
)
from morphological.corpus import Corpus as MorphologicalCorpus
from morphological.corpus import MorphologicalPsalm
from morphological.deploy import suffix_deploy_vectors
from morphological.generate_suffix import DATASET as SUFFIX_DATASET
from morphological.signature_support import MIN_EXTERNAL_SUPPORT_K as MORPH_SIGNATURE_K
from morphological.signature_vectorize import (
    ORDERED_DENSE_BUILDERS as MORPH_SIGNATURE_DENSE,
)
from morphological.signature_vectorize import (
    SPARSE_BUILDERS as MORPH_SIGNATURE_SPARSE,
)
from morphological.suffix import psalm_suffix_signatures
from morphological.vocabulary import SP_VOCABULARY, sp_columns
from syntactic.clause_kind import (
    clause_kind_1_2_3gram_psalm_vectors,
    clause_kind_1_2_3gram_vectors,
    clause_kind_1_2gram_psalm_vectors,
    clause_kind_1_2gram_vectors,
    clause_kind_columns,
)
from syntactic.clause_ordered import OrderedFamily, dense_vectors, resolve_family, sparse_vectors
from syntactic.clause_rela_vectorize import (
    clause_rela_1_2_3gram_psalm_vectors,
    clause_rela_1_2_3gram_vectors,
    clause_rela_1_2gram_psalm_vectors,
    clause_rela_1_2gram_vectors,
    clause_rela_columns,
)
from syntactic.clause_support import MIN_EXTERNAL_SUPPORT_K_CLAUSE_RELA as CLAUSE_RELA_K
from syntactic.corpus import ClausePsalm, PhrasePsalm, clause_corpus, phrase_corpus
from syntactic.corpus import Corpus as SyntacticCorpus
from syntactic.deploy import signature_deploy_vectors
from syntactic.det_vectorize import (
    phrase_det_1_2_3gram_psalm_vectors,
    phrase_det_1_2_3gram_vectors,
    phrase_det_1_2gram_psalm_vectors,
    phrase_det_1_2gram_vectors,
)
from syntactic.full_signature_vectorize import DENSE_BUILDERS as FULL_SIGNATURE_DENSE
from syntactic.full_signature_vectorize import SPARSE_BUILDERS as FULL_SIGNATURE_SPARSE
from syntactic.rela_vectorize import (
    phrase_rela_1_2_3gram_psalm_vectors,
    phrase_rela_1_2_3gram_vectors,
    phrase_rela_1_2gram_psalm_vectors,
    phrase_rela_1_2gram_vectors,
)
from syntactic.signature_support import MIN_EXTERNAL_SUPPORT_K as PHRASE_SIGNATURE_K
from syntactic.signature_support import MIN_EXTERNAL_SUPPORT_K_FULL as FULL_SIGNATURE_K
from syntactic.signature_vectorize import ORDERED_DENSE_BUILDERS as PHRASE_SIGNATURE_DENSE
from syntactic.signature_vectorize import SPARSE_BUILDERS as PHRASE_SIGNATURE_SPARSE
from syntactic.subphrase_vectorize import (
    subphrase_rela_1_2_3gram_psalm_vectors,
    subphrase_rela_1_2_3gram_vectors,
    subphrase_rela_1_2gram_psalm_vectors,
    subphrase_rela_1_2gram_vectors,
)
from syntactic.vocabulary import (
    FUNCTION_VOCABULARY,
    TYP_VOCABULARY,
    function_columns,
    typ_columns,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import numpy as np

#: One node's dense vector, or its nonzero (indices, values) pair when the family stores sparsely.
type Draw = dict[int, Any]

#: Every order-sensitive construction, keyed by the Hive partition its dataset is written under.
type Registry = dict[str, Callable[[Path], Draws]]

MORPH_SIGNATURE_SUPPORT = "morph_signature_external_support.csv"
PHRASE_SIGNATURE_SUPPORT = "phrase_signature_external_support.csv"
FULL_SIGNATURE_SUPPORT = "phrase_full_signature_external_support.csv"

LEXICAL_VOCABULARY_KEY: VocabularyKey = "lex0"

#: Every domain names its partition `feature=`, except the lexical one that predates it.
UNIT_KEYS = {"lexical": "unit"}


@dataclass(frozen=True, slots=True)
class Draws:
    """A loaded corpus and a bound builder: everything a seeded draw needs, pickled once."""

    key: str
    psalms: tuple[Any, ...]
    permute: Callable[[list[Any], int], dict[int, np.ndarray]]
    build: Callable[[list[Any], dict[int, np.ndarray]], Draw]
    sparse_width: int | None


def draw(draws: Draws, seed: int) -> Draw:
    """One seed's vectors under its permutation, with nothing written to disk."""
    psalms = list(draws.psalms)
    return draws.build(psalms, draws.permute(psalms, seed))


def _dataset_path(key: str, output_root: Path, construction: str) -> Path:
    """The Hive-partitioned file a family writes under, for whichever construction is named."""
    domain, *middle, _ = key.split("/")
    return dataset_path(
        output_root,
        middle[-1],
        construction,
        unit_key=UNIT_KEYS.get(domain, "feature"),
        level=middle[0] if len(middle) == 2 else None,
        domain=domain,
    )


def dataset_target(key: str, output_root: Path, seed: int) -> Path:
    """The Hive-partitioned file one family's seed is written at, which its key already names."""
    return _dataset_path(key, output_root, shuffle_construction_name(key.rsplit("/", 1)[1], seed))


def dataset_source(key: str, data_root: Path) -> Path:
    """The unshuffled dataset a family's draws are the null for, under the same partition."""
    return _dataset_path(key, data_root, key.rsplit("/", 1)[1])


# --- permutations, module level so a worker can unpickle them ------------------------------


def by_psalm(psalms: list[Any], seed: int) -> dict[int, np.ndarray]:
    """Permutes the order of a psalm's half-verses."""
    return shuffled_order_by_psalm(psalms, seed)


def within_half_verse(
    select: Callable[[Any], Any], psalms: list[Any], seed: int
) -> dict[int, np.ndarray]:
    """Permutes the order of the elements inside each half-verse."""
    return shuffled_within_half_verse_order(psalms, seed, half_verses=select)


def half_verse_sp(psalm: MorphologicalPsalm) -> tuple[tuple[str, ...], ...]:
    """The word-level sequence the morphological controls permute."""
    return psalm.half_verse_sp


def half_verse_typ(psalm: PhrasePsalm) -> tuple[tuple[str, ...], ...]:
    """The phrase-atom sequence the syntactic phrase controls permute."""
    return psalm.half_verse_typ


def clause_rela_columns_for(
    external_counts: dict[str, int], k: int
) -> Callable[[Any], tuple[tuple[str, ...], ...]]:
    """The clause-relation column reader, with its support table already bound."""
    return partial(_clause_rela_columns, external_counts, k)


def _clause_rela_columns(
    external_counts: dict[str, int], k: int, psalm: ClausePsalm
) -> tuple[tuple[str, ...], ...]:
    """One psalm's RARE-collapsed clause-relation sequences, per colon."""
    return clause_rela_columns(psalm, external_counts, k)


def half_verse_subphrase_rela(psalm: PhrasePsalm) -> tuple[tuple[str, ...], ...]:
    """The subphrase sequence, which is shorter than the phrase-atom sequence beside it."""
    return psalm.half_verse_subphrase_rela


# --- builders, each adapting one call shape to build(psalms, order) ------------------------


def build_lexical(
    builder: Callable[..., Draw],
    vocabulary: tuple[str, ...],
    icf_weights: dict[str, float],
    psalms: list[Any],
    order: dict[int, np.ndarray],
) -> Draw:
    """Lexical builders read column views rather than psalms, so the view is derived here."""
    columns = columns_for_key(psalms, LEXICAL_VOCABULARY_KEY)
    return builder(columns, vocabulary, icf_weights, order_by_psalm=order)


def build_plain(
    builder: Callable[..., Draw], psalms: list[Any], order: dict[int, np.ndarray]
) -> Draw:
    """Builders over a closed vocabulary, which need only the psalms and the permutation."""
    return builder(psalms, order_by_node=order)


def build_with_support(
    builder: Callable[..., Draw],
    vocabulary: tuple[str, ...],
    counts: dict[str, int],
    k: int,
    psalms: list[Any],
    order: dict[int, np.ndarray],
) -> Draw:
    """Builders whose vocabulary comes from a frozen external-support table."""
    return builder(psalms, vocabulary, counts, k, order)


def build_feature(
    builder: Callable[..., Draw],
    feature: FeatureKey,
    psalms: list[Any],
    order: dict[int, np.ndarray],
) -> Draw:
    """Atomic-feature builders take the feature they read before the permutation."""
    return builder(psalms, feature, order)


def build_deploy(
    builder: Callable[..., Draw], psalms: list[Any], order: dict[int, np.ndarray]
) -> Draw:
    """Deployment builders take the permutation by keyword and no vocabulary."""
    return builder(psalms, order_by_psalm=order)


def build_deploy_with_support(
    builder: Callable[..., Draw],
    vocabulary: tuple[str, ...],
    counts: dict[str, int],
    k: int,
    psalms: list[Any],
    order: dict[int, np.ndarray],
) -> Draw:
    """Deployment builders whose vocabulary comes from a frozen external-support table."""
    return builder(psalms, vocabulary, counts, k, order_by_psalm=order)


def build_clause(
    produce: Callable[..., Draw],
    family: OrderedFamily,
    construction: str,
    psalms: list[Any],
    order: dict[int, np.ndarray],
) -> Draw:
    """Clause builders resolve through a family record, which already binds their vocabulary."""
    return produce(family, psalms, construction, order)


# --- loaders, one per corpus and vocabulary shape ------------------------------------------


def load_lexical(
    key: str,
    builder: Callable[..., Draw],
    _config_root: Path,
    *,
    corpus_factory: Callable[[], LexicalCorpus] = LexicalCorpus.load,
) -> Draws:
    """Binds the lexical corpus, its lex0 vocabulary and its ICF weights."""
    corpus = corpus_factory()
    psalms = corpus.psalms()
    icf = compute_icf_weights(lex0_token_frequencies(corpus.api), total_token_count(corpus.api))
    vocabulary = build_vocabulary(psalms, key=LEXICAL_VOCABULARY_KEY)
    return Draws(
        key=key,
        psalms=tuple(psalms),
        permute=by_psalm,
        build=partial(build_lexical, builder, vocabulary, icf),
        sparse_width=None,
    )


def load_morphological_pos(
    key: str,
    builder: Callable[..., Draw],
    _config_root: Path,
    *,
    corpus_factory: Callable[[], MorphologicalCorpus] = MorphologicalCorpus.load,
) -> Draws:
    """Binds the morphological corpus for a POS construction, whose vocabulary is closed."""
    psalms = corpus_factory().psalms()
    return Draws(
        key=key,
        psalms=tuple(psalms),
        permute=partial(within_half_verse, half_verse_sp),
        build=partial(build_plain, builder),
        sparse_width=None,
    )


def load_morphological_signature(
    key: str,
    builder: Callable[..., Draw],
    config_root: Path,
    *,
    sparse: bool,
    corpus_factory: Callable[[], MorphologicalCorpus] = MorphologicalCorpus.load,
) -> Draws:
    """Binds the morphological corpus and the frozen grammatical-signature vocabulary."""
    psalms = corpus_factory().psalms()
    counts = load_external_signature_counts(config_root / MORPH_SIGNATURE_SUPPORT)
    vocabulary = build_signature_vocabulary(counts, MORPH_SIGNATURE_K)
    return Draws(
        key=key,
        psalms=tuple(psalms),
        permute=partial(within_half_verse, half_verse_sp),
        build=partial(build_with_support, builder, vocabulary, counts, MORPH_SIGNATURE_K),
        sparse_width=concatenated_1_2_3gram_dim(len(vocabulary)) if sparse else None,
    )


def load_morphological_feature(
    key: str,
    builder: Callable[..., Draw],
    feature: FeatureKey,
    _config_root: Path,
    *,
    sparse_width: int | None,
    corpus_factory: Callable[[], MorphologicalCorpus] = MorphologicalCorpus.load,
) -> Draws:
    """Binds the morphological corpus for one atomic feature's n-gram constructions."""
    psalms = corpus_factory().psalms()
    return Draws(
        key=key,
        psalms=tuple(psalms),
        permute=partial(within_half_verse, half_verse_sp),
        build=partial(build_feature, builder, feature),
        sparse_width=sparse_width,
    )


def load_morphological_deploy(
    key: str,
    _config_root: Path,
    *,
    corpus_factory: Callable[[], MorphologicalCorpus] = MorphologicalCorpus.load,
) -> Draws:
    """Binds the morphological corpus for the pronominal-suffix deployment vectors."""
    psalms = corpus_factory().psalms()
    return Draws(
        key=key,
        psalms=tuple(psalms),
        permute=by_psalm,
        build=partial(build_deploy, suffix_deploy_vectors),
        sparse_width=None,
    )


def load_phrase_unit(
    key: str,
    builder: Callable[..., Draw],
    _config_root: Path,
    *,
    sparse_dim: int | None,
    permute_columns: Callable[[Any], tuple[tuple[str, ...], ...]] = half_verse_typ,
    corpus_factory: Callable[[], SyntacticCorpus[Any]] = phrase_corpus,
) -> Draws:
    """Binds the phrase corpus for a closed-vocabulary construction, permuting its own column."""
    psalms = corpus_factory().psalms()
    return Draws(
        key=key,
        psalms=tuple(psalms),
        permute=partial(within_half_verse, permute_columns),
        build=partial(build_plain, builder),
        sparse_width=concatenated_1_2_3gram_dim(sparse_dim) if sparse_dim else None,
    )


def load_phrase_signature(
    key: str,
    builder: Callable[..., Draw],
    config_root: Path,
    *,
    sparse: bool,
    support_filename: str = PHRASE_SIGNATURE_SUPPORT,
    k: int = PHRASE_SIGNATURE_K,
    corpus_factory: Callable[[], SyntacticCorpus[PhrasePsalm]] = phrase_corpus,
) -> Draws:
    """Binds the phrase corpus and a frozen signature vocabulary, whichever table declares it."""
    psalms = corpus_factory().psalms()
    counts = load_external_signature_counts(config_root / support_filename)
    vocabulary = build_signature_vocabulary(counts, k)
    return Draws(
        key=key,
        psalms=tuple(psalms),
        permute=partial(within_half_verse, half_verse_typ),
        build=partial(build_with_support, builder, vocabulary, counts, k),
        sparse_width=concatenated_1_2_3gram_dim(len(vocabulary)) if sparse else None,
    )


def load_phrase_deploy(
    key: str,
    config_root: Path,
    *,
    corpus_factory: Callable[[], SyntacticCorpus[PhrasePsalm]] = phrase_corpus,
) -> Draws:
    """Binds the phrase corpus and signature vocabulary for the deployment vectors."""
    psalms = corpus_factory().psalms()
    counts = load_external_signature_counts(config_root / PHRASE_SIGNATURE_SUPPORT)
    vocabulary = build_signature_vocabulary(counts, PHRASE_SIGNATURE_K)
    return Draws(
        key=key,
        psalms=tuple(psalms),
        permute=by_psalm,
        build=partial(
            build_deploy_with_support,
            signature_deploy_vectors,
            vocabulary,
            counts,
            PHRASE_SIGNATURE_K,
        ),
        sparse_width=None,
    )


def load_clause_supported(
    key: str,
    builder: Callable[..., Draw],
    config_root: Path,
    *,
    support_filename: str,
    k: int,
    corpus_factory: Callable[[], SyntacticCorpus[ClausePsalm]] = clause_corpus,
) -> Draws:
    """Binds the clause corpus and a frozen support vocabulary for one clause construction."""
    psalms = corpus_factory().psalms()
    counts = load_external_signature_counts(config_root / support_filename)
    vocabulary = build_signature_vocabulary(counts, k)
    return Draws(
        key=key,
        psalms=tuple(psalms),
        permute=partial(within_half_verse, clause_rela_columns_for(counts, k)),
        build=partial(build_with_support, builder, vocabulary, counts, k),
        sparse_width=None,
    )


def load_clause(
    key: str,
    unit: str,
    construction: str,
    config_root: Path,
    *,
    sparse: bool,
    corpus_factory: Callable[[], SyntacticCorpus[ClausePsalm]] = clause_corpus,
) -> Draws:
    """Binds the clause corpus through the ordered-family record that carries its vocabulary."""
    psalms = corpus_factory().psalms()
    family = resolve_family(unit, config_root)
    return Draws(
        key=key,
        psalms=tuple(psalms),
        permute=partial(within_half_verse, family.permute_columns),
        build=partial(
            build_clause, sparse_vectors if sparse else dense_vectors, family, construction
        ),
        sparse_width=concatenated_1_2_3gram_dim(len(family.vocabulary)) if sparse else None,
    )


# --- the registry: one row per order-sensitive construction --------------------------------


def _row(key: str, loader: Callable[..., Draws], *args: object, **kwargs: object) -> Registry:
    """Binds one loader to its key, leaving only the config root for a caller to supply."""
    return {key: partial(loader, key, *args, **kwargs)}


def _rows(
    prefix: str, loader: Callable[..., Draws], builders: Mapping[str, object], **kwargs: object
) -> Registry:
    """One row per construction sharing a loader, keyed by the partition it is written under."""
    return {
        f"{prefix}/{name}": partial(loader, f"{prefix}/{name}", builder, **kwargs)
        for name, builder in builders.items()
    }


def _declared_rows(
    dataset: NgramDataset[Any],
    prefix: str,
    *,
    permute_columns: Callable[[Any], tuple[tuple[str, ...], ...]],
    corpus_factory: Callable[[], Any],
) -> Registry:
    """One row per order-sensitive construction a feature declares, read off its declaration."""
    return {
        f"{prefix}/{name}": partial(
            load_phrase_unit,
            f"{prefix}/{name}",
            construction.build,
            sparse_dim=len(dataset.vocabulary) if construction.sparse_dim else None,
            permute_columns=permute_columns,
            corpus_factory=corpus_factory,
        )
        for name, construction in order_sensitive_constructions(dataset)
    }


def _clause_rows(unit: str, sparse_by_construction: Mapping[str, bool]) -> Registry:
    """One row per clause construction of a unit, whose vocabulary its family record carries."""
    return {
        f"syntactic/clause/{unit}/{name}": partial(
            load_clause, f"syntactic/clause/{unit}/{name}", unit, name, sparse=sparse
        )
        for name, sparse in sparse_by_construction.items()
    }


TYP_WIDTH = len(TYP_VOCABULARY)
FUNCTION_WIDTH = len(FUNCTION_VOCABULARY)

LEXICAL_BUILDERS = {
    "icf_position4": partial(positional_icf_vectors, k=4),
    "icf_position_mean_psalm": psalm_position_mean_vectors,
}

MORPH_POS_BUILDERS = {
    "1_2gram": partial(
        ngram_vectors, columns_of=sp_columns, vocabulary=SP_VOCABULARY, orders=(1, 2)
    ),
    "1_2gram_psalm": partial(
        ngram_psalm_vectors, columns_of=sp_columns, vocabulary=SP_VOCABULARY, orders=(1, 2)
    ),
    "1_2_3gram": partial(
        ngram_vectors, columns_of=sp_columns, vocabulary=SP_VOCABULARY, orders=(1, 2, 3)
    ),
    "1_2_3gram_psalm": partial(
        ngram_psalm_vectors, columns_of=sp_columns, vocabulary=SP_VOCABULARY, orders=(1, 2, 3)
    ),
}

PHRASE_TYP_DENSE = {
    "1_2gram": partial(
        ngram_vectors, columns_of=typ_columns, vocabulary=TYP_VOCABULARY, orders=(1, 2)
    ),
    "1_2gram_psalm": partial(
        ngram_psalm_vectors, columns_of=typ_columns, vocabulary=TYP_VOCABULARY, orders=(1, 2)
    ),
}

PHRASE_TYP_SPARSE = {
    "1_2_3gram": partial(sparse_ngram_vectors, columns_of=typ_columns, vocabulary=TYP_VOCABULARY),
    "1_2_3gram_psalm": partial(
        sparse_ngram_psalm_vectors, columns_of=typ_columns, vocabulary=TYP_VOCABULARY
    ),
}

PHRASE_FUNCTION_DENSE = {
    "1_2gram": partial(
        ngram_vectors, columns_of=function_columns, vocabulary=FUNCTION_VOCABULARY, orders=(1, 2)
    ),
    "1_2gram_psalm": partial(
        ngram_psalm_vectors,
        columns_of=function_columns,
        vocabulary=FUNCTION_VOCABULARY,
        orders=(1, 2),
    ),
}

PHRASE_DET = {
    "1_2gram": phrase_det_1_2gram_vectors,
    "1_2gram_psalm": phrase_det_1_2gram_psalm_vectors,
    "1_2_3gram": phrase_det_1_2_3gram_vectors,
    "1_2_3gram_psalm": phrase_det_1_2_3gram_psalm_vectors,
}

PHRASE_RELA = {
    "1_2gram": phrase_rela_1_2gram_vectors,
    "1_2gram_psalm": phrase_rela_1_2gram_psalm_vectors,
    "1_2_3gram": phrase_rela_1_2_3gram_vectors,
    "1_2_3gram_psalm": phrase_rela_1_2_3gram_psalm_vectors,
}

PHRASE_SUBPHRASE_RELA = {
    "1_2gram": subphrase_rela_1_2gram_vectors,
    "1_2gram_psalm": subphrase_rela_1_2gram_psalm_vectors,
    "1_2_3gram": subphrase_rela_1_2_3gram_vectors,
    "1_2_3gram_psalm": subphrase_rela_1_2_3gram_psalm_vectors,
}

PHRASE_FUNCTION_SPARSE = {
    "1_2_3gram": partial(
        sparse_ngram_vectors, columns_of=function_columns, vocabulary=FUNCTION_VOCABULARY
    ),
    "1_2_3gram_psalm": partial(
        sparse_ngram_psalm_vectors, columns_of=function_columns, vocabulary=FUNCTION_VOCABULARY
    ),
}

#: Sparseness follows the construction, not the unit: only the 3-gram widths are stored sparse.
#: Every atomic morphology feature carries the same four order-sensitive constructions.
MORPH_FEATURES: tuple[FeatureKey, ...] = (
    "gn",
    "nu",
    "ps",
    "st",
    "vs",
    "vt",
    "prs_gn",
    "prs_nu",
    "prs_ps",
)


def _morph_feature_rows(feature: FeatureKey) -> Registry:
    """One feature's order-sensitive constructions, sparse where its trigram block demands it."""
    sparse = feature in SPARSE_TRIGRAM_FEATURES
    width = concatenated_1_2_3gram_dim(len(vocabulary_for_feature(feature))) if sparse else None
    prefix = f"morphological/morph_{feature}"
    builders = {
        "1_2gram": feature_1_2gram_vectors,
        "1_2gram_psalm": feature_1_2gram_psalm_vectors,
        "1_2_3gram": feature_sparse_trigram_vectors if sparse else feature_1_2_3gram_vectors,
        "1_2_3gram_psalm": (
            feature_sparse_trigram_psalm_vectors if sparse else feature_1_2_3gram_psalm_vectors
        ),
    }
    return {
        f"{prefix}/{name}": partial(
            load_morphological_feature,
            f"{prefix}/{name}",
            builder,
            feature,
            sparse_width=width if name.startswith("1_2_3gram") else None,
        )
        for name, builder in builders.items()
    }


CLAUSE_RELA_SUPPORT = "clause_rela_external_support.csv"

CLAUSE_RELA = {
    "1_2gram": clause_rela_1_2gram_vectors,
    "1_2gram_psalm": clause_rela_1_2gram_psalm_vectors,
    "1_2_3gram": clause_rela_1_2_3gram_vectors,
    "1_2_3gram_psalm": clause_rela_1_2_3gram_psalm_vectors,
}

CLAUSE_KIND = {
    "1_2gram": clause_kind_1_2gram_vectors,
    "1_2gram_psalm": clause_kind_1_2gram_psalm_vectors,
    "1_2_3gram": clause_kind_1_2_3gram_vectors,
    "1_2_3gram_psalm": clause_kind_1_2_3gram_psalm_vectors,
}

CLAUSE_NGRAMS = {
    "1_2gram": False,
    "1_2gram_psalm": False,
    "1_2_3gram": True,
    "1_2_3gram_psalm": True,
}

FAMILIES: Registry = {
    **_rows("lexical/homograph", load_lexical, LEXICAL_BUILDERS),
    **_rows("morphological/sp", load_morphological_pos, MORPH_POS_BUILDERS),
    **_rows(
        "morphological/morph_signature",
        load_morphological_signature,
        MORPH_SIGNATURE_DENSE,
        sparse=False,
    ),
    **_rows(
        "morphological/morph_signature",
        load_morphological_signature,
        MORPH_SIGNATURE_SPARSE,
        sparse=True,
    ),
    **_row("morphological/morph_suffix/posmean", load_morphological_deploy),
    **_rows("syntactic/phrase/typ", load_phrase_unit, PHRASE_TYP_DENSE, sparse_dim=None),
    **_rows("syntactic/phrase/typ", load_phrase_unit, PHRASE_TYP_SPARSE, sparse_dim=TYP_WIDTH),
    **_rows("syntactic/phrase/function", load_phrase_unit, PHRASE_FUNCTION_DENSE, sparse_dim=None),
    **_rows(
        "syntactic/phrase/function",
        load_phrase_unit,
        PHRASE_FUNCTION_SPARSE,
        sparse_dim=FUNCTION_WIDTH,
    ),
    **_rows(
        "syntactic/phrase/signature", load_phrase_signature, PHRASE_SIGNATURE_DENSE, sparse=False
    ),
    **_rows(
        "syntactic/phrase/signature", load_phrase_signature, PHRASE_SIGNATURE_SPARSE, sparse=True
    ),
    **_rows("syntactic/phrase/det", load_phrase_unit, PHRASE_DET, sparse_dim=None),
    **_rows("syntactic/phrase/rela", load_phrase_unit, PHRASE_RELA, sparse_dim=None),
    **_rows(
        "syntactic/phrase/subphrase_rela",
        load_phrase_unit,
        PHRASE_SUBPHRASE_RELA,
        sparse_dim=None,
        permute_columns=half_verse_subphrase_rela,
    ),
    **_row("syntactic/phrase/signature/posmean", load_phrase_deploy),
    **_rows(
        "syntactic/phrase/full_signature",
        partial(load_phrase_signature, support_filename=FULL_SIGNATURE_SUPPORT, k=FULL_SIGNATURE_K),
        {name: FULL_SIGNATURE_DENSE[name] for name in ("1_2gram", "1_2gram_psalm")},
        sparse=False,
    ),
    **_rows(
        "syntactic/phrase/full_signature",
        partial(load_phrase_signature, support_filename=FULL_SIGNATURE_SUPPORT, k=FULL_SIGNATURE_K),
        FULL_SIGNATURE_SPARSE,
        sparse=True,
    ),
    **_declared_rows(
        SUFFIX_DATASET,
        "morphological/morph_suffix",
        permute_columns=psalm_suffix_signatures,
        corpus_factory=MorphologicalCorpus.load,
    ),
    **_clause_rows("typ", CLAUSE_NGRAMS),
    **_clause_rows("signature", CLAUSE_NGRAMS),
    **_clause_rows("tab", {"transition_psalm": False}),
    **_rows(
        "syntactic/clause/rela",
        partial(load_clause_supported, support_filename=CLAUSE_RELA_SUPPORT, k=CLAUSE_RELA_K),
        CLAUSE_RELA,
    ),
    **_rows(
        "syntactic/clause/kind",
        load_phrase_unit,
        CLAUSE_KIND,
        sparse_dim=None,
        permute_columns=clause_kind_columns,
        corpus_factory=clause_corpus,
    ),
    **{key: row for feature in MORPH_FEATURES for key, row in _morph_feature_rows(feature).items()},
}


def load_draws(key: str, config_root: Path, *, families: Registry = FAMILIES) -> Draws:
    """The loaded corpus and bound builder for one family, refusing a key the registry lacks."""
    if key not in families:
        raise KeyError(f"{key!r} is not an order-sensitive construction; see FAMILIES")
    return families[key](config_root)
