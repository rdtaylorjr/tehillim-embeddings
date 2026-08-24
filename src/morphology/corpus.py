"""Loads BHSA word-level morphology features per colon for the Psalms via Text-Fabric."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tf.fabric import Fabric

DEFAULT_BHSA_TF_PATH = Path.home() / "Developer" / "hebrew" / "bhsa" / "tf" / "2021"

_REQUIRED_FEATURES = "otype book chapter verse sp gn nu ps st vs vt prs_gn prs_nu prs_ps"

_PSALMS_BOOK_NAME = "Psalmi"

_MORPHOLOGY_FEATURES = ("sp", "gn", "nu", "ps", "st", "vs", "vt", "prs_gn", "prs_nu", "prs_ps")


@dataclass(frozen=True, slots=True)
class MorphologicalPsalm:
    """One psalm's colon morphology feature sequences, aligned word-for-word, and node ids."""

    number: int
    colon_nodes: tuple[int, ...] = ()
    colon_sp: tuple[tuple[str, ...], ...] = ()
    colon_gn: tuple[tuple[str, ...], ...] = ()
    colon_nu: tuple[tuple[str, ...], ...] = ()
    colon_ps: tuple[tuple[str, ...], ...] = ()
    colon_st: tuple[tuple[str, ...], ...] = ()
    colon_vs: tuple[tuple[str, ...], ...] = ()
    colon_vt: tuple[tuple[str, ...], ...] = ()
    colon_prs_gn: tuple[tuple[str, ...], ...] = ()
    colon_prs_nu: tuple[tuple[str, ...], ...] = ()
    colon_prs_ps: tuple[tuple[str, ...], ...] = ()


class Corpus:
    """A loaded BHSA Text-Fabric corpus, scoped to colon morphology feature extraction."""

    def __init__(self, api: Any) -> None:
        self._api = api

    @property
    def api(self) -> Any:
        """The underlying Text-Fabric API, for whole-corpus queries outside Psalms."""
        return self._api

    @classmethod
    def load(cls, tf_path: Path | None = None) -> Corpus:
        """Loads BHSA from `tf_path`, or `DEFAULT_BHSA_TF_PATH`."""
        path = tf_path or DEFAULT_BHSA_TF_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"BHSA Text-Fabric data not found at {path}. "
                "Clone https://github.com/ETCBC/bhsa and pass its tf/<version> directory."
            )
        tf = Fabric(locations=[str(path)], silent="deep")
        api = tf.load(_REQUIRED_FEATURES, silent="deep")
        if api is None:
            raise RuntimeError(f"Text-Fabric failed to load required features from {path}")
        return cls(api)

    def psalms(self) -> list[MorphologicalPsalm]:
        """Extracts all 150 psalms' colon morphology feature sequences, in canonical order."""
        F, L, T = self._api.F, self._api.L, self._api.T  # noqa: N806

        book_nodes = [b for b in F.otype.s("book") if F.book.v(b) == _PSALMS_BOOK_NAME]
        if not book_nodes:
            raise RuntimeError(f"Book '{_PSALMS_BOOK_NAME}' not found in loaded corpus")

        psalms: list[MorphologicalPsalm] = []
        for chapter_node in L.d(book_nodes[0], otype="chapter"):
            _, psalm_number = T.sectionFromNode(chapter_node)
            colon_nodes = L.d(chapter_node, otype="half_verse")
            words_by_colon = [L.d(hv, otype="word") for hv in colon_nodes]

            colon_by_feature = {
                feature: tuple(
                    tuple(getattr(F, feature).v(w) for w in words) for words in words_by_colon
                )
                for feature in _MORPHOLOGY_FEATURES
            }

            psalms.append(
                MorphologicalPsalm(
                    number=psalm_number,
                    colon_nodes=tuple(colon_nodes),
                    colon_sp=colon_by_feature["sp"],
                    colon_gn=colon_by_feature["gn"],
                    colon_nu=colon_by_feature["nu"],
                    colon_ps=colon_by_feature["ps"],
                    colon_st=colon_by_feature["st"],
                    colon_vs=colon_by_feature["vs"],
                    colon_vt=colon_by_feature["vt"],
                    colon_prs_gn=colon_by_feature["prs_gn"],
                    colon_prs_nu=colon_by_feature["prs_nu"],
                    colon_prs_ps=colon_by_feature["prs_ps"],
                )
            )

        psalms.sort(key=lambda p: p.number)
        return psalms
