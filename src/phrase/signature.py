"""Per-phrase-atom signatures: the joint (typ, function) conjunction, e.g. `NP:Subj`."""

from __future__ import annotations

from phrase.corpus import PhrasePsalm


def build_phrase_signature(*, typ: str, function: str) -> str:
    """`typ:function`-style signature, e.g. `NP:Subj`; both fields always present, never NA."""
    return f"{typ}:{function}"


def build_full_phrase_signature(*, typ: str, function: str, det: str) -> str:
    """`typ:function[:det]`-style signature; `det` appended only when not NA (H5.8's S+det)."""
    base = build_phrase_signature(typ=typ, function=function)
    return base if det == "NA" else f"{base}:{det}"


def colon_signatures(*, typ: tuple[str, ...], function: tuple[str, ...]) -> tuple[str, ...]:
    """One signature per phrase atom, aligned across the two per-atom feature sequences."""
    return tuple(
        build_phrase_signature(typ=w_typ, function=w_function)
        for w_typ, w_function in zip(typ, function, strict=True)
    )


def psalm_signatures(psalm: PhrasePsalm) -> tuple[tuple[str, ...], ...]:
    """One signature sequence per colon of `psalm`."""
    return tuple(
        colon_signatures(typ=typ, function=function)
        for typ, function in zip(psalm.half_verse_typ, psalm.half_verse_function, strict=True)
    )


def colon_full_signatures(
    *, typ: tuple[str, ...], function: tuple[str, ...], det: tuple[str, ...]
) -> tuple[str, ...]:
    """One `typ:function[:det]` signature per phrase atom, aligned across all three features."""
    return tuple(
        build_full_phrase_signature(typ=w_typ, function=w_function, det=w_det)
        for w_typ, w_function, w_det in zip(typ, function, det, strict=True)
    )


def psalm_full_signatures(psalm: PhrasePsalm) -> tuple[tuple[str, ...], ...]:
    """One `typ:function[:det]` signature sequence per colon of `psalm`."""
    return tuple(
        colon_full_signatures(typ=typ, function=function, det=det)
        for typ, function, det in zip(
            psalm.half_verse_typ, psalm.half_verse_function, psalm.half_verse_det, strict=True
        )
    )
