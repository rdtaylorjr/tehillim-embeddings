"""Per-word grammatical signatures: sp plus every applicable field, NA omitted, unknown kept."""

from __future__ import annotations

from morphology.corpus import MorphologicalPsalm

_SIGNATURE_FIELD_ORDER: tuple[str, ...] = ("vs", "vt", "ps", "gn", "nu", "st")


def build_signature(*, sp: str, gn: str, nu: str, ps: str, st: str, vs: str, vt: str) -> str:
    """`sp|vs|vt|ps|gn|nu|st`-style signature, e.g. `verb|qal|perf|p3|m|sg`, NA fields dropped."""
    values_by_field = {"vs": vs, "vt": vt, "ps": ps, "gn": gn, "nu": nu, "st": st}
    parts = [sp]
    for field in _SIGNATURE_FIELD_ORDER:
        value = values_by_field[field]
        if value != "NA":
            parts.append(value)
    return "|".join(parts)


def colon_signatures(
    *,
    sp: tuple[str, ...],
    gn: tuple[str, ...],
    nu: tuple[str, ...],
    ps: tuple[str, ...],
    st: tuple[str, ...],
    vs: tuple[str, ...],
    vt: tuple[str, ...],
) -> tuple[str, ...]:
    """One signature per word, aligned across the seven per-word feature sequences."""
    return tuple(
        build_signature(sp=w_sp, gn=w_gn, nu=w_nu, ps=w_ps, st=w_st, vs=w_vs, vt=w_vt)
        for w_sp, w_gn, w_nu, w_ps, w_st, w_vs, w_vt in zip(sp, gn, nu, ps, st, vs, vt, strict=True)
    )


def psalm_signatures(psalm: MorphologicalPsalm) -> tuple[tuple[str, ...], ...]:
    """One signature sequence per colon of `psalm`."""
    return tuple(
        colon_signatures(sp=sp, gn=gn, nu=nu, ps=ps, st=st, vs=vs, vt=vt)
        for sp, gn, nu, ps, st, vs, vt in zip(
            psalm.half_verse_sp,
            psalm.half_verse_gn,
            psalm.half_verse_nu,
            psalm.half_verse_ps,
            psalm.half_verse_st,
            psalm.half_verse_vs,
            psalm.half_verse_vt,
            strict=True,
        )
    )
