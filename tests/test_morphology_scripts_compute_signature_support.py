from __future__ import annotations

from morphology.scripts.compute_signature_support import build_external_signature_counts


class _FakeFeature:
    def __init__(self, values: dict[int, str]) -> None:
        self._values = values

    def v(self, node: int) -> str:
        return self._values[node]


class _FakeOtype:
    def __init__(self, books: list[int]) -> None:
        self._books = books

    def s(self, otype: str) -> list[int]:
        assert otype == "book"
        return self._books


class _FakeL:
    def __init__(self, words_by_book: dict[int, list[int]]) -> None:
        self._words_by_book = words_by_book

    def d(self, node: int, otype: str) -> list[int]:
        assert otype == "word"
        return self._words_by_book[node]


class _FakeF:
    def __init__(self, otype, book, sp, gn, nu, ps, st, vs, vt) -> None:  # noqa: N803
        self.otype = otype
        self.book = book
        self.sp = sp
        self.gn = gn
        self.nu = nu
        self.ps = ps
        self.st = st
        self.vs = vs
        self.vt = vt


class _FakeApi:
    def __init__(self, F: _FakeF, L: _FakeL) -> None:  # noqa: N803
        self.F = F
        self.L = L


def _fake_two_book_api() -> _FakeApi:
    # Book 1 = "Psalmi" (must be excluded): words 10, 11.
    # Book 2 = "Genesis" (included): words 20, 21, 22.
    otype = _FakeOtype([1, 2])
    book = _FakeFeature({1: "Psalmi", 2: "Genesis"})
    all_na = {n: "NA" for n in (10, 11, 20, 21, 22)}
    sp = _FakeFeature({10: "verb", 11: "subs", 20: "subs", 21: "subs", 22: "verb"})
    gn = _FakeFeature(all_na)
    nu = _FakeFeature(all_na)
    ps = _FakeFeature(all_na)
    st = _FakeFeature(all_na)
    vs = _FakeFeature(all_na)
    vt = _FakeFeature(all_na)
    L = _FakeL({1: [10, 11], 2: [20, 21, 22]})  # noqa: N806
    return _FakeApi(_FakeF(otype, book, sp, gn, nu, ps, st, vs, vt), L)


class TestBuildExternalSignatureCounts:
    def test_excludes_words_belonging_to_the_psalms_book(self) -> None:
        counts = build_external_signature_counts(_fake_two_book_api())

        # Only Genesis's 2 "subs" words and 1 "verb" word should be counted; Psalms' 1 verb
        # and 1 subs must not contribute, so "subs" totals 2 (not 3) and "verb" totals 1 (not 2).
        assert counts == {"subs": 2, "verb": 1}
