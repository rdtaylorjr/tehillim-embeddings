from __future__ import annotations

from syntax.scripts.compute_signature_support import build_external_signature_counts


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
    def __init__(self, atoms_by_book: dict[int, list[int]], mother_by_atom: dict[int, int]) -> None:
        self._atoms_by_book = atoms_by_book
        self._mother_by_atom = mother_by_atom

    def d(self, node: int, otype: str) -> list[int]:
        assert otype == "phrase_atom"
        return self._atoms_by_book[node]

    def u(self, node: int, otype: str) -> list[int]:
        assert otype == "phrase"
        return [self._mother_by_atom[node]]


class _FakeF:
    def __init__(self, otype, book, typ, function) -> None:  # noqa: N803
        self.otype = otype
        self.book = book
        self.typ = typ
        self.function = function


class _FakeApi:
    def __init__(self, F: _FakeF, L: _FakeL) -> None:  # noqa: N803
        self.F = F
        self.L = L


def _fake_two_book_api() -> _FakeApi:
    # Book 1 = "Psalmi" (must be excluded): atom 10, mother phrase 100.
    # Book 2 = "Genesis" (included): atoms 20, 21, 22, mothers 200, 200, 201.
    otype = _FakeOtype([1, 2])
    book = _FakeFeature({1: "Psalmi", 2: "Genesis"})
    typ = _FakeFeature({10: "NP", 20: "NP", 21: "NP", 22: "VP"})
    function = _FakeFeature({100: "Subj", 200: "Subj", 201: "Pred"})
    L = _FakeL(  # noqa: N806
        atoms_by_book={1: [10], 2: [20, 21, 22]},
        mother_by_atom={10: 100, 20: 200, 21: 200, 22: 201},
    )
    return _FakeApi(_FakeF(otype, book, typ, function), L)


class TestBuildExternalSignatureCounts:
    def test_excludes_atoms_belonging_to_the_psalms_book(self) -> None:
        counts = build_external_signature_counts(_fake_two_book_api())

        # Only Genesis's 2 "NP:Subj" atoms and 1 "VP:Pred" atom should be counted; Psalms'
        # single "NP:Subj" atom must not contribute, so "NP:Subj" totals 2, not 3.
        assert counts == {"NP:Subj": 2, "VP:Pred": 1}

    def test_resolves_function_from_the_atom_s_mother_phrase_not_the_atom_itself(self) -> None:
        counts = build_external_signature_counts(_fake_two_book_api())

        # Atom 22 (typ=VP) is under mother 201 (function=Pred), giving VP:Pred, not VP:Subj.
        assert "VP:Pred" in counts
        assert "VP:Subj" not in counts
