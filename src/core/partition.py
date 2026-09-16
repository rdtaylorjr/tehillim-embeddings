"""The one grammar of the data tree: every path a writer makes and every path a reader parses."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields
from pathlib import Path, PurePosixPath

PART_FILE = "part-0.parquet"

#: The corpora rows can be keyed to, each a Text-Fabric dataset with its own node numbering.
BHSA = "bhsa"
DSS = "dss"

#: A corpus that holds several witnesses names one under `witness=`, and its letters under
#: `reconstruction=`; a corpus that is one text has neither key.
MULTI_WITNESS_CORPORA: frozenset[str] = frozenset({DSS})

#: The unit a row is keyed to, as the corpus names its node type.
HALF_VERSE = "half_verse"

LEXICAL = "lexical"
MORPHOLOGICAL = "morphological"
SYNTACTIC = "syntactic"
SEMANTIC = "semantic"
DOMAINS: tuple[str, ...] = (LEXICAL, MORPHOLOGICAL, SYNTACTIC, SEMANTIC)

#: The lexical type whose datasets carry a text tier: surface forms exist per text state.
SURFACE_TYPE = "word"

#: Every key of the tree in path order; a partition's segments appear in this order and no other.
KEYS: tuple[str, ...] = (
    "corpus",
    "witness",
    "reconstruction",
    "unit",
    "domain",
    "level",
    "type",
    "feature",
    "model",
    "text",
    "construction",
)


class PartitionError(ValueError):
    """A partition that the grammar does not admit, or a path that spells none."""


@dataclass(frozen=True, slots=True)
class Scope:
    """What the rows of a partition are: the corpus, the witness, its letters, and the unit."""

    corpus: str
    unit: str
    witness: str | None = None
    reconstruction: str | None = None

    def __post_init__(self) -> None:
        """A multi-witness corpus names witness and reconstruction; a single text has neither."""
        several = self.corpus in MULTI_WITNESS_CORPORA
        if several and (self.witness is None or self.reconstruction is None):
            raise PartitionError(
                f"corpus={self.corpus} holds several witnesses: witness and reconstruction required"
            )
        if not several and (self.witness is not None or self.reconstruction is not None):
            raise PartitionError(f"corpus={self.corpus} is one text: no witness or reconstruction")

    @property
    def directory(self) -> str:
        """The Hive-relative directory every partition of this scope sits under."""
        return "/".join(
            f"{key}={value}"
            for key in KEYS
            if key in _SCOPE_KEYS and (value := getattr(self, key)) is not None
        )


#: The only scope this repository writes today: the Masoretic Psalms at the accentual half-verse.
BHSA_HALF_VERSE = Scope(BHSA, HALF_VERSE)


@dataclass(frozen=True, slots=True)
class Partition:
    """One dataset's place in the tree: its scope and the representation keys its domain takes."""

    scope: Scope
    domain: str
    level: str | None = None
    type: str | None = None
    feature: str | None = None
    model: str | None = None
    text: str | None = None
    construction: str | None = None

    def __post_init__(self) -> None:
        """Each domain takes exactly its own keys."""
        if self.domain not in _DOMAIN_KEYS:
            raise PartitionError(f"domain={self.domain} is not one of {DOMAINS}")
        required, optional = _DOMAIN_KEYS[self.domain]
        present = {k for k in _REPRESENTATION_KEYS if getattr(self, k) is not None}
        missing = set(required) - present
        foreign = present - set(required) - set(optional)
        if missing or foreign:
            raise PartitionError(
                f"domain={self.domain} takes {required} and optionally {optional}: "
                f"missing {sorted(missing)}, foreign {sorted(foreign)}"
            )
        if self.domain == LEXICAL and (self.text is not None) != (self.type == SURFACE_TYPE):
            raise PartitionError(f"lexical text tier belongs to type={SURFACE_TYPE} only")

    def values(self) -> Iterator[tuple[str, str]]:
        """Every present key with its value, in path order."""
        for key in KEYS:
            value = getattr(self.scope, key) if key in _SCOPE_KEYS else getattr(self, key)
            if value is not None:
                yield key, value

    @property
    def directory(self) -> str:
        """The Hive-relative directory, `key=value` segments in tree order."""
        return "/".join(f"{key}={value}" for key, value in self.values())

    def file(self, root: Path) -> Path:
        """The one Parquet file the partition holds, under a data root."""
        return root / self.directory / PART_FILE

    @property
    def identifier(self) -> str:
        """The name consumers know a dataset by: its values below `domain=`, underscore-joined."""
        return "_".join(value for key, value in self.values() if key in _REPRESENTATION_KEYS)

    def metadata(self) -> dict[str, str]:
        """The partition's keys as Parquet schema metadata, so a file says where it belongs."""
        return dict(self.values())

    @classmethod
    def parse(cls, path: Path | str) -> Partition:
        """Reads a partition back from a file or directory path, refusing what the grammar does."""
        found: dict[str, str] = {}
        for segment in PurePosixPath(Path(path).as_posix()).parts:
            if segment == PART_FILE:
                continue
            key, equals, value = segment.partition("=")
            if not equals:
                continue
            if key not in KEYS or not value:
                raise PartitionError(f"{path}: {segment!r} is not a key of the tree")
            if key in found:
                raise PartitionError(f"{path}: {key}= appears twice")
            found[key] = value
        if not found:
            raise PartitionError(f"{path} carries no Hive partition, so it names no dataset")
        ordered = [key for key in KEYS if key in found]
        if ordered != list(found):
            raise PartitionError(f"{path}: keys out of tree order, expected {ordered}")
        return cls.from_mapping(found, source=str(path))

    @classmethod
    def from_mapping(cls, keys: Mapping[str, str], *, source: str = "") -> Partition:
        """Builds a partition from key-value pairs, as parsed from a path or read from metadata."""
        unknown = set(keys) - set(KEYS)
        if unknown:
            raise PartitionError(f"{source or keys}: unknown keys {sorted(unknown)}")
        try:
            scope = Scope(
                corpus=keys["corpus"],
                unit=keys["unit"],
                witness=keys.get("witness"),
                reconstruction=keys.get("reconstruction"),
            )
            return cls(scope, keys["domain"], **{k: keys.get(k) for k in _REPRESENTATION_KEYS})
        except KeyError as error:
            raise PartitionError(f"{source or dict(keys)}: missing {error}") from None


_SCOPE_KEYS: frozenset[str] = frozenset(f.name for f in fields(Scope))
_REPRESENTATION_KEYS: tuple[str, ...] = tuple(
    f.name for f in fields(Partition) if f.name not in ("scope", "domain")
)

#: (required, optional) representation keys per domain.
_DOMAIN_KEYS: Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    LEXICAL: (("type", "construction"), ("text",)),
    MORPHOLOGICAL: (("feature", "construction"), ()),
    SYNTACTIC: (("level", "feature", "construction"), ()),
    SEMANTIC: (("model", "text"), ()),
}


def family(
    scope: Scope,
    domain: str,
    constructions: tuple[str, ...] = (),
    *,
    level: str | None = None,
    type: str | None = None,  # noqa: A002 -- the tree's key, named as the tree names it
    feature: str | None = None,
    model: str | None = None,
    text: str | None = None,
) -> tuple[Partition, ...]:
    """One partition per construction, or the single partition of a domain that has none."""
    shared = {"level": level, "type": type, "feature": feature, "model": model, "text": text}
    if not constructions:
        return (Partition(scope, domain, **shared),)
    if len(set(constructions)) != len(constructions):
        raise PartitionError(f"{domain}: duplicate constructions {constructions}")
    return tuple(Partition(scope, domain, construction=c, **shared) for c in constructions)
