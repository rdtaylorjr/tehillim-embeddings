"""Shared BHSA Text-Fabric loading and Psalms half-verse iteration for every domain's corpus."""

from __future__ import annotations

import os
import queue
import threading
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_BHSA_CLONE = Path.home() / "Developer" / "hebrew" / "bhsa" / "tf" / "2021"
BHSA_PATH_ENV = "TEHILLIM_BHSA_PATH"
#: use() re-verifies the release against GitHub even when cached, which can hang for minutes.
DEFAULT_USE_TIMEOUT_SECONDS = 30.0
DEFAULT_CHECKOUT = "v1.8.1"

PSALMS_BOOK_NAME = "Psalmi"


def bhsa_clone_location(env: Mapping[str, str] | None = None) -> Path:
    """The local BHSA directory: $TEHILLIM_BHSA_PATH when set, else the conventional clone."""
    environment = os.environ if env is None else env
    return Path(environment.get(BHSA_PATH_ENV) or DEFAULT_BHSA_CLONE)


def _real_fabric(locations: list[str], silent: str) -> Any:
    """Returns a real Text-Fabric `Fabric` for `locations`."""
    from tf.fabric import Fabric

    return Fabric(locations=locations, silent=silent)


def _real_use(spec: str, checkout: str, silent: str) -> Any:
    """Returns a real Text-Fabric app downloaded through `use()`."""
    from tf.app import use

    return use(spec, checkout=checkout, silent=silent)


def _call_with_timeout(
    fn: Callable[..., Any], timeout_seconds: float, /, *args: Any, **kwargs: Any
) -> Any:
    """Runs fn in a daemon thread, returning its result, the error it raised, or None on timeout."""
    result: queue.Queue[Any] = queue.Queue(maxsize=1)

    def _run() -> None:
        try:
            result.put(fn(*args, **kwargs))
        except Exception as error:  # noqa: BLE001 -- returned to the caller rather than swallowed
            result.put(error)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)
    if thread.is_alive():
        return None
    return result.get_nowait()


def _as_api(result: Any) -> Any:
    """Text-Fabric reports failure as False and bare success as True, so demand a real api."""
    return result if getattr(result, "F", None) is not None else None


def _load_local(path: Path, required_features: str, fabric: Callable[..., Any]) -> Any:
    """Loads required_features from a local Text-Fabric directory, or None if unavailable."""
    if not path.exists():
        return None
    loaded = fabric(locations=[str(path)], silent="deep").load(required_features, silent="deep")
    return _as_api(loaded)


def _is_loaded(api: Any, name: str) -> bool:
    """Text-Fabric puts node features on F and edge features on E, so a name can be on either."""
    return hasattr(api.F, name) or hasattr(getattr(api, "E", None), name)


def load_api(
    path: Path | None = None,
    required_features: str = "",
    *,
    fabric: Callable[..., Any] = _real_fabric,
    use_fn: Callable[..., Any] = _real_use,
    env: Mapping[str, str] | None = None,
    checkout: str = DEFAULT_CHECKOUT,
    timeout_seconds: float = DEFAULT_USE_TIMEOUT_SECONDS,
) -> Any:
    """Loads `required_features` from the local BHSA clone, falling back to Text-Fabric's use()."""
    location = path if path is not None else bhsa_clone_location(env)
    try:
        api = _load_local(location, required_features, fabric)
    except Exception as error:  # noqa: BLE001 -- Text-Fabric raises anything; fall back regardless
        api = None
        reason: object = error
    else:
        reason = "no usable Text-Fabric data there" if api is None else None
    if api is None:
        #: A silent fallback here looks identical to a cold cache and hides real loader bugs.
        warnings.warn(
            f"local BHSA clone at {location} unusable ({reason!r}), falling back to use()",
            RuntimeWarning,
            stacklevel=2,
        )
        outcome = _call_with_timeout(use_fn, timeout_seconds, "etcbc/bhsa", checkout, "deep")
        app = None if isinstance(outcome, BaseException) else outcome
        use_reason: object = outcome if isinstance(outcome, BaseException) else None
        if app is None and use_reason is None:
            use_reason = f"timed out after {timeout_seconds}s"
        api = _as_api(getattr(app, "api", None)) if app is not None else None
        if api is not None and required_features:
            api.TF.load(required_features, add=True, silent="deep")
        if api is None and use_reason is None:
            use_reason = "returned no usable api"
    if api is None:
        raise RuntimeError(
            f"Text-Fabric failed to load BHSA from {location} "
            f"or via use(checkout={checkout!r}): {use_reason!r}"
        )
    missing = [name for name in required_features.split() if not _is_loaded(api, name)]
    if missing:
        raise RuntimeError(f"Text-Fabric did not load required features: {missing}")
    return api


#: One entry, because a loaded corpus is gigabytes and callers group their families by corpus.
@lru_cache(maxsize=1)
def shared_api(
    path: Path | None = None,
    required_features: str = "",
    *,
    loader: Callable[..., Any] = load_api,
) -> Any:
    """One loaded API per (path, feature set), so a sweep over families loads BHSA once each."""
    return loader(path, required_features)


class BaseCorpus[PsalmT](ABC):
    """A loaded BHSA corpus that walks the Psalms' chapters and half-verse nodes once."""

    def __init__(self, api: Any) -> None:
        """Wraps an already-loaded Text-Fabric API."""
        self._api = api

    @property
    def api(self) -> Any:
        """The underlying Text-Fabric API, for whole-corpus queries outside Psalms."""
        return self._api

    @abstractmethod
    def _extract(self, number: int, half_verse_nodes: tuple[int, ...]) -> PsalmT:
        """Builds one domain's psalm record from a psalm number and its half-verse nodes."""

    def psalms(self) -> list[PsalmT]:
        """Extracts all 150 psalms, in canonical order."""
        F, L, T = self._api.F, self._api.L, self._api.T  # noqa: N806

        book_nodes = [b for b in F.otype.s("book") if F.book.v(b) == PSALMS_BOOK_NAME]
        if not book_nodes:
            raise RuntimeError(f"Book '{PSALMS_BOOK_NAME}' not found in loaded corpus")

        numbered: list[tuple[int, PsalmT]] = []
        for chapter_node in L.d(book_nodes[0], otype="chapter"):
            _, number = T.sectionFromNode(chapter_node)
            half_verse_nodes = tuple(L.d(chapter_node, otype="half_verse"))
            numbered.append((number, self._extract(number, half_verse_nodes)))

        numbered.sort(key=lambda item: item[0])
        return [psalm for _, psalm in numbered]
