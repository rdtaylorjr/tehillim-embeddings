"""Computes and writes semantic embedding features to Text-Fabric."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from semantic import api_models
from semantic.api_models import API_KEY_ENV_VARS
from semantic.export import feature_path, node_values, write_feature
from semantic.local_models import _select_half_verses, compute_half_verse_embeddings
from semantic.registry import (
    MODEL_REGISTRY,
    feature_description,
    feature_name,
    variations_for_model,
)

if TYPE_CHECKING:
    from semantic.corpus import Psalm

_API_SLUGS = {"gemini", "cohere", "openai", "voyage"}

#: slug -> real fetch function, used when `fetch` isn't passed.
_REAL_FETCHERS: dict[str, Callable[..., np.ndarray]] = {
    "gemini": api_models.fetch_gemini_embeddings,
    "cohere": api_models.fetch_cohere_embeddings,
    "openai": api_models.fetch_openai_embeddings,
    "voyage": api_models.fetch_voyage_embeddings,
}


def generate_local(
    psalms: list[Psalm],
    output_root: Path,
    slug: str,
    *,
    device: str | None = None,
    torch_dtype: str | None = None,
    compute: Callable[..., dict[int, np.ndarray]] = compute_half_verse_embeddings,
) -> list[str]:
    """Generates every not-yet-written variation for one local model slug."""
    technical_name = MODEL_REGISTRY[slug][0]
    written: list[str] = []
    for variation, vocalized, niqqud_only, variation_description in variations_for_model(slug):
        name = feature_name(slug, variation)
        if feature_path(output_root, name).exists():
            continue
        print(f"computing {name} from {technical_name}...", file=sys.stderr)
        embeddings = compute(
            psalms,
            technical_name,
            vocalized=vocalized,
            niqqud_only=niqqud_only,
            device=device,
            torch_dtype=torch_dtype,
        )
        values = node_values(embeddings, psalms)
        write_feature(output_root, name, values, feature_description(slug, variation_description))
        written.append(name)
    return written


def generate_api(
    psalms: list[Psalm],
    output_root: Path,
    slug: str,
    *,
    fetch: Callable[..., np.ndarray] | None = None,
    env: dict[str, str] | None = None,
) -> list[str]:
    """Reads the API key only if a variation is actually missing."""
    if fetch is None:
        fetch = _REAL_FETCHERS[slug]
    if env is None:
        env = dict(os.environ)

    written: list[str] = []
    for variation, vocalized, niqqud_only, variation_description in variations_for_model(slug):
        name = feature_name(slug, variation)
        if feature_path(output_root, name).exists():
            continue

        env_var = API_KEY_ENV_VARS[slug]
        api_key = env.get(env_var)
        if not api_key:
            raise RuntimeError(f"{env_var} is not set, cannot fetch {slug} embeddings")

        texts: list[str] = []
        spans: list[tuple[int, int, int]] = []
        for psalm in psalms:
            half_verses = _select_half_verses(psalm, vocalized=vocalized, niqqud_only=niqqud_only)
            start = len(texts)
            texts.extend(half_verses)
            spans.append((psalm.number, start, len(texts)))

        print(f"fetching {name} from {slug}...", file=sys.stderr)
        vectors = fetch(texts, api_key=api_key)
        embeddings = {number: vectors[start:end] for number, start, end in spans}
        values = node_values(embeddings, psalms)
        write_feature(output_root, name, values, feature_description(slug, variation_description))
        written.append(name)
    return written


def main() -> None:
    """Generates every missing feature for every registered model."""
    from semantic.corpus import Corpus

    output_root = Path(__file__).resolve().parents[3]
    psalms = Corpus.load().psalms()

    written: list[str] = []
    for slug in MODEL_REGISTRY:
        if slug in _API_SLUGS:
            written.extend(generate_api(psalms, output_root, slug))
        else:
            written.extend(generate_local(psalms, output_root, slug))

    print(f"wrote {len(written)} feature files", file=sys.stderr)


if __name__ == "__main__":
    main()
