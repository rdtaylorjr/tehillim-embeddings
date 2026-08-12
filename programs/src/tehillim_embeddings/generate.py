"""Computes and writes semantic embedding features for the Hebrew
Psalms, one (model, tier) pair at a time. A feature's `.tf` file
existing on disk is treated as that pair already being done; run this
script again after adding a new model or finishing a Colab run and it
picks up only what's missing.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from tehillim_embeddings import api_providers
from tehillim_embeddings.api_providers import API_KEY_ENV_VARS
from tehillim_embeddings.export import feature_path, node_values, write_feature
from tehillim_embeddings.local_models import _select_half_verses, compute_half_verse_embeddings
from tehillim_embeddings.registry import (
    MODEL_REGISTRY,
    feature_description,
    feature_name,
    tiers_for_model,
)

if TYPE_CHECKING:
    from tehillim_pipeline.corpus import Psalm

_API_SLUGS = {"gemini", "cohere", "openai", "voyage"}

#: slug -> real fetch function, the default `fetch` argument for
#: `generate_api`. Tests pass their own fake instead of relying on this
#: default.
_REAL_FETCHERS: dict[str, Callable[..., np.ndarray]] = {
    "gemini": api_providers.fetch_gemini_embeddings,
    "cohere": api_providers.fetch_cohere_embeddings,
    "openai": api_providers.fetch_openai_embeddings,
    "voyage": api_providers.fetch_voyage_embeddings,
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
    """Generates every not-yet-written tier for one local model slug.
    Returns the feature names it wrote.
    """
    technical_name = MODEL_REGISTRY[slug][0]
    written: list[str] = []
    for tier, vocalized, niqqud_only, tier_description in tiers_for_model(slug):
        name = feature_name(slug, tier)
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
        write_feature(output_root, name, values, feature_description(slug, tier_description))
        written.append(name)
    return written


def generate_api(
    psalms: list[Psalm],
    output_root: Path,
    slug: str,
    *,
    fetch: Callable[..., np.ndarray] | None = None,
) -> list[str]:
    """Generates every not-yet-written tier for one API provider slug.
    Reads the provider's API key only if a tier is actually missing.
    `fetch` defaults to that provider's real fetch function.
    """
    if fetch is None:
        fetch = _REAL_FETCHERS[slug]

    written: list[str] = []
    for tier, vocalized, niqqud_only, tier_description in tiers_for_model(slug):
        name = feature_name(slug, tier)
        if feature_path(output_root, name).exists():
            continue

        env_var = API_KEY_ENV_VARS[slug]
        api_key = os.environ.get(env_var)
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
        write_feature(output_root, name, values, feature_description(slug, tier_description))
        written.append(name)
    return written


def main() -> None:
    from tehillim_pipeline.corpus import Corpus

    output_root = Path(__file__).resolve().parents[2]
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
