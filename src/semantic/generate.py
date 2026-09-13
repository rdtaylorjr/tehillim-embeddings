"""Computes and writes semantic embedding datasets as Parquet."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from core.cli import add_output_root_argument, report_generated
from core.export import write_vectors
from core.spec import GeneratorSpec, partition_dirs
from semantic import api_models
from semantic.api_models import API_KEY_ENV_VARS
from semantic.export import node_vectors, path_to_write
from semantic.large_models import LARGE_MODELS
from semantic.local_models import compute_half_verse_embeddings, select_half_verses
from semantic.registry import (
    MODEL_REGISTRY,
    dataset_description,
    dataset_name,
    variations_for_model,
)

if TYPE_CHECKING:
    from semantic.corpus import Corpus, SemanticPsalm

_API_SLUGS = {"gemini", "cohere", "openai", "voyage"}
_GPU_SLUGS = {slug for slug, _, _ in LARGE_MODELS}
#: The dtype each large model loads in, so a driver cell runs it as the Colab notebook did.
_LARGE_DTYPES = {slug: dtype for slug, _, dtype in LARGE_MODELS}


def resource_for(slug: str) -> str | None:
    """The serial resource a model's cell holds: an API for hosted models, a GPU for large ones."""
    if slug in _API_SLUGS:
        return "api"
    if slug in _GPU_SLUGS:
        return "gpu"
    return None


def _spec_for(slug: str) -> GeneratorSpec:
    """One model's cell: its text partitions and the resource its generation holds."""
    model_slug = MODEL_REGISTRY[slug][1]
    partitions = tuple(
        p
        for tier, _ in variations_for_model(slug)
        for p in partition_dirs("semantic", "model", model_slug, (), text=tier)
    )
    return GeneratorSpec(
        module=f"{__name__}:{slug}", partitions=partitions, resource=resource_for(slug)
    )


SPECS: tuple[GeneratorSpec, ...] = tuple(_spec_for(slug) for slug in MODEL_REGISTRY)

#: slug -> real fetch function, used when `fetch` isn't passed.
_REAL_FETCHERS: dict[str, Callable[..., np.ndarray]] = {
    "gemini": api_models.fetch_gemini_embeddings,
    "cohere": api_models.fetch_cohere_embeddings,
    "openai": api_models.fetch_openai_embeddings,
    "voyage": api_models.fetch_voyage_embeddings,
}


def generate_local(
    psalms: list[SemanticPsalm],
    output_root: Path,
    slug: str,
    *,
    variation: str | None = None,
    device: str | None = None,
    torch_dtype: str | None = None,
    compute: Callable[..., dict[int, np.ndarray]] = compute_half_verse_embeddings,
) -> list[str]:
    """Generates every not-yet-written variation for one local model slug, or only `variation`."""
    technical_name, model_slug, _ = MODEL_REGISTRY[slug]
    written: list[str] = []
    for variation_name, variation_description in variations_for_model(slug):
        if variation is not None and variation_name != variation:
            continue
        name = dataset_name(slug, variation_name)
        path = path_to_write(output_root, model_slug, variation_name)
        if path is None:
            continue
        embeddings = compute(
            psalms,
            technical_name,
            tier=variation_name,
            device=device,
            torch_dtype=torch_dtype,
        )
        values = node_vectors(embeddings, psalms)
        description = dataset_description(slug, variation_description)
        write_vectors(path, values, description)
        written.append(name)
    return written


def generate_api(
    psalms: list[SemanticPsalm],
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

    model_slug = MODEL_REGISTRY[slug][1]
    written: list[str] = []
    for variation, variation_description in variations_for_model(slug):
        name = dataset_name(slug, variation)
        path = path_to_write(output_root, model_slug, variation)
        if path is None:
            continue

        env_var = API_KEY_ENV_VARS[slug]
        api_key = env.get(env_var)
        if not api_key:
            raise RuntimeError(f"{env_var} is not set, cannot fetch {slug} embeddings")

        texts: list[str] = []
        spans: list[tuple[int, int, int]] = []
        for psalm in psalms:
            half_verses = select_half_verses(psalm, variation)
            start = len(texts)
            texts.extend(half_verses)
            spans.append((psalm.number, start, len(texts)))

        print(f"fetching {name} from {slug}...", file=sys.stderr)
        vectors = fetch(texts, api_key=api_key)
        embeddings = {number: vectors[start:end] for number, start, end in spans}
        values = node_vectors(embeddings, psalms)
        description = dataset_description(slug, variation_description)
        write_vectors(path, values, description)
        written.append(name)
    return written


def generate(
    psalms: list[SemanticPsalm],
    output_root: Path,
    slugs: tuple[str, ...] = tuple(MODEL_REGISTRY),
    *,
    local: Callable[..., list[str]] = generate_local,
    api: Callable[..., list[str]] = generate_api,
) -> list[str]:
    """Writes every missing dataset for the named models, hosted API and local alike."""
    written: list[str] = []
    for slug in slugs:
        if slug in _API_SLUGS:
            written.extend(api(psalms, output_root, slug))
        elif slug in _GPU_SLUGS:
            written.extend(local(psalms, output_root, slug, torch_dtype=_LARGE_DTYPES[slug]))
        else:
            written.extend(local(psalms, output_root, slug))
    return written


def load_corpus() -> Corpus:
    """Imported at call time because the semantic corpus pulls in the local model stack."""
    from semantic.corpus import Corpus

    return Corpus.load()


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus] = load_corpus,
    local: Callable[..., list[str]] = generate_local,
    api: Callable[..., list[str]] = generate_api,
) -> None:
    """Generates every missing dataset for every registered model, or for one named model."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_output_root_argument(parser)
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY), default=None)
    args = parser.parse_args(argv)
    slugs = tuple(MODEL_REGISTRY) if args.model is None else (args.model,)
    written = generate(corpus_factory().psalms(), args.output_root, slugs, local=local, api=api)
    report_generated(written)


if __name__ == "__main__":
    main()
