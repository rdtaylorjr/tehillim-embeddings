"""Computes and writes semantic embedding datasets as Parquet, one source and model per cell."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from core.cli import add_output_root_argument, report_generated
from core.export import write_vectors
from core.partition import Scope
from core.spec import GeneratorSpec
from semantic import api_models, local_models
from semantic.api_models import API_KEY_ENV_VARS
from semantic.export import partition, path_to_write
from semantic.large_models import LARGE_MODELS
from semantic.registry import (
    MODEL_REGISTRY,
    dataset_description,
    dataset_name,
    variations_for_model,
)
from semantic.sources import SOURCES, Source, scope_arguments, source_for
from semantic.units import Encoder, Unit, embed

if TYPE_CHECKING:
    from core.text import TextTier

_API_SLUGS = {"gemini", "cohere", "openai", "voyage"}
_GPU_SLUGS = {slug for slug, _, _ in LARGE_MODELS}
#: The dtype each large model loads in, so a driver cell runs it as the Colab notebook did.
_LARGE_DTYPES = {slug: dtype for slug, _, dtype in LARGE_MODELS}

#: slug -> hosted fetch function, each `(texts, api_key=...) -> rows`.
_FETCHERS: dict[str, Callable[..., np.ndarray]] = {
    "gemini": api_models.fetch_gemini_embeddings,
    "cohere": api_models.fetch_cohere_embeddings,
    "openai": api_models.fetch_openai_embeddings,
    "voyage": api_models.fetch_voyage_embeddings,
}

#: Opens a model as an Encoder for the length of one cell, freeing it after.
type EncoderFactory = Callable[[str], AbstractContextManager[Encoder]]


def resource_for(slug: str) -> str | None:
    """The serial resource a model's cell holds: an API for hosted models, a GPU for large ones."""
    if slug in _API_SLUGS:
        return "api"
    if slug in _GPU_SLUGS:
        return "gpu"
    return None


def tiers_for(slug: str, source: Source) -> list[tuple[TextTier, str]]:
    """The text tiers a model writes for a source: those its tokenizer keeps and the source has."""
    return [(tier, prose) for tier, prose in variations_for_model(slug) if tier in source.tiers]


def _spec_for(source: Source, slug: str) -> GeneratorSpec | None:
    """One cell per source and model, or none where the model has no tier of the source."""
    model_slug = MODEL_REGISTRY[slug][1]
    tiers = tiers_for(slug, source)
    if not tiers:
        return None
    return GeneratorSpec(
        module=__name__,
        partitions=tuple(partition(model_slug, tier, source.scope) for tier, _ in tiers),
        resource=resource_for(slug),
        variant=f"{source.slug}.{slug}",
        args=(*scope_arguments(source.scope), "--model", slug),
    )


SPECS: tuple[GeneratorSpec, ...] = tuple(
    spec
    for source in SOURCES
    for slug in MODEL_REGISTRY
    if (spec := _spec_for(source, slug)) is not None
)


@contextmanager
def _hosted(slug: str, env: Mapping[str, str]) -> Iterator[Encoder]:
    """A hosted model as an Encoder, reading its key only when a dataset is actually missing."""
    env_var = API_KEY_ENV_VARS[slug]
    api_key = env.get(env_var)
    if not api_key:
        raise RuntimeError(f"{env_var} is not set, cannot fetch {slug} embeddings")
    print(f"fetching from {slug}...", file=sys.stderr)
    yield partial(_FETCHERS[slug], api_key=api_key)


def encoder_for(
    slug: str,
    *,
    device: str | None = None,
    env: Mapping[str, str] | None = None,
    local: Callable[..., AbstractContextManager[Encoder]] = local_models.encoder,
) -> AbstractContextManager[Encoder]:
    """Opens the registered model behind one slug: hosted, large local, or small local."""
    if slug in _API_SLUGS:
        return _hosted(slug, os.environ if env is None else env)
    technical_name = MODEL_REGISTRY[slug][0]
    return local(technical_name, device=device, torch_dtype=_LARGE_DTYPES.get(slug))


def generate(
    units: Sequence[Unit],
    output_root: Path,
    source: Source,
    slugs: Sequence[str],
    *,
    variation: str | None = None,
    encoder: EncoderFactory | None = None,
) -> list[str]:
    """Writes every missing dataset of the source for the named models, returns the names."""
    scope = source.scope
    written: list[str] = []
    for slug in slugs:
        model_slug = MODEL_REGISTRY[slug][1]
        pending = [
            (tier, prose, path)
            for tier, prose in tiers_for(slug, source)
            if (variation is None or tier == variation)
            and (path := path_to_write(output_root, model_slug, tier, scope)) is not None
        ]
        if not pending:
            continue
        with (encoder or encoder_for)(slug) as encode:
            for tier, prose, path in pending:
                write_vectors(path, embed(units, tier, encode), dataset_description(slug, prose))
                written.append(dataset_name(slug, tier))
    return written


def main(
    argv: list[str] | None = None,
    *,
    source_factory: Callable[[Scope], Source] = source_for,
    encoder: EncoderFactory | None = None,
) -> None:
    """Generates one source's missing datasets for every registered model, or for one model."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_output_root_argument(parser)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--witness", default=None)
    parser.add_argument("--reconstruction", default=None)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY), default=None)
    args = parser.parse_args(argv)
    scope = Scope(args.corpus, args.unit, witness=args.witness, reconstruction=args.reconstruction)
    source = source_factory(scope)
    slugs = tuple(MODEL_REGISTRY) if args.model is None else (args.model,)
    written = generate(source.load(), args.output_root, source, slugs, encoder=encoder)
    report_generated(written)


if __name__ == "__main__":
    main()
