"""Writes one embedding matrix as a Text-Fabric feature file."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from tf.fabric import Fabric

from semantic.registry import encode_vector

if TYPE_CHECKING:
    from semantic.corpus import Psalm

MODULE_VERSION = "1.0"


def feature_path(output_root: Path, name: str) -> Path:
    """Returns the `.tf` file path for a feature name."""
    return output_root / "tf" / MODULE_VERSION / f"{name}.tf"


def node_values(
    embeddings: dict[int, np.ndarray], psalms: list[Psalm]
) -> dict[int, str]:
    """Maps each psalm's embedding vectors to its BHSA half-verse node ids."""
    values: dict[int, str] = {}
    for psalm in psalms:
        vectors = embeddings.get(psalm.number)
        if vectors is None:
            continue
        for node, vector in zip(psalm.half_verse_nodes, vectors, strict=True):
            values[node] = encode_vector(vector)
    return values


def write_feature(output_root: Path, name: str, values: dict[int, str], description: str) -> None:
    """Writes one Text-Fabric feature file."""
    location = output_root / "tf" / MODULE_VERSION
    TF = Fabric(locations=[], modules=[], silent="deep")
    TF.save(
        nodeFeatures={name: values},
        metaData={
            "": {
                "project": "tehillim-embeddings",
                "source": "https://github.com/rdtaylorjr/tehillim",
                "version": MODULE_VERSION,
            },
            name: {"valueType": "str", "description": description},
        },
        location=str(location),
        silent="deep",
    )
