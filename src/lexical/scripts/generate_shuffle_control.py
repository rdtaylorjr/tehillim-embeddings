"""Generates N colon-order-shuffled icf_posmean_psalm datasets, a shuffle-null order control."""

from __future__ import annotations

import sys
from pathlib import Path

from lexical.corpus import Corpus, LexicalPsalm
from lexical.export import write_dataset
from lexical.frequency import icf_weights as compute_icf_weights
from lexical.frequency import lex0_token_frequencies, total_token_count
from lexical.psalm_zoning import psalm_positional_centroid_vectors
from lexical.shuffle_control import shuffled_order_by_psalm
from lexical.vocabulary import build_vocabulary


def generate_shuffle_control(
    psalms: list[LexicalPsalm], output_root: Path, icf_weights: dict[str, float], n_shuffles: int
) -> list[str]:
    """Writes n_shuffles seeded, colon-order-shuffled icf_posmean_psalm datasets, returns names."""
    vocabulary = build_vocabulary(psalms, key="lex0")
    written: list[str] = []
    for seed in range(1, n_shuffles + 1):
        order = shuffled_order_by_psalm(psalms, seed)
        vectors = psalm_positional_centroid_vectors(
            psalms, vocabulary, "lex0", icf_weights, order_by_psalm=order
        )
        weight = f"icf_posmean_psalm_shuffle{seed:02d}"
        description = f"Shuffle-null order-effect control for icf_posmean_psalm, seed {seed}."
        write_dataset(output_root, "lex0", weight, vectors, description)
        written.append(weight)
    return written


def main() -> None:
    """Generates the shuffle-null control datasets for icf_posmean_psalm."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--n-shuffles", type=int, default=30)
    args = parser.parse_args()

    corpus = Corpus.load()
    psalms = corpus.psalms()
    lex0_frequencies = lex0_token_frequencies(corpus.api)
    total_tokens = total_token_count(corpus.api)
    icf_lookup = compute_icf_weights(lex0_frequencies, total_tokens)
    written = generate_shuffle_control(psalms, args.output_root, icf_lookup, args.n_shuffles)
    print(f"wrote {len(written)} shuffle-control datasets", file=sys.stderr)


if __name__ == "__main__":
    main()
