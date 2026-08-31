"""Generates N colon-order-shuffled icf_position4 datasets, a shuffle-null control (parallelism)."""

from __future__ import annotations

import sys
from pathlib import Path

from lexical.corpus import Corpus, LexicalPsalm
from lexical.export import write_dataset
from lexical.frequency import icf_weights as compute_icf_weights
from lexical.frequency import lex0_token_frequencies, total_token_count
from lexical.positional import positional_icf_vectors
from lexical.shuffle_control import DEFAULT_N_SHUFFLES, shuffled_order_by_psalm
from lexical.vocabulary import build_vocabulary

_K = 4


def generate_shuffle_control_colon(
    psalms: list[LexicalPsalm], output_root: Path, icf_weights: dict[str, float], n_shuffles: int
) -> list[str]:
    """Writes n_shuffles seeded, colon-order-shuffled icf_position4 datasets, returns names."""
    vocabulary = build_vocabulary(psalms, key="lex0")
    written: list[str] = []
    for seed in range(1, n_shuffles + 1):
        order = shuffled_order_by_psalm(psalms, seed)
        vectors = positional_icf_vectors(
            psalms, vocabulary, "lex0", icf_weights, k=_K, order_by_psalm=order
        )
        weight = f"icf_position4_shuffle{seed:02d}"
        description = f"Shuffle-null order-effect control for icf_position4, seed {seed}."
        write_dataset(output_root, "homograph", weight, vectors, description, unit_key="unit")
        written.append(weight)
    return written


def main() -> None:
    """Generates the shuffle-null control datasets for icf_position4."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES)
    args = parser.parse_args()

    corpus = Corpus.load()
    psalms = corpus.psalms()
    lex0_frequencies = lex0_token_frequencies(corpus.api)
    total_tokens = total_token_count(corpus.api)
    icf_lookup = compute_icf_weights(lex0_frequencies, total_tokens)
    written = generate_shuffle_control_colon(psalms, args.output_root, icf_lookup, args.n_shuffles)
    print(f"wrote {len(written)} shuffle-control datasets", file=sys.stderr)


if __name__ == "__main__":
    main()
