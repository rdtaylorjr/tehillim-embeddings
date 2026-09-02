"""Symbols vulture cannot see used: the Colab notebook's entry points and test-only oracles."""

from morphology import signature_vectorize
from semantic import large_models
from syntax import function_ngram, typ_ngram
from syntax import signature_vectorize as syntax_signature

#: Called from scripts/compute_large_embeddings.ipynb, which vulture does not scan.
large_models.models_for_choice
large_models.ensure_corpus_data
large_models.gpu_memory_summary

#: Dense references the sparse path is asserted against; production uses the sparse builders.
signature_vectorize.morph_signature_1_2_3gram_vectors
signature_vectorize.morph_signature_1_2_3gram_psalm_vectors
signature_vectorize.morph_signature_1_2_3gram_vectors
typ_ngram.phrase_typ_1_2_3gram_vectors
typ_ngram.phrase_typ_1_2_3gram_psalm_vectors
function_ngram.phrase_function_1_2_3gram_vectors
function_ngram.phrase_function_1_2_3gram_psalm_vectors
syntax_signature.phrase_signature_1_2_3gram_vectors
syntax_signature.phrase_signature_1_2_3gram_psalm_vectors
