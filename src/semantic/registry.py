"""The model and variation registry shared by every generation path."""

from __future__ import annotations

from semantic.api_models import COHERE_MODEL, GEMINI_MODEL, OPENAI_MODEL, VOYAGE_MODEL
from semantic.local_models import (
    ALEPHBERT_MODEL,
    BEREL_MODEL,
    BGE_M3_MODEL,
    BGE_MULTILINGUAL_GEMMA2_MODEL,
    F2LLM_V2_MODEL,
    GTE_MULTILINGUAL_MODEL,
    HARRIER_OSS_MODEL,
    KALM_EMBEDDING_MODEL,
    LLAMA_EMBED_NEMOTRON_MODEL,
    ME5_LARGE_INSTRUCT_MODEL,
    MIQRABERT_MODEL,
    NEODICTABERT_MODEL,
    QWEN3_EMBEDDING_MODEL,
)

#: slug -> (technical model id, dataset-name slug, description).
MODEL_REGISTRY: dict[str, tuple[str, str, str]] = {
    "miqrabert": (
        MIQRABERT_MODEL,
        "miqrabert",
        "MiqraBERT, fine-tuned from AlephBERT for parallel-passage detection.",
    ),
    "alephbert": (
        ALEPHBERT_MODEL,
        "alephbert",
        "AlephBERT sentence-transformers wrapper. Unfinetuned Hebrew BERT baseline.",
    ),
    "neodictabert": (
        NEODICTABERT_MODEL,
        "neodictabert",
        "NeoDictaBERT bilingual embedding model.",
    ),
    "berel": (
        BEREL_MODEL,
        "berel",
        "BEREL, BERT trained on Rabbinic Hebrew.",
    ),
    "bge-multilingual-gemma2": (
        BGE_MULTILINGUAL_GEMMA2_MODEL,
        "bge_multilingual_gemma2",
        "~9B param multilingual encoder, fine-tuned from Gemma2-9B.",
    ),
    "qwen3-embedding": (
        QWEN3_EMBEDDING_MODEL,
        "qwen3_embedding_8b",
        "~8B param multilingual encoder.",
    ),
    "kalm-embedding": (
        KALM_EMBEDDING_MODEL,
        "kalm_embedding_gemma3_12b_2511",
        "~12B param multilingual encoder, fine-tuned from Gemma3-12B.",
    ),
    "llama-embed-nemotron": (
        LLAMA_EMBED_NEMOTRON_MODEL,
        "llama_embed_nemotron_8b",
        (
            "~8B param multilingual encoder, fine-tuned from Llama-3.1-8B "
            "with bidirectional attention. Non-commercial/research-use license."
        ),
    ),
    "harrier-oss-v1": (
        HARRIER_OSS_MODEL,
        "harrier_oss_v1_27b",
        (
            "~27B param multilingual encoder, decoder-only with last-token "
            "pooling and L2 normalization."
        ),
    ),
    "f2llm-v2": (
        F2LLM_V2_MODEL,
        "f2llm_v2_14b",
        "~14B param multilingual encoder, fine-tuned from Qwen3-14B.",
    ),
    "bge-m3": (
        BGE_M3_MODEL,
        "bge_m3",
        "~568M param multilingual encoder, CLS-token pooled.",
    ),
    "gte-multilingual-base": (
        GTE_MULTILINGUAL_MODEL,
        "gte_multilingual_base",
        "~305M param multilingual encoder, CLS-token pooled.",
    ),
    "me5-large-instruct": (
        ME5_LARGE_INSTRUCT_MODEL,
        "me5_large_instruct",
        "~560M param multilingual encoder.",
    ),
    "gemini": (
        GEMINI_MODEL,
        "gemini_embedding_2",
        "Gemini Embedding 2, accessed via OpenRouter.",
    ),
    "openai": (
        OPENAI_MODEL,
        "openai_text_embedding_3_large",
        "OpenAI text-embedding-3-large, accessed via OpenRouter.",
    ),
    "cohere": (
        COHERE_MODEL,
        "cohere_embed_v4",
        "Cohere Embed v4, accessed via Cohere's API.",
    ),
    "voyage": (
        VOYAGE_MODEL,
        "voyage_4",
        "Voyage 4, accessed via OpenRouter. Distinct from voyage-4-large/voyage-4-lite.",
    ),
}

#: These tokenizers strip niqqud and cantillation, so only the consonantal variation is generated.
TOKENIZER_STRIPS_ALL_DIACRITICS = {"miqrabert", "alephbert", "neodictabert", "berel"}

#: (variation, vocalized, niqqud_only, description), matching `select_half_verses`'s parameters.
VARIATIONS: list[tuple[str, bool, bool, str]] = [
    ("consonantal", False, False, "Bare consonants. No niqqud, no cantillation."),
    ("vocalized", True, True, "Niqqud (vowel points) only. No cantillation marks."),
    (
        "cantillation",
        True,
        False,
        "Niqqud and cantillation/accent marks together (full Masoretic pointing).",
    ),
]


def variations_for_model(slug: str) -> list[tuple[str, bool, bool, str]]:
    """Returns the text variations to generate for a model slug."""
    if slug in TOKENIZER_STRIPS_ALL_DIACRITICS:
        return [
            (
                "consonantal",
                False,
                False,
                (
                    "Bare consonants. No niqqud, no cantillation. This model's "
                    "tokenizer strips niqqud and cantillation identically, so the "
                    "vocalized and cantillation variations would carry no "
                    "additional signal."
                ),
            )
        ]
    return VARIATIONS


def dataset_name(slug: str, variation: str) -> str:
    """Returns the Parquet dataset name for a model slug and variation."""
    _, name_slug, _ = MODEL_REGISTRY[slug]
    return f"semantic_{name_slug}_{variation}"


def dataset_description(slug: str, variation_description: str) -> str:
    """Returns the descriptive text stored in the Parquet file's metadata."""
    _, _, model_description = MODEL_REGISTRY[slug]
    return (
        f"{model_description} {variation_description} "
        "Columns: node_id (int32, BHSA half-verse node id), vector (float32 list)."
    )
