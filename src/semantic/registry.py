"""The model and variation registry shared by every generation path."""

from __future__ import annotations

#: slug -> (technical model id, dataset-name slug, description).
MODEL_REGISTRY: dict[str, tuple[str, str, str]] = {
    "miqrabert": (
        "davidmsmiley/MiqraBERT",
        "miqrabert",
        "MiqraBERT, fine-tuned from AlephBERT for parallel-passage detection.",
    ),
    "alephbert": (
        "imvladikon/sentence-transformers-alephbert",
        "alephbert",
        "AlephBERT sentence-transformers wrapper. Unfinetuned Hebrew BERT baseline.",
    ),
    "neodictabert": (
        "dicta-il/neodictabert-bilingual-embed",
        "neodictabert",
        "NeoDictaBERT bilingual embedding model.",
    ),
    "berel": (
        "dicta-il/BEREL",
        "berel",
        "BEREL, BERT trained on Rabbinic Hebrew.",
    ),
    "bge-multilingual-gemma2": (
        "BAAI/bge-multilingual-gemma2",
        "bge_multilingual_gemma2",
        "~9B param multilingual encoder, fine-tuned from Gemma2-9B.",
    ),
    "qwen3-embedding": (
        "Qwen/Qwen3-Embedding-8B",
        "qwen3_embedding_8b",
        "~8B param multilingual encoder.",
    ),
    "kalm-embedding": (
        "tencent/KaLM-Embedding-Gemma3-12B-2511",
        "kalm_embedding_gemma3_12b_2511",
        "~12B param multilingual encoder, fine-tuned from Gemma3-12B.",
    ),
    "llama-embed-nemotron": (
        "nvidia/llama-embed-nemotron-8b",
        "llama_embed_nemotron_8b",
        "~8B param multilingual encoder, fine-tuned from Llama-3.1-8B "
        "with bidirectional attention. Non-commercial/research-use license.",
    ),
    "harrier-oss-v1": (
        "microsoft/harrier-oss-v1-27b",
        "harrier_oss_v1_27b",
        "~27B param multilingual encoder, decoder-only with last-token "
        "pooling and L2 normalization.",
    ),
    "f2llm-v2": (
        "codefuse-ai/F2LLM-v2-14B",
        "f2llm_v2_14b",
        "~14B param multilingual encoder, fine-tuned from Qwen3-14B.",
    ),
    "bge-m3": (
        "BAAI/bge-m3",
        "bge_m3",
        "~568M param multilingual encoder, CLS-token pooled.",
    ),
    "gte-multilingual-base": (
        "Alibaba-NLP/gte-multilingual-base",
        "gte_multilingual_base",
        "~305M param multilingual encoder, CLS-token pooled.",
    ),
    "me5-large-instruct": (
        "intfloat/multilingual-e5-large-instruct",
        "me5_large_instruct",
        "~560M param multilingual encoder.",
    ),
    "gemini": (
        "google/gemini-embedding-2",
        "gemini_embedding_2",
        "Gemini Embedding 2, accessed via OpenRouter.",
    ),
    "openai": (
        "openai/text-embedding-3-large",
        "openai_text_embedding_3_large",
        "OpenAI text-embedding-3-large, accessed via OpenRouter.",
    ),
    "cohere": (
        "embed-v4.0",
        "cohere_embed_v4",
        "Cohere Embed v4, accessed via Cohere's API.",
    ),
    "voyage": (
        "voyageai/voyage-4",
        "voyage_4",
        "Voyage 4, accessed via OpenRouter. Distinct from voyage-4-large/voyage-4-lite.",
    ),
}

#: Models whose tokenizer strips niqqud and cantillation before the
#: model sees either. Only the consonantal variation is generated for these.
TOKENIZER_STRIPS_ALL_DIACRITICS = {"miqrabert", "alephbert", "neodictabert", "berel"}

#: (variation name, vocalized, niqqud_only, description). vocalized/
#: niqqud_only match `_select_half_verses`'s parameters.
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
                "Bare consonants. No niqqud, no cantillation. This model's "
                "tokenizer strips niqqud and cantillation identically, so the "
                "vocalized and cantillation variations would carry no "
                "additional signal.",
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
