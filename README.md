# Tehillim Representations

A Parquet dataset of pre-computed feature representations for every
half-verse of the Hebrew psalms: semantic embedding vectors from 17
different Hebrew and multilingual embedding models (4 Hebrew
BERT-family models, 9 open-weight multilingual encoders, and 4
embedding APIs — Gemini, OpenAI, Cohere, Voyage), plus lexical
representations built from BHSA's lexeme features.

## About

Part of [tehillim](https://github.com/rdtaylorjr/tehillim), a
computational analysis of the Hebrew psalms.

Keyed to BHSA's `half_verse` (colon) node ids: every `node_id` in this
dataset is a real Text-Fabric node id from BHSA version `2021`, so the
two are directly joinable without any separate alignment step. This
dataset is not itself a Text-Fabric module, since dense vector data
doesn't fit Text-Fabric's node-feature model well. The
corpus-linguistic data it's keyed to (BHSA, `tehillim-parallelism`) is
real Text-Fabric, loaded via `use()`.

## Semantic representations

* **Dataset files**: 43 Parquet files under `data/`, one per (model, text
  variant) pair, laid out as a Hive-partitioned directory tree:
  `type=semantic/model=<slug>/text=<variant>/part-0.parquet`. Columns:
  `node_id` (int32, a BHSA `half_verse` node id) and `vector` (float32
  list). Every dataset is scoped to the book of Psalms: rows exist for its
  5,203 `half_verse` nodes, not for the rest of BHSA's `half_verse` node
  space. Vector dimension varies by model (see the table below), so the
  directory can't be read as a single table across all models at once.
  Read one `model=`/`text=` file at a time, or use
  `pyarrow.dataset.dataset(..., partitioning="hive")` scoped to a single
  model.
* **Generation code**: `src/semantic/`, loads psalm text from BHSA via
  Text-Fabric and writes model output as Parquet, with no intermediate
  cache. Run with `.venv/bin/python3 -m semantic.generate` for the local
  and API models. The six Colab-only models run from
  `scripts/compute_large_embeddings.ipynb`. Each model checks for its
  already-written dataset file first, so both are safe to re-run.

### Models

| Model | Technical identifier | Dataset slug | Dimensions |
| --- | --- | --- | --- |
| MiqraBERT | `davidmsmiley/MiqraBERT` | `miqrabert` | 768 |
| AlephBERT | `imvladikon/sentence-transformers-alephbert` | `alephbert` | 768 |
| NeoDictaBERT | `dicta-il/neodictabert-bilingual-embed` | `neodictabert` | 768 |
| BEREL | `dicta-il/BEREL` | `berel` | 768 |
| bge-multilingual-gemma2 | `BAAI/bge-multilingual-gemma2` | `bge_multilingual_gemma2` | 3584 |
| Qwen3-Embedding-8B | `Qwen/Qwen3-Embedding-8B` | `qwen3_embedding_8b` | 4096 |
| KaLM-Embedding-Gemma3-12B | `tencent/KaLM-Embedding-Gemma3-12B-2511` | `kalm_embedding_gemma3_12b_2511` | 3840 |
| Llama-Embed-Nemotron-8B | `nvidia/llama-embed-nemotron-8b` | `llama_embed_nemotron_8b` | 4096 |
| Harrier-OSS-v1-27B | `microsoft/harrier-oss-v1-27b` | `harrier_oss_v1_27b` | 5376 |
| F2LLM-v2-14B | `codefuse-ai/F2LLM-v2-14B` | `f2llm_v2_14b` | 5120 |
| BGE-M3 | `BAAI/bge-m3` | `bge_m3` | 1024 |
| GTE-multilingual-base | `Alibaba-NLP/gte-multilingual-base` | `gte_multilingual_base` | 768 |
| mE5-large-instruct | `intfloat/multilingual-e5-large-instruct` | `me5_large_instruct` | 1024 |
| Gemini Embedding 2 | `google/gemini-embedding-2` | `gemini_embedding_2` | 3072 |
| OpenAI text-embedding-3-large | `openai/text-embedding-3-large` | `openai_text_embedding_3_large` | 3072 |
| Cohere Embed v4 | `embed-v4.0` | `cohere_embed_v4` | 1536 |
| Voyage 4 | `voyageai/voyage-4` | `voyage_4` | 1024 |

### Variations

* `consonantal`: bare consonants, no niqqud, no cantillation
* `vocalized`: niqqud (vowel points) only, no cantillation marks
* `cantillation`: niqqud and cantillation/accent marks together

### Usage

```python
import pandas as pd

vectors = pd.read_parquet("data/type=semantic/model=bge_m3/text=vocalized/part-0.parquet")
vectors = vectors.set_index("node_id")["vector"]
```

Join against BHSA or `tehillim-parallelism` (both real Text-Fabric,
loaded via `use()`) on `node_id`.

### Citations

* Smiley, D. M. (2026). MiqraBERT: Regression-Based Sentence-BERT
  Finetuning for Biblical Hebrew Parallel Detection. arXiv:2606.19638.
* Seker, A., Bandel, E., Bareket, D., Brusilovsky, I., Greenfeld, R. S.,
  & Tsarfaty, R. (2021). AlephBERT: A Hebrew Large Pre-Trained Language
  Model to Start-off your Hebrew NLP Application With. arXiv:2104.04052.
* Shmidman, S., Shmidman, A., & Koppel, M. (2025). NeoDictaBERT:
  Pushing the Frontier of BERT models for Hebrew. arXiv:2510.20386.
* Shmidman, A., Guedalia, J., Shmidman, S., Shmidman, C. S., Handel,
  E., & Koppel, M. (2022). Introducing BEREL: BERT Embeddings for
  Rabbinic-Encoded Language. arXiv:2208.01875.
* Li, C. et al. (2024). Making Text Embedders Few-Shot Learners.
  arXiv:2409.15700.
* Zhang, Y. et al. (2025). Qwen3 Embedding: Advancing Text Embedding
  and Reranking Through Foundation Models. arXiv:2506.05176.
* Zhao, X. et al. (2025). KaLM-Embedding-V2: Superior Training
  Techniques and Data Inspire A Versatile Embedding Model.
  arXiv:2506.20923.
* Babakhin, Y. et al. (2025). Llama-Embed-Nemotron-8B: A Universal
  Text Embedding Model for Multilingual and Cross-Lingual Tasks.
  arXiv:2511.07025.
* Microsoft (2026). harrier-oss-v1: Open-Source Multilingual Text
  Embeddings. https://huggingface.co/microsoft/harrier-oss-v1-27b
* Zhang, Z., Liao, Z., Yu, H., Di, P., & Wang, R. (2026). F2LLM-v2:
  Inclusive, Performant, and Efficient Embeddings for a Multilingual
  World. arXiv:2603.19223.
* Chen, J. et al. (2024). BGE M3-Embedding: Multi-Lingual,
  Multi-Functionality, Multi-Granularity Text Embeddings Through
  Self-Knowledge Distillation. arXiv:2402.03216.
* Zhang, X. et al. (2024). mGTE: Generalized Long-Context Text
  Representation and Reranking Models for Multilingual Text Retrieval.
  arXiv:2407.19669.
* Wang, L. et al. (2024). Multilingual E5 Text Embeddings: A Technical
  Report. arXiv:2402.05672.
* Shanbhogue, M. et al. (2026). Gemini Embedding 2: A Native
  Multimodal Embedding Model from Gemini. arXiv:2605.27295.
* OpenAI (2024). New embedding models and API updates.
  https://openai.com/index/new-embedding-models-and-api-updates/
* Cohere (2025). Embed v4.0.
  https://docs.cohere.com/changelog/embed-multimodal-v4
* Voyage AI (2026). The Voyage 4 model family: shared embedding space
  with MoE architecture. https://blog.voyageai.com/2026/01/15/voyage-4/
* Spärck Jones, K. (1972). A Statistical Interpretation of Term
  Specificity and Its Application in Retrieval. Journal of
  Documentation, 28(1), 11-21.

## Lexical representations

`src/lexical` builds representations from BHSA's `lex`/`lex0` features, independent of any
learned embedding model: exact word choice and repetition, not what a semantic model infers.
Datasets live under `data/type=lexical/`, in the same `node_id`/`vector` Parquet schema as the
semantic datasets above, so they slot into any script that reads a `tehillim-representations`
checkout with no code changes.

* **Identity** (`lexical.vocabulary`): two vocabularies were compared, `lex0` (BHSA's bare
  consonantal lexeme, homonyms collapsed) and `lex` (BHSA's disambiguated lexeme, homonyms kept
  separate). Benchmarked against parallelism and genre, disambiguation showed no measurable
  advantage under binary presence, so `lex0` is the default vocabulary. `lexeme_binary` (the `lex`
  variant) is kept as a frozen control.
* **Weighting** (`lexical.vectorize`, `lexical.frequency`): five colon-level weightings over the
  `lex0` vocabulary were compared: `binary` (presence), `count` (raw term frequency), `log_count`
  (`log(1+tf)`), `icf` (binary times smoothed inverse corpus frequency), and `tf_icf`
  (`log(1+tf)` times ICF). ICF weight for lexeme ℓ is `log((T+1)/(f_ℓ+1)) + 1`, a smoothed
  corpus-frequency weighting (Spärck Jones 1972) using whole-Hebrew-Bible token counts rather than
  document frequency: `T` is the total token count across the whole Bible and `f_ℓ` is `lex0` ℓ's
  whole-Bible token frequency, summed across every `lex` homonym sharing it. `icf` beat `binary` on
  both parallelism and genre. Raw/log-count weighting did not help at either scale.

| dataset | weighting |
|---|---|
| `data/type=lexical/vocab=form/weight=binary/` | `lex0`, binary presence (frozen) |
| `data/type=lexical/vocab=form/weight=count/` | `lex0`, raw term frequency |
| `data/type=lexical/vocab=form/weight=log_count/` | `lex0`, `log(1+tf)` |
| `data/type=lexical/vocab=form/weight=icf/` | `lex0`, ICF-weighted binary presence |
| `data/type=lexical/vocab=form/weight=tf_icf/` | `lex0`, ICF-weighted `log(1+tf)` |
| `data/type=lexical/vocab=lexeme/weight=binary/` | `lex`, binary presence (frozen control) |

Generate with `.venv/bin/python3 -m lexical.generate` (skips any dataset already written).

## Family

* [tehillim](https://github.com/rdtaylorjr/tehillim): computational
  analysis of the Hebrew psalms
* [bhsa](https://github.com/etcbc/bhsa): the core text and linguistic
  annotation for the Hebrew Bible

## License

MIT

## Author

* [Russell D. Taylor Jr.](mailto:rdtaylorjr@gatech.edu)
