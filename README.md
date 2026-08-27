# Tehillim Embeddings

A Parquet dataset of pre-computed feature representations for every
colon of the Hebrew psalms: semantic embedding vectors from 17
different Hebrew and multilingual embedding models (4 Hebrew
BERT-family models, 9 open-weight multilingual encoders, and 4
embedding APIs: Gemini, OpenAI, Cohere, Voyage), lexical
representations built from BHSA's lexical and surface-form features,
and morphology representations built from BHSA's word-level
grammatical features.

## About

Part of [tehillim](https://github.com/rdtaylorjr/tehillim), a
computational analysis of the Hebrew psalms.

Keyed to BHSA's `half_verse` (colon) node ids: every `node_id` in this
dataset is a real Text-Fabric node id from BHSA version `2021`, so the
two are directly joinable without any separate alignment step. This
dataset is not itself a Text-Fabric module, since dense vector data
doesn't fit Text-Fabric's node-feature model well. The
corpus-linguistic data it's keyed to (BHSA) is
real Text-Fabric, loaded via `use()`.

## Semantic representations

* **Dataset files**: 43 Parquet files under `data/`, one per (model, text
  variant) pair, laid out as a Hive-partitioned directory tree:
  `domain=semantic/model=<slug>/text=<variant>/part-0.parquet`. Columns:
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

vectors = pd.read_parquet("data/domain=semantic/model=bge_m3/text=vocalized/part-0.parquet")
vectors = vectors.set_index("node_id")["vector"]
```

Join against BHSA (real Text-Fabric, loaded via `use()`) on `node_id`.

### Citations

> Smiley, David M. "MiqraBERT: Regression-Based Sentence-BERT Finetuning for Biblical Hebrew
> Parallel Detection." arXiv, 2026. https://arxiv.org/abs/2606.19638.

> Seker, Amit, Elron Bandel, Dan Bareket, Idan Brusilovsky, Refael Shaked Greenfeld, and Reut
> Tsarfaty. "AlephBERT: A Hebrew Large Pre-Trained Language Model to Start-Off Your Hebrew NLP
> Application With." arXiv, 2021. https://arxiv.org/abs/2104.04052.

> Shmidman, Shaltiel, Avi Shmidman, and Moshe Koppel. "NeoDictaBERT: Pushing the Frontier of BERT
> Models for Hebrew." arXiv, 2025. https://arxiv.org/abs/2510.20386.

> Shmidman, Avi, Joshua Guedalia, Shaltiel Shmidman, Cheyn Shmuel Shmidman, Eli Handel, and Moshe
> Koppel. "Introducing BEREL: BERT Embeddings for Rabbinic-Encoded Language." arXiv, 2022.
> https://arxiv.org/abs/2208.01875.

> Li, Chaofan, MingHao Qin, Shitao Xiao, Jianlyu Chen, Kun Luo, Yingxia Shao, Defu Lian, and Zheng
> Liu. "Making Text Embedders Few-Shot Learners." arXiv, 2024. https://arxiv.org/abs/2409.15700.

> Zhang, Yanzhao, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, et al.
> "Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models." arXiv, 2025.
> https://arxiv.org/abs/2506.05176.

> Zhao, Xinping, Xinshuo Hu, Zifei Shan, Shouzheng Huang, Yao Zhou, Xin Zhang, Zetian Sun, et al.
> "KaLM-Embedding-V2: Superior Training Techniques and Data Inspire a Versatile Embedding Model."
> arXiv, 2025. https://arxiv.org/abs/2506.20923.

> Babakhin, Yauhen, Radek Osmulski, Ronay Ak, Gabriel Moreira, Mengyao Xu, Benedikt Schifferer, Bo
> Liu, and Even Oldridge. "Llama-Embed-Nemotron-8B: A Universal Text Embedding Model for
> Multilingual and Cross-Lingual Tasks." arXiv, 2025. https://arxiv.org/abs/2511.07025.

> Microsoft. "harrier-oss-v1: Open-Source Multilingual Text Embeddings." Hugging Face, 2026.
> https://huggingface.co/microsoft/harrier-oss-v1-27b.

> Zhang, Ziyin, Zihan Liao, Hang Yu, Peng Di, and Rui Wang. "F2LLM-v2: Inclusive, Performant, and
> Efficient Embeddings for a Multilingual World." arXiv, 2026. https://arxiv.org/abs/2603.19223.

> Chen, Jianlv, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. "M3-Embedding:
> Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge
> Distillation." arXiv, 2024. https://arxiv.org/abs/2402.03216.

> Zhang, Xin, Yanzhao Zhang, Dingkun Long, Wen Xie, Ziqi Dai, Jialong Tang, Huan Lin, et al. "mGTE:
> Generalized Long-Context Text Representation and Reranking Models for Multilingual Text
> Retrieval." arXiv, 2024. https://arxiv.org/abs/2407.19669.

> Wang, Liang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. "Multilingual
> E5 Text Embeddings: A Technical Report." arXiv, 2024. https://arxiv.org/abs/2402.05672.

> Shanbhogue, Madhuri, Zhe Li, Shanfeng Zhang, Gustavo Hernández Ábrego, Shih-Cheng Huang, Aashi
> Jain, Daniel Salz, et al. "Gemini Embedding 2: A Native Multimodal Embedding Model from Gemini."
> arXiv, 2026. https://arxiv.org/abs/2605.27295.

> OpenAI. "New Embedding Models and API Updates." OpenAI, 25 January 2024.
> https://openai.com/index/new-embedding-models-and-api-updates/.

> Cohere. "Embed v4.0." Cohere Documentation, 2025.
> https://docs.cohere.com/changelog/embed-multimodal-v4.

> Voyage AI. "The Voyage 4 Model Family: Shared Embedding Space with MoE Architecture." Voyage AI
> Blog, 2026. https://blog.voyageai.com/2026/01/15/voyage-4/.

> Spärck Jones, Karen. "A Statistical Interpretation of Term Specificity and Its Application in
> Retrieval." *Journal of Documentation* 28.1 (1972): 11-21.

## Lexical representations

`src/lexical` builds representations from BHSA's lexical and surface-form features, independent of
any learned embedding model: exact word choice and repetition, not what a semantic model infers.
Datasets live under `data/domain=lexical/`, in the same `node_id`/`vector` Parquet schema as the
semantic datasets above, so they slot into any script that reads a `tehillim-embeddings`
checkout with no code changes.

* **Identity** (`lexical.vocabulary`, `lexical.surface_vocabulary`): three units, `homograph`
  (BHSA's `lex0` feature, bare consonantal spelling, words that merely look alike are merged into
  one entry), `lexeme` (BHSA's `lex` feature, disambiguated, words that look alike but mean
  different things stay separate), and `word` (the inflected surface form as it occurs in text, no
  lemmatization at all). Benchmarked against parallelism and genre, disambiguation showed no
  measurable advantage over `homograph` under binary presence, so `homograph` is the default
  unit. `lexeme` was initially frozen at binary presence only, then given the full
  weighting/positional family (below) as a follow-up: `homograph`'s whole-Bible ICF weighting
  merges a homograph's frequency across every disambiguated lexeme sharing that spelling, which
  `lexeme` ICF does not, since each `lexeme` has an independent whole-Bible frequency. That is a
  genuine interaction the binary-presence tie never tested. `word` is built at three text tiers,
  `consonantal`, `vocalized`, and `cantillation` (`lexical.surface_corpus`), so its full family is
  three times the size of `homograph`'s or `lexeme`'s.
* **Weighting** (`lexical.vectorize`, `lexical.frequency`, `lexical.surface_vectorize`): five
  colon-level weightings, run over all three units: `binary` (presence), `count` (raw term
  frequency), `log_count` (`log(1+tf)`), `icf` (binary times smoothed inverse corpus frequency),
  and `tf_icf` (`log(1+tf)` times ICF). ICF weight for unit entry ℓ is
  `log((T+1)/(f_ℓ+1)) + 1`. That add-one form is scikit-learn's `TfidfTransformer` smoothing
  convention, substituting whole-Bible token frequency for document frequency. It is not itself the
  formula in Spärck Jones (1972), whose original idf is unsmoothed (`log(N/df)`) and
  document-frequency-based, so it is cited here as an engineering adaptation, not a direct
  implementation of that paper. `T` is the total token count across the whole Bible.
  For `homograph`, `f_ℓ` sums every disambiguated `lexeme`'s whole-Bible frequency sharing that
  spelling (`lexical.frequency.lex0_token_frequencies`). For `lexeme` and `word`, `f_ℓ` is that
  entry's whole-Bible frequency directly, no aggregation
  (`lexical.frequency.lex_token_frequencies`, `lexical.surface_frequency.surface_token_frequencies`,
  the latter computed per text tier). `icf` beat `binary` on both parallelism and genre for
  `homograph`. Raw/log-count weighting did
  not help at either scale.

| dataset | weighting |
|---|---|
| `data/domain=lexical/unit=homograph/construction=binary/` | homograph, binary presence |
| `data/domain=lexical/unit=homograph/construction=count/` | homograph, raw term frequency |
| `data/domain=lexical/unit=homograph/construction=log_count/` | homograph, `log(1+tf)` |
| `data/domain=lexical/unit=homograph/construction=icf/` | homograph, ICF-weighted binary presence |
| `data/domain=lexical/unit=homograph/construction=tf_icf/` | homograph, ICF-weighted `log(1+tf)` |
| `data/domain=lexical/unit=lexeme/construction=binary/` | lexeme, binary presence |
| `data/domain=lexical/unit=lexeme/construction=count/` | lexeme, raw term frequency |
| `data/domain=lexical/unit=lexeme/construction=log_count/` | lexeme, `log(1+tf)` |
| `data/domain=lexical/unit=lexeme/construction=icf/` | lexeme, ICF-weighted binary presence (its frequency) |
| `data/domain=lexical/unit=lexeme/construction=tf_icf/` | lexeme, ICF-weighted `log(1+tf)` (its frequency) |
| `data/domain=lexical/unit=word/text={consonantal,vocalized,cantillation}/construction=binary/` | word, binary presence |
| `data/domain=lexical/unit=word/text={consonantal,vocalized,cantillation}/construction=count/` | word, raw term frequency |
| `data/domain=lexical/unit=word/text={consonantal,vocalized,cantillation}/construction=log_count/` | word, `log(1+tf)` |
| `data/domain=lexical/unit=word/text={consonantal,vocalized,cantillation}/construction=icf/` | word, ICF-weighted binary presence (per-tier frequency) |
| `data/domain=lexical/unit=word/text={consonantal,vocalized,cantillation}/construction=tf_icf/` | word, ICF-weighted `log(1+tf)` (per-tier frequency) |

* **Position and recurrence, colon-level vs. psalm-level** (`lexical.positional`, `lexical.zoning`,
  `lexical.recurrence` for colon-level, `lexical.psalm_position`, `lexical.psalm_zoning`,
  `lexical.psalm_recurrence` for psalm-level, with a `lexical.surface_*` counterpart of each for
  `word`): each colon-level vector is nonzero only in that colon's region of the psalm, so
  two colons in the same psalm get different vectors, the correct construction for parallelism
  (comparing one colon against another). Each psalm-level vector broadcasts one whole-psalm summary
  to every colon in that psalm, the correct construction for genre (which pools colons into a psalm
  centroid by mean, and mean-of-identical-broadcasts differs from mean-of-distinct-per-colon-vectors).
  Both constructions share the same underlying formulas (k-bin ICF-weighted positional pyramid,
  `[binary, mean-position]` zoning, spacing-binned cosine-similarity recurrence profile), only the
  broadcast scope differs. Use the colon-level files for parallelism and the `_psalm`-suffixed
  files for genre. Run over all three units, the `unit=lexeme/` and `unit=word/` table
  rows below are the same 12 constructions, each keyed by that entry's whole-Bible (or, for
  `word`, per-tier) frequency rather than a homograph's merged one. For `word` specifically,
  exact-surface-form recurrence within a short colon-level lag window is rare enough that some
  `construction=icf_spacing{2,4,8}/` datasets (no `_psalm` suffix) leave almost every colon zero, so few
  enough psalms survive mean-pooling to make genre scoring degenerate for those particular
  colon-level datasets. This does not affect the `_psalm`-broadcast recurrence datasets, which pool
  across the whole psalm before scoring.

| dataset | construction |
|---|---|
| `data/domain=lexical/unit=homograph/construction=icf_position{2,4,8}/` | colon-level, `k`-bin positional pyramid |
| `data/domain=lexical/unit=homograph/construction=icf_position_mean/` | colon-level, `[binary, mean-position]` zoning |
| `data/domain=lexical/unit=homograph/construction=icf_spacing{2,4,8}/` | colon-level, spacing-binned recurrence profile |
| `data/domain=lexical/unit=homograph/construction=icf_position{2,4,8}_psalm/` | psalm-broadcast positional pyramid |
| `data/domain=lexical/unit=homograph/construction=icf_position_mean_psalm/` | psalm-broadcast `[binary, mean-position]` zoning |
| `data/domain=lexical/unit=homograph/construction=icf_spacing{2,4,8}_psalm/` | psalm-broadcast recurrence profile |
| `data/domain=lexical/unit=lexeme/construction=icf_position{2,4,8}/` | colon-level, `k`-bin positional pyramid |
| `data/domain=lexical/unit=lexeme/construction=icf_position_mean/` | colon-level, `[binary, mean-position]` zoning |
| `data/domain=lexical/unit=lexeme/construction=icf_spacing{2,4,8}/` | colon-level, spacing-binned recurrence profile |
| `data/domain=lexical/unit=lexeme/construction=icf_position{2,4,8}_psalm/` | psalm-broadcast positional pyramid |
| `data/domain=lexical/unit=lexeme/construction=icf_position_mean_psalm/` | psalm-broadcast `[binary, mean-position]` zoning |
| `data/domain=lexical/unit=lexeme/construction=icf_spacing{2,4,8}_psalm/` | psalm-broadcast recurrence profile |
| `data/domain=lexical/unit=word/text={consonantal,vocalized,cantillation}/construction=icf_position{2,4,8}/` | colon-level, `k`-bin positional pyramid |
| `data/domain=lexical/unit=word/text={consonantal,vocalized,cantillation}/construction=icf_position_mean/` | colon-level, `[binary, mean-position]` zoning |
| `data/domain=lexical/unit=word/text={consonantal,vocalized,cantillation}/construction=icf_spacing{2,4,8}/` | colon-level, spacing-binned recurrence profile |
| `data/domain=lexical/unit=word/text={consonantal,vocalized,cantillation}/construction=icf_position{2,4,8}_psalm/` | psalm-broadcast positional pyramid |
| `data/domain=lexical/unit=word/text={consonantal,vocalized,cantillation}/construction=icf_position_mean_psalm/` | psalm-broadcast `[binary, mean-position]` zoning |
| `data/domain=lexical/unit=word/text={consonantal,vocalized,cantillation}/construction=icf_spacing{2,4,8}_psalm/` | psalm-broadcast recurrence profile |

* **Shuffle-null order control** (`lexical.shuffle_control`,
  `lexical.scripts.generate_shuffle_control`, `lexical.scripts.generate_shuffle_control_colon`):
  a within-psalm colon-order permutation, seeded per `(psalm.number, seed)`, used to test whether a
  positional representation's benchmark score reflects genuine colon-order signal rather than a
  mechanical artifact of the binning itself. `generate_shuffle_control` writes N seeded
  `icf_position_mean_psalm_shuffleNN` psalm-broadcast datasets. `generate_shuffle_control_colon` writes N
  seeded `icf_position4_shuffleNN` colon-level datasets. Scored in `tehillim-benchmarks` via
  `order_shuffle_result` (real score minus mean shuffled score, plus a rank-based permutation
  p-value), not a z-score against the shuffled distribution's mean/std.

Generate with `.venv/bin/python3 -m lexical.generate` (skips any dataset already written).

## Morphology representations

`src/morphology` builds representations from BHSA's word-level grammatical features (part of
speech; gender, number, person, state; verbal stem/tense; pronominal-suffix gender/number/person),
independent of lexical identity or meaning. Datasets live under `data/domain=morphology/`, in the
same `node_id`/`vector` Parquet schema as lexical/semantic, except the `morph_signature` trigram
family, stored sparse (`node_id`/`indices`/`values`, `sparse=true` in the file's schema metadata)
since its dimension (42 + 42² + 42³ = 75,894) is almost entirely zero per colon.

* **POS skeleton** (`unit=sp`, `morphology.pos_ngram`): part-of-speech-only unigram/bigram/trigram
  histograms, over the 14-value closed `SP_VOCABULARY` (no NA/unknown, `sp` always applies).
* **Atomic morphology** (`unit=morph_atomic`, `unit=morph_gn`/`nu`/`ps`/`st`/`vs`/`vt`/`prs_gn`/`prs_nu`/`prs_ps`,
  `unit=morph_full`, `morphology.atomic`): one histogram per feature over its own closed
  vocabulary (`morphology.vocabulary`), `NA` counted as part of the distribution rather than
  excluded, so a feature's applicability rate (e.g. "how many words are verbs") is visible
  alongside its value distribution. `construction=atomic` is that feature alone,
  `construction=sp_plus` is `[sp; feature]`, `unit=morph_atomic`/`construction=core` is
  `[sp; gn; nu; ps; st; vs; vt]` (dim 66), `unit=morph_full`/`construction=all` additionally
  includes the three `prs_*` suffix features (dim 77). A feature is never dropped from a later
  bundle based on how it scored individually.
* **Grammatical signatures** (`unit=morph_signature`, `morphology.signature`,
  `morphology.signature_vectorize`): a per-word signature string concatenating every field that
  word actually carries (`NA` fields omitted, `unknown` kept as its own literal token, field order
  `vs|vt|ps|gn|nu|st` after `sp`), e.g. `verb|qal|perf|p3|m|sg`. Rare signatures (whole-Bible count
  outside Psalms below `MIN_EXTERNAL_SUPPORT_K=1000`, frozen in
  `morphology.signature_support` before any benchmark run, from the curve in
  `config/morph_signature_external_support.csv`) collapse to `<RARE>` before n-grams are
  formed. `construction=inventory` is the unigram histogram, `1_2gram`/`1_2_3gram` add cumulative
  bigram/trigram blocks, the latter stored sparse (above).
* **Pronominal suffix** (`unit=morph_suffix`, `morphology.suffix`): the same signature
  construction restricted to the three `prs_*` fields, ordered `prs_ps|prs_gn|prs_nu`, with a
  dedicated `<NONE>` token when a word carries no suffix at all (all three fields `NA`).
  `construction=inventory` is the suffix histogram alone, `host_plus_suffix` is
  `[morph_signature inventory; suffix inventory]`, `posmean` is the psalm-scale deployment below.
* **Colon-level vs. psalm-broadcast**: every construction above ships both a colon-level file
  (nonzero only in that colon, correct for parallelism) and a `_psalm`-suffixed psalm-broadcast
  file (one whole-psalm vector repeated across its colons, correct for genre), the same convention
  established in the lexical family above.
* **Psalm-scale deployment** (`unit=morph_suffix`/`construction=posmean`, `morphology.deploy`):
  `[b; m]` over the suffix vocabulary, `b` = 1.0 if present anywhere in the psalm, `m` = present
  times `(2 * mean colon position - 1)`, uniform weight (no ICF: a closed-class categorical feature
  doesn't carry rarity signal the way open-class lexical vocabulary does). Suffix was chosen as the
  strongest single atomic-scale signal found across the feature/bundle checkpoints.
* **Shuffle-null order control**: within-colon word-order shuffles
  (`morphology.shuffle_control.shuffled_within_colon_order`) for the POS and signature n-gram
  families, `construction=*_shuffleNN`; a within-psalm colon-order shuffle
  (`lexical.shuffle_control.shuffled_order_by_psalm`, reused directly) for `posmean`,
  `construction=posmean_shuffleNN`. 30 seeds each, scored the same way as the lexical family's
  shuffle control.

Generate with `.venv/bin/python3 -m morphology.generate_pos`,
`.venv/bin/python3 -m morphology.generate_morphology`,
`.venv/bin/python3 -m morphology.generate_signature`,
`.venv/bin/python3 -m morphology.generate_suffix`, and
`.venv/bin/python3 -m morphology.generate_deploy` (each skips any dataset already written).

## Syntax representations

`src/syntax` builds representations from BHSA's phrase- and subphrase-level syntactic
annotations, independent of word-level morphology. Datasets live under `data/domain=syntax/`, all
dense (the largest signature-trigram family is dim 14,424, safely below the sparse threshold used
for morphology's 75,894-dim trigram family).

* **Phrase type and function** (`unit=phrase_typ`/`phrase_function`, `syntax.typ_ngram`/
  `function_ngram`): unigram/bigram/trigram histograms over the closed `TYP_VOCABULARY` (13
  values) and `FUNCTION_VOCABULARY` (29 values), reusing `morphology.ngram`'s vocabulary-agnostic
  histogram/pooling functions directly.
* **Marginal baseline** (`unit=phrase_marginal`, `syntax.marginal`): `[phrase_typ; phrase_function]`
  independent-marginals concatenation, the baseline H5.5 asks whether a joint type-function
  signature beats.
* **Phrase signatures** (`unit=phrase_signature`, `syntax.signature`/`signature_vectorize`): a
  per-atom `typ:function` signature, e.g. `NP:Subj`. Rare signatures (whole-Bible count outside
  Psalms below `MIN_EXTERNAL_SUPPORT_K=1000`, frozen in `syntax.signature_support` before any
  benchmark run, from `config/phrase_signature_external_support.csv`'s 105-value curve)
  collapse to `<RARE>` before n-grams are formed, leaving a 24-value vocabulary.
  `construction=inventory` is the unigram histogram, `1_2gram`/`1_2_3gram` add cumulative
  bigram/trigram blocks.
* **Determination and full signature** (`unit=phrase_det`, `syntax.det_vectorize`; `unit=
  phrase_full_signature`, `syntax.full_signature_vectorize`): `det`'s own independent 3-value
  histogram (`DET_VOCABULARY`), and a `typ:function:det` full-signature inventory (135 distinct
  whole-Bible values, `MIN_EXTERNAL_SUPPORT_K_FULL=1000` collapsing to a 29-value vocabulary,
  frozen from `config/phrase_full_signature_external_support.csv`), tested only conditionally
  against the plain `phrase_signature` inventory.
* **Structural complexity** (`unit=phrase_complexity`, `syntax.complexity`): conventional per-colon
  counts, not new inferential machinery: `[n_atoms, n_phrases, mean_words_per_atom,
  proportion_multi_atom_phrases]`.
* **Phrase-atom relations** (`unit=phrase_rela`, `syntax.rela`/`rela_vectorize`): a unigram
  histogram over `SAFE_RELA_VOCABULARY` (`NA, Appo, Link, Sfxs, Spec`), with `rela=Para`
  (phrase-atom parallelism marker) masked to `NA` before histogramming and permanently excluded
  from the vocabulary, since it would otherwise leak the parallelism benchmark's own target
  variable into a representation scored against it.
* **Subphrase relations** (`unit=phrase_subphrase_rela`, `syntax.subphrase`/
  `subphrase_vectorize`): the same quarantine pattern for subphrase `rela=par` (BHSA's
  subphrase-level parallelism marker, distinct from and far more common than phrase-atom `Para`),
  over `SAFE_SUBPHRASE_RELA_VOCABULARY` (`NA, adj, atr, dem, mod, rec`). 61% of colons have no
  subphrases at all, so this representation's colon-level population is unusually sparse;
  genre-side scoring against it must tolerate a degenerate (e.g. single-psalm) population rather
  than assume every representation clears the usual minimum.
* **Colon-level vs. psalm-broadcast**: every construction above ships both a colon-level file
  (correct for parallelism) and a `_psalm`-suffixed psalm-broadcast file (correct for genre), the
  same convention established in the lexical and morphology families above.
* **Psalm-scale deployment** (`unit=phrase_signature`/`construction=posmean`, `syntax.deploy`):
  `[b; m]` over the (RARE-collapsed) signature vocabulary, reusing morphology's deployment
  formula and `lexical.shuffle_control.shuffled_order_by_psalm` for its shuffle-null.
* **Shuffle-null order control**: within-colon atom-order shuffles
  (`syntax.shuffle_control.shuffled_within_colon_order`) for the type/function/signature n-gram
  families, `construction=*_shuffleNN`, 30 seeds, scored the same way as the lexical and
  morphology families' shuffle controls.

Generate with `.venv/bin/python3 -m syntax.generate_typ`,
`.venv/bin/python3 -m syntax.generate_function`,
`.venv/bin/python3 -m syntax.generate_marginal`,
`.venv/bin/python3 -m syntax.generate_signature`,
`.venv/bin/python3 -m syntax.generate_det`,
`.venv/bin/python3 -m syntax.generate_full_signature`,
`.venv/bin/python3 -m syntax.generate_complexity`,
`.venv/bin/python3 -m syntax.generate_rela`,
`.venv/bin/python3 -m syntax.generate_subphrase`, and
`.venv/bin/python3 -m syntax.generate_deploy` (each skips any dataset already written).

## Family

* [tehillim](https://github.com/rdtaylorjr/tehillim): computational
  analysis of the Hebrew psalms
* [tehillim-benchmarks](https://github.com/rdtaylorjr/tehillim-benchmarks): scores these
  representations against parallelism and genre benchmarks
* [tehillim-data](https://github.com/rdtaylorjr/tehillim-data): hosts the benchmark results
* [tehillim](https://github.com/rdtaylorjr/tehillim): the results page
* [bhsa](https://github.com/etcbc/bhsa): the core text and linguistic
  annotation for the Hebrew Bible

## License

MIT

## Author

* [Russell D. Taylor Jr.](mailto:rdtaylorjr@gatech.edu)
