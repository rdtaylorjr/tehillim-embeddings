# tehillim-embeddings

## Overview

This repository generates Parquet feature representations of the 150 Psalms from the
[ETCBC Biblia Hebraica Stuttgartensia Amstelodamensis](https://github.com/ETCBC/bhsa) (BHSA).
It supplies a common representation layer for `tehillim-benchmarks`: lexical, morphological,
syntactic, and semantic vectors keyed to the same BHSA `half_verse` nodes.

## Data

`data/` contains 24,226 committed Hive-partitioned Parquet files, totaling 3.6 GB. Every dense
file has `node_id` (`int32`) and `vector` (`float32` fixed-size list) columns. Sparse trigram
files use `node_id`, `indices`, and `values`. Schema metadata records the construction and dataset
format version. Each semantic export has 5,203 rows, one for each `half_verse` node in the Hebrew Psalms.

`node_id` is a BHSA node identifier, so exported vectors join to the source corpus without an
alignment layer. BHSA `half_verse` is an accentual section of the Masoretic text, marked `A`, `B`,
or `C`. It provides a stable comparison target. It does not settle the relation between accentual,
syntactic, prosodic, and literary segmentation.

The repository reads BHSA features through [Text-Fabric](https://github.com/annotation/text-fabric)
at checkout `v1.8.1`. Its lexical, morphological, and syntactic features preserve ETCBC analytical
decisions. The files are data derived from that analysis, rather than theory-free descriptions of
Hebrew. `config/` contains whole-Bible-outside-Psalms support counts used for signature vocabularies.
Model identifiers and text-state availability are registered in
[`src/semantic/registry.py`](src/semantic/registry.py). The registry records the technical
identifier used for each model family. It does not record an immutable checkpoint revision for every
external model.

| Domain | Files | Registered material |
| --- | ---: | --- |
| Lexical | 2,095 | Homographs, disambiguated lexemes, and surface word forms |
| Morphology | 9,057 | Word-level grammatical features and signatures |
| Syntax | 13,031 | Phrase-atom, phrase, and subphrase annotations |
| Semantic | 43 | Half-verse embeddings from 17 Hebrew and multilingual models |

## Methodology

Each family registers a different representation of the same `half_verse` sequence. Lexical
features distinguish BHSA `lex0` homographs, `lex` values, and surface forms in consonantal,
vocalized, and fully pointed text states. Morphology records part of speech, gender, number, person,
state, verbal stem and tense, and pronominal-suffix features. `NA` remains a vocabulary value, so a
feature's applicability contributes to the representation. Syntax records phrase-atom `typ`, mother
phrase `function`, `det`, relations, subphrase relations, phrase complexity, and joint
`typ:function` signatures. Semantic models encode the corresponding Hebrew string in
`consonantal`, `vocalized`, and `cantillation` text states where their tokenizers preserve the distinction.

Feature inventories are normalized within a `half_verse`. N-gram constructions concatenate
unigram, bigram, and trigram proportions, each normalized by its available positions. Psalm-scale
variants pool raw counts across a psalm, normalize each n-gram order once, then broadcast the psalm
vector to its constituent nodes. This distinguishes a representation of local correspondence from a
representation of psalm-scale distribution.

Lexical ICF uses whole-Bible token frequency: \(\log((T + 1)/(f + 1)) + 1\). It increases the weight
of infrequent vocabulary entries while retaining a finite value for every observed entry. This is an
engineering adaptation of term-specificity weighting. It uses token frequency rather than document
frequency and is a representation condition rather than a standard information-retrieval statistic.
The generator retains binary, count, log-count, ICF, and TF-ICF variants for downstream comparison.

Lexical placement vectors assign ICF-weighted vocabulary presence to equal-width bins of normalized
psalm position. Recurrence vectors average ICF-weighted cosine similarity over normalized lag bins.
Syntactic complexity records phrase-atom count, phrase count, mean words per atom, and the share of
multi-atom phrases. `phrase_marginal` keeps phrase type and function separate, providing a baseline
for the joint `typ:function` signature.

The repository preserves several comparison conditions in the generated data.

- Joint signatures sit alongside atomic inventories. `phrase_marginal` provides a separate
  type-and-function baseline for the phrase signature. Signatures with fewer than 1,000 occurrences
  outside Psalms collapse to `<RARE>`. The threshold was fixed from external support counts before
  benchmark scoring.
- `rela=Para` at phrase-atom level and `rela=par` at subphrase level are masked to `NA` before
  syntax vectorization. These values would disclose the parallelism target evaluated elsewhere.
- Order-sensitive lexical placement and recurrence features receive within-psalm half-verse-order
  shuffles. Morphological and syntactic n-grams receive within-half-verse word or phrase-atom-order
  shuffles. Each permutation is deterministic from its seed and node or psalm identifier. The
  default set contains 1,000 seeds and is generated across processes.

The computations derive distributions, sequences, and controlled alternatives. They do not assign a
literary function or decide an interpretation of a psalm.

## Results

The committed result of this repository is the representation corpus described above. The 43
semantic datasets cover the 17 registered models and the applicable text states. The remaining
24,183 files include linguistic representations and their controlled permutations.

This repository does not calculate retrieval scores, clustering outcomes, significance tests, or
claims about Hebrew poetic categories. [`tehillim-benchmarks`](https://github.com/rdtaylorjr/tehillim-benchmarks)
performs those comparisons. [`tehillim-data`](https://github.com/rdtaylorjr/tehillim-data) publishes
the resulting measurements. Keeping representation generation separate from evaluation makes the
construction, control, and scoring stages inspectable on their terms.

## Limitations

The `half_verse` key gives every family a common target, while also imposing an accentual division
on the data. A vector associated with that node can register words, phrase atoms, or a model's text
encoding across the division. It cannot demonstrate that the division is a syntactic or poetic unit.

BHSA linguistic features implement a particular grammar. A result for `typ`, `function`, or a
derived signature concerns that annotation system. Semantic vectors add a separate set of model and
tokenizer assumptions. Model identifiers are recorded, though remote checkpoints, APIs, and
hardware-dependent execution can change or remain unavailable.

External support thresholding reduces the influence of rare signatures without establishing a
linguistically privileged cutoff. Shuffled data test sensitivity to the disrupted ordering. They do
not validate a grammatical analysis. Short units can also yield sparse or zero vectors, especially
for higher-order sequences and exact surface-form recurrence. Consumers must inspect coverage and
zero-vector behavior for each construction.

The committed vectors preserve derived outputs, while construction decisions remain distributed across generators and support files. A future revision should attach code revision, corpus revision, configuration values, and support-count inputs to each output partition. It should also regenerate a defined subset under clause and phrase-atom units, so the effect of the accentual target unit becomes a measured sensitivity condition.

No top-level command reconstructs the full linguistic corpus. The semantic entry point also processes the 17 registered models sequentially. Model jobs are independent and expensive, while hardware and provider rate limits require an explicit scheduling policy. A future orchestrator should record its execution plan and run compatible jobs across controlled workers without changing the representation formula.

## Reproducibility

The corpus loader requests BHSA Text-Fabric checkout `v1.8.1` from `TEHILLIM_BHSA_PATH` or a local
default path, then falls back to Text-Fabric's corpus loader. The source code requires Python 3.12 or
later. Dense and sparse exports use Parquet format version metadata `1.0` and Zstandard compression.

The committed linguistic files can be regenerated from BHSA and the support-count CSVs. Semantic
regeneration additionally requires the named model checkpoints or provider credentials. The project
records minimum package versions rather than a fully locked environment. Remote-model revisions and
provider outputs limit byte-level replication of a fresh semantic run. A technical model identifier
can identify a model family without identifying a frozen artifact.

Generation is distributed across domain modules. `python -m lexical.generate` builds lexical
homograph and lexeme files. Morphological and syntactic generators require separate module and
configuration invocations. The repository has no manifest that enumerates every committed partition
with its generating command and input fingerprint.

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
export TEHILLIM_BHSA_PATH=/path/to/bhsa/tf/2021
```

## Usage

Read a committed representation by its partition path:

```python
import pyarrow.parquet as pq

table = pq.read_table("data/domain=lexical/unit=homograph/construction=icf/part-0.parquet")
vectors = dict(zip(table["node_id"].to_pylist(), table["vector"].to_pylist(), strict=True))
```

Regenerate a linguistic family into a separate directory, or run the full verification suite:

```bash
.venv/bin/python -m lexical.generate --output-root /path/to/output
./check.sh
```

The lexical command does not rebuild morphology or syntax. Their generators and external-support
CSV inputs are in `src/morphology`, `src/syntax`, and `config`. A full corpus rebuild requires an
explicit invocation plan for those modules.

Semantic generation uses `python -m semantic.generate --output-root /path/to/output`. It downloads
local model checkpoints or requires the provider environment variables documented in
[`src/semantic/api_models.py`](src/semantic/api_models.py).

## References

Eep Talstra Centre for Bible and Computer. *Biblia Hebraica Stuttgartensia Amstelodamensis*.
Persistent identifier: https://doi.org/10.17026/dans-z6y-skyh.

Andersen, Francis I., and A. Dean Forbes. [“The Andersen-Forbes Computational Analysis of Biblical Hebrew Grammar.”](https://doi.org/10.25159/1013-8471/2936) *Journal for Semitics* 27, no. 1 (2018).

Berman, Joshua. [“Measuring Style in Isaiah: Isaiah 34-35 and the Tiberias Stylistic Classifier for the Hebrew Bible.”](https://doi.org/10.1163/15685330-12341070) *Vetus Testamentum* 71, no. 3 (2021): 303-316.

Roorda, Dirk. 2019. “Text-Fabric: Handling Biblical Data with IKEA Logistics.” *HIPHIL Novum* 5
(2): 126-135. https://doi.org/10.7146/hn.v5i2.142740.

Seker, Amit, Elron Bandel, Dan Bareket, Idan Brusilovsky, Refael Greenfeld, and Reut Tsarfaty. 2022. [“AlephBERT: Language Model Pre-training and Evaluation from Sub-Word to Sentence Level.”](https://doi.org/10.18653/v1/2022.acl-long.4) In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics*, 46-56.

Shmidman, Avi, Joshua Guedalia, Shaltiel Shmidman, Cheyn Shmuel Shmidman, Eli Handel, and Moshe Koppel. 2022. [*BEREL: BERT Embeddings for Rabbinic-Encoded Language*](https://huggingface.co/dicta-il/BEREL), revision `029fa610debddd0cd798f8babc33e41388fb2bac`.

Chen, Jianlv, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024. “M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation.” In *Findings of the Association for Computational Linguistics: ACL 2024*, 2318-2335.

Zhang, Xin, Yanzhao Zhang, Dingkun Long, Wen Xie, Ziqi Dai, Jialong Tang, Huan Lin, Baosong Yang, Pengjun Xie, Fei Huang, Meishan Zhang, Wenjie Li, and Min Zhang. 2024. “mGTE: Generalized Long-Context Text Representation and Reranking Models for Multilingual Text Retrieval.” In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track*, 1393-1412.

Zhang, Yanzhao, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. 2025. [*Qwen3-Embedding-8B*](https://huggingface.co/Qwen/Qwen3-Embedding-8B), revision `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`.

Smiley, David M. [“MiqraBERT: Regression-Based Sentence-BERT Finetuning for Biblical Hebrew Parallel Detection.”](https://doi.org/10.48550/arXiv.2606.19638) 2026.

Spärck Jones, Karen. 1972. “A Statistical Interpretation of Term Specificity and Its Application
in Retrieval.” *Journal of Documentation* 28 (1): 11-21. https://doi.org/10.1108/EB026526.

## License

MIT. BHSA source data is distributed separately under its stated terms.
