# Feature documentation

## Feature naming

`semantic_<model>_<variation>`, e.g. `semantic_bge_m3_vocalized`,
`semantic_openai_text_embedding_3_large_cantillation`.

`<model>` is the model's technical identifier, e.g. `bge_m3`,
`gemini_embedding_2`, `openai_text_embedding_3_large`.

Each value is base64-encoded raw float32 bytes of the embedding vector:

```python
import base64
import numpy as np

vector = np.frombuffer(base64.b64decode(value), dtype="<f4")
```

Every feature is scoped to the book of Psalms: values exist for its
5,203 `half_verse` nodes, not for the rest of BHSA's `half_verse` node
space.

## Models

| Model | Technical identifier | Feature slug |
| --- | --- | --- |
| MiqraBERT | `davidmsmiley/MiqraBERT` | `miqrabert` |
| AlephBERT | `imvladikon/sentence-transformers-alephbert` | `alephbert` |
| NeoDictaBERT | `dicta-il/neodictabert-bilingual-embed` | `neodictabert` |
| BEREL | `dicta-il/BEREL` | `berel` |
| bge-multilingual-gemma2 | `BAAI/bge-multilingual-gemma2` | `bge_multilingual_gemma2` |
| Qwen3-Embedding-8B | `Qwen/Qwen3-Embedding-8B` | `qwen3_embedding_8b` |
| KaLM-Embedding-Gemma3-12B | `tencent/KaLM-Embedding-Gemma3-12B-2511` | `kalm_embedding_gemma3_12b_2511` |
| Llama-Embed-Nemotron-8B | `nvidia/llama-embed-nemotron-8b` | `llama_embed_nemotron_8b` |
| BGE-M3 | `BAAI/bge-m3` | `bge_m3` |
| GTE-multilingual-base | `Alibaba-NLP/gte-multilingual-base` | `gte_multilingual_base` |
| mE5-large-instruct | `intfloat/multilingual-e5-large-instruct` | `me5_large_instruct` |
| Gemini Embedding 2 | `google/gemini-embedding-2` | `gemini_embedding_2` |
| OpenAI text-embedding-3-large | `openai/text-embedding-3-large` | `openai_text_embedding_3_large` |
| Cohere Embed v4 | `embed-v4.0` | `cohere_embed_v4` |
| Voyage 4 | `voyageai/voyage-4` | `voyage_4` |

## Citations

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

## Variations

* `consonantal` - bare consonants, no niqqud, no cantillation
* `vocalized` - niqqud (vowel points) only, no cantillation marks
* `cantillation` - niqqud and cantillation/accent marks together

## Usage

```python
from tf.fabric import Fabric

TF = Fabric(locations=[
    "path/to/bhsa/tf/2021",
    "path/to/tehillim-embeddings/tf/1.0",
])
api = TF.load("semantic_bge_m3_vocalized semantic_openai_text_embedding_3_large_cantillation")
```
