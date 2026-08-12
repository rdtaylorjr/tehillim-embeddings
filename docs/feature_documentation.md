# Feature documentation

## Feature naming

`semantic_<model>_<tier>`, e.g. `semantic_bge_m3_vocalized`,
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
