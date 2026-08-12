# Tehillim Embeddings

A Text-Fabric companion module containing pre-computed embedding
vectors for every half-verse of the Hebrew psalms, produced by 15
different Hebrew and multilingual embedding models.

See also [about](docs/about.md) and
[feature documentation](docs/feature_documentation.md).

## About

Part of [tehillim](https://github.com/rdtaylorjr/tehillim), a
computational analysis of the Hebrew psalms.

Uses BHSA's `half_verse` (colon) node numbering. Built and tested
against BHSA version `2021`.

## Data

* **Feature files**: 31 Text-Fabric feature files in `tf/1.0/`, one per
  (model, variation) pair, named `semantic_<model>_<variation>`. Values are
  base64-encoded float32 embedding vectors, keyed to BHSA's
  `half_verse` nodes. See
  [feature documentation](docs/feature_documentation.md) for the naming
  scheme, the three variations, and the full model list.
* **Generation code**: `programs/`, reading tehillim's cached embeddings
  and writing them out as Text-Fabric features. Run with
  `tehillim/pipeline/.venv/bin/python3`, with `tehillim` and
  `tehillim-embeddings` as sibling directories, after `tehillim-build`
  or a Colab heavy-model run has populated
  `tehillim/pipeline/data/embeddings_cache/`. Safe to re-run.

## Family

* [tehillim](https://github.com/rdtaylorjr/tehillim) - computational
  analysis of the Hebrew psalms
* [bhsa](https://github.com/etcbc/bhsa) - the core text and linguistic
  annotation for the Hebrew Bible

## License

MIT

## Author

* [Russell D. Taylor Jr.](mailto:rdtaylorjr@protonmail.com)
