"""Morphological half-verse representations over BHSA word-level grammatical features."""

#: The `domain=` Hive partition every dataset this package writes lands under.
DATASET_TYPE = "morphological"

#: The `feature=` Hive partition each representation family is written under.
ATOMIC_UNIT = "morph_atomic"
SIGNATURE_UNIT = "morph_signature"
SUFFIX_UNIT = "morph_suffix"
