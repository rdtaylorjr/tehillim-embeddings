"""Morphological half-verse representations over BHSA word-level grammatical features."""

#: The `domain=` Hive partition every dataset this package writes lands under.
DOMAIN = "morphological"

#: The `feature=` Hive partition each representation family is written under.
ATOMIC_FEATURE = "morph_atomic"
SIGNATURE_FEATURE = "morph_signature"
SUFFIX_FEATURE = "morph_suffix"
