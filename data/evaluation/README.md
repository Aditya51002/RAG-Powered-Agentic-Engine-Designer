# Evaluation QA Set

`qa_set.json` is the version-controlled artifact location, not a populated benchmark. It is empty because this repository currently has no verified source corpus or hand-labeled questions. Do not report retrieval quality or RAGAS scores from synthetic tests as production results.

Each case must have a stable `id`, a specific `question`, a source-grounded `reference_answer`, and one or more `relevant_source_ids` that resolve to the exact source identifiers used by retrieval. Increment `dataset_version` when labels change and set `source_corpus_version` to the immutable corpus revision evaluated. Have labels independently reviewed before using this set as a release gate.
