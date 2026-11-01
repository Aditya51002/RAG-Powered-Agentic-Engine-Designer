# Independent QA Review Packet

`qa_review_packet.csv` is generated from the current draft QA dataset by
`python scripts/export_qa_review_packet.py`. The export validates the dataset against the
corpus checksum and page-addressable PDF before writing. It intentionally omits each case's
current answerability label, reference answer, and relevant page labels so the reviewer can
work independently.

Give the reviewer this CSV and the corpus PDF listed in each row. For each case, independently
decide whether the corpus answers the question, write a concise answer or an explicit
unanswerable result, and list every supporting source in the stable
`repository/path.pdf#page=N` format. Separate multiple source IDs with semicolons. For an
unanswerable judgment, search the entire corpus; the packet does not disclose the draft label.

Fill `reviewed_answerable` with `true` or `false`, then complete the answer and source fields
consistently. Mark `searched_full_corpus=true` when the entire corpus was checked; this is
required for an unanswerable judgment. Record reviewer name, review date as an ISO 8601 UTC
timestamp (for example, `2026-11-01T10:00:00Z`), and any rationale or ambiguity. Do not change
the dataset version, hash, corpus version, case ID, question, or corpus references.

After review, compare the packet with the current draft:

```sh
python scripts/compare_qa_review.py --packet data/evaluation/review/qa_review_packet.csv
```

The comparer validates all rows, metadata, labels, and cited pages, then writes
`reports/evaluation/qa_review_comparison.json`. Disagreements require human adjudication.
The report never changes `qa_set.json` or sets `label_review_status`; update and validate a
new dataset revision only after the reviewer and adjudicator agree on the final labels.
