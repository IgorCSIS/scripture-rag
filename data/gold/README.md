# Gold set

Format: one JSON object per line.

```json
{"question": "Who is my shepherd?", "answerable": true, "book_id": "kjv", "page": 2}
{"question": "What is the capital of France?", "answerable": false, "book_id": null, "page": null}
```

Planned size: 30 answerable (with book and PDF page), 15 unanswerable, 5 adversarial, plus 5 PT and 5 ES items for FR-14. Pending: the demo books must be chosen first, then Igor reviews the items. A 12 item smoke set over the test fixtures lives in `tests/fixtures/gold_fixture.jsonl`.
