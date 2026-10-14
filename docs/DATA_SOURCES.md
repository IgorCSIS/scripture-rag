# Data sources and license evidence

## Bundled text
| File | Content | Translation / edition | Status |
|---|---|---|---|
| `tests/fixtures/books/kjv_sample.txt` | Genesis 1:1-5, Psalm 23:1-4, Matthew 5:3-9 (excerpts) | King James Version, 1611 | Public domain in the US (UNVERIFIED this session; note the KJV is under Crown letters patent in the UK) |
| `tests/fixtures/books/tao_te_ching_legge.txt` | Chapters 1 and 8 (excerpts, lightly abridged) | James Legge, 1891 | Public domain by age (UNVERIFIED this session) |
| `tests/fixtures/books/dhammapada_muller.txt` | Verses 1 and 5 | F. Max Muller, 1881 | Public domain by age (UNVERIFIED this session) |
| `site/assets/sample/psalm23_kjv.txt` | Psalm 23 | King James Version, 1611 | As above; used by the browser demo "Add sample" button |
| `tests/fixtures/injection.txt` | Written for the prompt-injection test | Original | Project content |

Excerpts were typed from the well-known public-domain wording; they are test fixtures, not authoritative editions.

## Demo books (pending)
`data/demo/` is empty. Candidates: a KJV Bible, the Dhammapada (Max Muller), Legge's Tao Te Ching. For each, record the Internet Archive identifier and the `licenseurl`, `rights`, `possible-copyright-status`, `date` and `publicdate` fields here, and get Igor's confirmation before committing. IA fields are uploader supplied evidence, not proof (https://archive.org/developers/metadata-schema/).

## Models (downloaded at runtime, not bundled)
`intfloat/multilingual-e5-small`, `Xenova/multilingual-e5-small` (browser), `cross-encoder/ms-marco-MiniLM-L-6-v2`, optional `BAAI/bge-*`. Check each model card license before redistribution.
