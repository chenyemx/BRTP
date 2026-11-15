# Evidence additions in v1.2.0

The numerical results in this directory are unchanged from v1.1.0, the version cited by the manuscript. This release adds four records that document how the experiments were set up and how some reported statements were checked. No value in `tables.csv`, `test-panels.csv` or the other result files has been modified.

| File | What it records |
|---|---|
| `splits.json` | The eight frozen split files (four native and four shared-16 views), their SHA-256 digests, class and image counts, the native-ID class order, and the label rule. |
| `per-class-recall.csv` | Per-class recall for the bounded direct classifier (F00) and the unrestricted variant (U) on the three cross-dataset targets, one row per target and seed. |
| `joint-vocabulary.csv` | The joint base-and-new vocabulary comparison, including which prediction branch each method used. |
| `baseline-provenance.csv` | Per-row provenance of the comparison tables: scores transcribed from FrogDogNet, scores from local matched retraining, and the matched control. |

## Provenance

- `splits.json` is compiled from the recorded split provenance of the training protocol: each entry gives the APPLeNet release file, its SHA-256, and the counts before support-set sampling. The class-order rule follows the explicit native IDs used in training.
- `per-class-recall.csv` was recomputed from the archived evaluation predictions of F00 and U (three seeds, three cross-dataset targets). Class recall is the fraction of images of a class that are predicted as that class; the paper's comparison is the mean over these nine panels.
- `joint-vocabulary.csv` reproduces the recorded prediction branch and the base, new and harmonic scores of the joint-vocabulary run.
- `baseline-provenance.csv` separates published baselines from methods trained or evaluated here, so a reader of Tables 2–4 and 7 can identify the origin of each row without consulting the text.

## Reproducing the recall record

The recall numbers are derived from per-image logits and labels. Those prediction arrays are not redistributed in this package, so the record can be checked against the archived predictions rather than recomputed from the files here. All other files remain verifiable with `python verify.py`, which now also checks the digests of the four records above and of this note.
