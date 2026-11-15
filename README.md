# BRTP

BRTP implements shared prompt learning for CLIP classifiers. It keeps the image and text encoders frozen, constrains changes to class representations, and transfers residuals between classes. Its Python modules implement the model, training objective, native-label evaluation, and checks for prepared runs.

## BRTP performance

- B2N overall harmonic mean: **80.57%**
- CD mean target accuracy: **62.65%**
- DG mean target accuracy: **85.28%**

The [evaluation counts and calculation instructions](data/) describe these measurements.

## Results package

The [results package](results/) records the numerical evidence of the paper:

- `tables.csv`: every value printed in Tables 1–11 and A.1–A.4, with the value checked against the saved evidence and the kind of evidence used (762 entries).
- `test-panels.csv`: 2,844 model, task, source, target, seed and test-vocabulary panels rebuilt from the saved test predictions, giving test images, competing labels, correct predictions and accuracy.
- `fig4-mechanisms.csv`, `fig5-sensitivity.csv`, `fig6-ten-seeds.csv` and `fig6-summary.csv`: the matched-configuration comparison, the shot and radius sweeps, and the ten-seed scores behind Figures 4–6.
- `class-geometry.csv`: per-class, per-seed coverage, residual length, bound indicators and final angle for F00, the data behind Table 6.
- `external-targets.csv` and `calibration.csv`: per-seed external-target scores and calibration errors behind Tables A.3 and A.4.
- `protocol.txt`: dataset counts, support construction, training settings and the aggregation rule for each metric.
- `splits.json`, `per-class-recall.csv`, `joint-vocabulary.csv`, `baseline-provenance.csv` and `EVIDENCE.md`: split provenance with digests, the per-class recall record of the F00 and U comparison, the prediction branch used in the joint-vocabulary run, and the provenance of each comparison row (added in v1.2.0).
- `verify.py` and `SHA256SUMS`: the recomputation entry point and the digests it checks.

Run `python results/verify.py` from any directory. It needs Python 3 and the files beside it, verifies the digests and all 2,844 panel accuracies, rebuilds the main-table values of the five locally trained methods, checks the ten-seed summaries and prints the three headline scores. The package holds no images, model weights, per-image predictions or third-party texts. Release v1.1.0 is the version cited by the manuscript; v1.2.0 adds the evidence records without changing any reported value.

## Install and check

Use Python 3.10 on Linux for CUDA execution. Install a matching PyTorch and torchvision pair, then the remaining dependencies. The CUDA runner expects the following versions:

```sh
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python train.py --help
python train.py list
```

For software tests on a CPU, use the same PyTorch and torchvision versions from the [CPU wheel index](https://download.pytorch.org/whl/cpu). Run `python -m unittest discover -s tests -v` from the checkout root.

## Train on your own data

Use [train_custom.py](train_custom.py) to train and evaluate BRTP with local image folders and CLIP weights. The [custom-data instructions](CUSTOM_DATA.md) give the folder layout and commands.

## Use a local configuration

The supported launcher configuration is `Full`. Prepare it using your matching local protocol archive and the official OpenAI CLIP ViT-B/16 weights. The input archive supplies its training runner, dataset index, support selections, and task plan. Their checksums are pinned in [public_inputs.json](brtp/public_inputs.json). Provide a JSON object mapping dataset directory prefixes recorded in your archive to their absolute paths on this machine whenever they differ.

```sh
python train.py prepare --recipe Full --protocol-archive /path/to/full_sgd50_inputs.zip --weights /path/to/ViT-B-16.pt --path-map /path/to/paths.json --output /path/to/new_run
python train.py verify --run-dir /path/to/new_run
python train.py execute --run-dir /path/to/new_run --mode preflight
python train.py execute --run-dir /path/to/new_run --mode smoke
python train.py execute --run-dir /path/to/new_run --mode run
```

Preparation creates a new directory and validates the input bytes. Preflight checks the local data and configuration. Smoke tests exercise the training loop before the full run. If your input archive is already at `protocols/full_sgd50_inputs.zip` in your local checkout, the original `prepare` syntax still works without the new flag. Existing prepared directories retain their original execution bindings.

To use the model with your own data and training settings, import [the model](brtp/models/residual_transport.py), [the training objective](brtp/objective.py), or [the evaluation functions](brtp/evaluation/native.py) directly. Those interfaces do not require the Full protocol archive. See [third-party sources](THIRD_PARTY.md) and the [license](LICENSE.md).

## License

The BRTP code is released under the [MIT License](LICENSE). Third-party components in `CLIP/clip/` and the optional dataset configuration keep their own licenses, listed in [LICENSES/](LICENSES/) and [THIRD_PARTY.md](THIRD_PARTY.md).
