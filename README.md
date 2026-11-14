# BRTP

BRTP implements shared prompt learning for CLIP classifiers. It keeps the image and text encoders frozen, constrains changes to class representations, and transfers residuals between classes. Its Python modules implement the model, training objective, native-label evaluation, and checks for prepared runs.

## BRTP performance

- B2N overall harmonic mean: **80.57%**
- CD mean target accuracy: **62.65%**
- DG mean target accuracy: **85.28%**

The [evaluation counts and calculation instructions](data/) describe these measurements.

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

To use the model with your own data and training settings, import [the model](brtp/models/residual_transport.py), [the training objective](brtp/objective.py), or [the evaluation functions](brtp/evaluation/native.py) directly. Those interfaces do not require the Full protocol archive. See [third-party sources](THIRD_PARTY.md) and the [project license notice](LICENSE.md).
