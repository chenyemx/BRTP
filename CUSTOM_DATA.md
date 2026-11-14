# Train BRTP on your own images

Place `train_custom.py` at the repository root alongside `train.py`. Install the project dependencies and provide a local OpenAI CLIP weights file. The script accepts separate training and evaluation folders. Each folder contains one subdirectory per class. The subdirectory name becomes the class prompt, with underscores replaced by spaces. Provide at least two distinct source classes. For a few-shot run, put only the labeled support images in the training folder.

```text
train_images/
  bridge/
  forest/
test_images/
  bridge/
  forest/
```

From the repository root:

```sh
python train_custom.py train --images train_images --weights weights/ViT-B-16.pt --checkpoint outputs/custom_prompt.pt --epochs 10
python train_custom.py evaluate --images test_images --weights weights/ViT-B-16.pt --checkpoint outputs/custom_prompt.pt
```

Training freezes both CLIP encoders, learns the shared BRTP prompt using the main and donor-exclusion losses, and saves the learned prompt, its initial context, the source-class order, and a checksum binding it to the supplied backbone. Choose a new checkpoint filename for each training run. Evaluation uses the class folders in the evaluation directory as candidate classes, including classes absent from the training folders, and reports the proportion classified correctly. Both commands accept `--device cpu` or `--device cuda`. The default is CUDA when available. The script does not require a protocol archive, fetch weights, or save per-image predictions.
