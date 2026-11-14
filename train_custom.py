"""Train or evaluate BRTP using local image folders and local CLIP weights."""

import argparse
from pathlib import Path
import os
import random
import tempfile

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from brtp.models.adapter import build_model, training_loss
from brtp.utils.artifacts import file_sha256
from clip.model import build_model as build_clip


def dataset(folder, size, training):
    if training:
        geometry = [
            transforms.RandomResizedCrop(size, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        geometry = [
            transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
        ]
    transform = transforms.Compose(
        geometry
        + [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    images = datasets.ImageFolder(folder, transform=transform)
    names = [name.replace("_", " ").strip() for name in images.classes]
    if len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError(
            "Class folders must have distinct, nonempty names after replacing underscores."
        )
    return images, names


def save_new(checkpoint, value):
    """Write a complete checkpoint without overwriting another run."""
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=checkpoint.parent, delete=False) as temporary:
        staged = Path(temporary.name)
    try:
        torch.save(value, staged)
        if os.name == "nt":
            # Windows rename fails atomically if the destination already exists.
            os.rename(staged, checkpoint)
        else:
            os.link(staged, checkpoint)
    finally:
        staged.unlink(missing_ok=True)


def train(args, backbone, device):
    if args.checkpoint.exists():
        raise FileExistsError(f"Choose a new checkpoint path: {args.checkpoint}")
    if args.epochs < 1 or not 0 < args.lr < float("inf"):
        raise ValueError(
            "Use a positive epoch count and finite positive learning rate."
        )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    images, names = dataset(
        args.images, backbone.visual.input_resolution, training=True
    )
    loader = DataLoader(images, batch_size=args.batch_size, shuffle=True)
    model = build_model(backbone, {}, names, device=device)
    model.train()
    optimizer = torch.optim.SGD(
        [model.prompt_learner.ctx], lr=args.lr, momentum=0.9, weight_decay=0.0005
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            features = model.encode_image(batch.to(device))
            loss, _ = training_loss(
                model, features, names, labels.to(device), lambda_weight=0.25
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Loss became non-finite.")
            loss.backward()
            if not torch.isfinite(model.prompt_learner.ctx.grad).all():
                raise FloatingPointError("Prompt gradient became non-finite.")
            optimizer.step()
            model.clear_derived_cache()
            running_loss += loss.item() * len(labels)
        scheduler.step()
        print(
            f"Epoch {epoch + 1}/{args.epochs}: mean training loss {running_loss / len(images):.4f}"
        )
    save_new(
        args.checkpoint,
        {
            "format": "brtp-folder-1",
            "source_names": names,
            "backbone_sha256": file_sha256(args.weights),
            "prompt": {
                k: v.detach().cpu()
                for k, v in model.prompt_learner.state_dict().items()
            },
        },
    )


@torch.no_grad()
def evaluate(args, backbone, device):
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if state.get("format") != "brtp-folder-1" or state[
        "backbone_sha256"
    ] != file_sha256(args.weights):
        raise ValueError("The checkpoint format or CLIP backbone does not match.")
    model = build_model(
        backbone, {}, state["source_names"], prompt_state=state["prompt"], device=device
    )
    model.eval()
    images, candidates = dataset(
        args.images, backbone.visual.input_resolution, training=False
    )
    correct = 0
    for batch, labels in DataLoader(images, batch_size=args.batch_size):
        features = model.encode_image(batch.to(device))
        logits = model.classify_features(features, candidates)
        correct += int((logits.argmax(1).cpu() == labels).sum())
    print(
        f"Top-1 accuracy: {100 * correct / len(images):.2f}% ({correct}/{len(images)})"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--weights", type=Path, required=True, help="Local OpenAI CLIP weights file"
    )
    shared.add_argument(
        "--images", type=Path, required=True, help="Folder with one subfolder per class"
    )
    shared.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Output for train; input for evaluate",
    )
    shared.add_argument("--batch-size", type=int, default=8)
    shared.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    actions = parser.add_subparsers(dest="action", required=True)
    trainer = actions.add_parser("train", parents=[shared])
    trainer.add_argument("--epochs", type=int, default=10)
    trainer.add_argument("--lr", type=float, default=0.002)
    trainer.add_argument("--seed", type=int, default=1)
    actions.add_parser("evaluate", parents=[shared])
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not args.weights.is_file() or not args.images.is_dir():
        parser.error(
            "--weights must name a local file and --images an existing directory"
        )
    device = torch.device(args.device)
    # File objects also support weight paths with non-ASCII directory names on Windows.
    with args.weights.open("rb") as stream:
        try:
            state = torch.jit.load(stream, map_location="cpu").state_dict()
        except RuntimeError:
            stream.seek(0)
            state = torch.load(stream, map_location="cpu", weights_only=True)
    backbone = build_clip(state)
    del state
    backbone = backbone.float().requires_grad_(False).to(device).eval()
    if args.action == "train":
        train(args, backbone, device)
    else:
        evaluate(args, backbone, device)


if __name__ == "__main__":
    main()
