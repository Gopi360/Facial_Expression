r"""Fine-tune MobileNetV3-Small on ear squares and export ONNX.

Run from the repo root:
    .venv-train\Scripts\python training\train.py --data dataset\ears --out models\ears.onnx
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import models  # noqa: E402

from app.core.config import CONFIG  # noqa: E402
from training.split import split_by_run  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image(path: Path, size: int) -> np.ndarray:
    """RGB float32 in 0..1, HxWx3, resized to size x size."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0


def augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Small shift, brightness/contrast jitter and a random horizontal flip."""
    if rng.random() < 0.5:
        img = img[:, ::-1]
    size = img.shape[0]
    shift = max(1, size // 12)
    dx, dy = rng.randint(-shift, shift), rng.randint(-shift, shift)
    img = np.roll(img, (dy, dx), axis=(0, 1))
    contrast = rng.uniform(0.8, 1.2)
    brightness = rng.uniform(-0.1, 0.1)
    return np.clip((img - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)


def to_tensor(imgs: list[np.ndarray]) -> torch.Tensor:
    arr = (np.stack(imgs) - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 3, 1, 2)))


def batches(items, size, shuffle, rng):
    items = list(items)
    if shuffle:
        rng.shuffle(items)
    for i in range(0, len(items), size):
        yield items[i : i + size]


def run_epoch(model, data, optimiser, loss_fn, rng):
    model.train(True)
    # Keep every BatchNorm layer in eval mode in both phases: the ear dataset is small, so letting
    # the running statistics drift would quietly change the "frozen" backbone that gets exported.
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
    total, loss_sum = 0, 0.0
    for chunk in batches(data, 32, True, rng):
        imgs = [augment(img, rng) for img, _ in chunk]
        x = to_tensor(imgs)
        y = torch.tensor([lab for _, lab in chunk])
        loss = loss_fn(model(x), y)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        loss_sum += loss.item() * len(chunk)
        total += len(chunk)
    return loss_sum / max(total, 1)


def evaluate(model, data, n_labels):
    model.eval()
    table = [[0] * n_labels for _ in range(n_labels)]  # rows true, columns predicted
    with torch.no_grad():
        for chunk in batches(data, 64, False, None):
            pred = model(to_tensor([img for img, _ in chunk])).argmax(1).tolist()
            for (_, true), p in zip(chunk, pred):
                table[true][p] += 1
    return table


def print_report(labels, table):
    print("\nHeld-out accuracy per label:")
    for i, name in enumerate(labels):
        n = sum(table[i])
        acc = table[i][i] / n if n else float("nan")
        print(f"  {name:10s} {acc:6.1%}  ({table[i][i]}/{n})")
    print("\nConfusion table (rows = true, columns = predicted):")
    print("  " + " " * 10 + "".join(f"{n:>10s}" for n in labels))
    for i, name in enumerate(labels):
        print(f"  {name:10s}" + "".join(f"{c:10d}" for c in table[i]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="folder with one subfolder of PNGs per label")
    ap.add_argument("--out", type=Path, required=True, help="where to write the .onnx file")
    ap.add_argument("--epochs", type=int, default=8, help="passes over the data in total (default 8)")
    args = ap.parse_args()

    size = CONFIG.ear_input_px
    labels = sorted(d.name for d in args.data.iterdir() if d.is_dir())
    if len(labels) < 2:
        sys.exit(f"Need at least two label folders in {args.data}, found {labels}")
    paths = sorted(p for lab in labels for p in (args.data / lab).glob("*.png"))
    print(f"{len(paths)} photos, labels {labels}")

    train_p, test_p = split_by_run(paths)
    if not train_p or not test_p:
        sys.exit("Not enough runs to split into train and test. Collect photos over more Start-to-Stop runs.")
    print(f"train {len(train_p)} photos, test {len(test_p)} photos (split by run)")

    for name, side in (("train", train_p), ("test", test_p)):
        for lab in labels:
            if not any(p.parent.name == lab for p in side):
                sys.exit(
                    f"Label '{lab}' has no photos on the {name} side (it may be empty, or all its runs landed on "
                    f"the other side). Take more Start-to-Stop runs of '{lab}' and try again."
                )

    idx = {lab: i for i, lab in enumerate(labels)}
    train = [(load_image(p, size), idx[p.parent.name]) for p in train_p]
    test = [(load_image(p, size), idx[p.parent.name]) for p in test_p]

    random.seed(0)
    torch.manual_seed(0)
    rng = random.Random(0)

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(labels))
    loss_fn = nn.CrossEntropyLoss()

    head_epochs = max(1, args.epochs // 2)
    tune_epochs = max(0, args.epochs - head_epochs)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    opt = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    t_start = time.time()
    for e in range(head_epochs):
        t = time.time()
        loss = run_epoch(model, train, opt, loss_fn, rng)
        print(f"pass {e + 1}/{args.epochs} (head only) loss {loss:.4f} {time.time() - t:.1f}s")

    if tune_epochs:
        for p in model.features[-3:].parameters():
            p.requires_grad = True
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=1e-4)
        for e in range(tune_epochs):
            t = time.time()
            loss = run_epoch(model, train, opt, loss_fn, rng)
            print(f"pass {head_epochs + e + 1}/{args.epochs} (last blocks) loss {loss:.4f} {time.time() - t:.1f}s")
    print(f"Training time {time.time() - t_start:.1f}s")

    print_report(labels, evaluate(model, test, len(labels)))

    model.eval()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.zeros(1, 3, size, size),
        str(args.out),
        opset_version=17,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    meta = args.out.with_suffix(".json")
    meta.write_text(json.dumps({"labels": labels, "input_px": size}), encoding="utf-8")
    print(f"\nWrote {args.out} and {meta}")


if __name__ == "__main__":
    main()
