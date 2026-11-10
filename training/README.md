# Ear model training

Trains the small `clear` / `covered` ear classifier from the photos taken in photo mode (`?collect=1`). The app itself never needs PyTorch; only this folder does.

## 1. Make the training setup

From the repo root, with the same Python version as the app (3.12):

```
python -m venv .venv-train
.venv-train\Scripts\pip install -r training\requirements-train.txt
```

This downloads CPU-only PyTorch (a few hundred MB). `.venv-train/` is git-ignored. The ImageNet base weights are downloaded once, on the first training run, to the torch cache. Training is the only step that touches the network.

## 2. Photos per label

About 100 to 200 per label, taken over several Start-to-Stop runs (the split is by run, so one run gives the test set nothing to test on). Vary head turn and light, glasses on. Cover clear, hand over the left ear, hand over the right ear, and hair over an ear if possible.

## 3. What counts as `covered`

- `clear`: the ear is visible. An ear only partly seen because of the head angle is `clear`. Glasses on, front-on and turned, different light. These are the cases that must not flag.
- `covered`: most of the ear is hidden by something in front of it (hand, hair pulled over, hood, scarf).
- Leave out: borderline photos where the ear is about half hidden. Delete them rather than guess.

## 4. Train

```
.venv-train\Scripts\python training\train.py --data dataset\ears --out models\ears.onnx
```

Optional: `--epochs N` (default 8, total passes: the first half trains only the new last layer, the second half also trains the last blocks at a lower rate). It writes `models\ears.onnx` and `models\ears.json` (`{"labels": [...], "input_px": 96}`, labels are the sorted folder names).

**Restart the app after training.** The server loads `models\ears.onnx` once and remembers a missing model until it stops. Stop it, start it again and reload the page: the "Ear check is off" warning then no longer shows.

## 5. Read the report

- Held-out accuracy per label: the share of test photos of each label that it got right. The test photos come from whole runs the model never saw.
- Confusion table: rows are the true label, columns what the model said. For this app the costly error is a `clear` ear predicted `covered` (a false flag), so look at that cell first.
- If test accuracy is far below the last training loss would suggest, you have too few runs. Take more runs in different light.
- Note the training time. If it is far over about 30 minutes, use the Colab fallback.

## 6. Colab fallback

If training on the laptop fails or is too slow, the same script can be run on a free Colab GPU: owner's decision, nothing is set up for it yet.

## Licenses

Looked up 2026-11-10 from primary sources.

| Piece | License found | Source |
|---|---|---|
| torchvision code (`mobilenet_v3_small` architecture) | **BSD 3-Clause** (LICENSE file, copyright Soumith Chintala 2016) | [pytorch/vision LICENSE](https://raw.githubusercontent.com/pytorch/vision/main/LICENSE) |
| MobileNetV3-Small ImageNet weights (`IMAGENET1K_V1`) | **Not confirmed.** The model page states no license for the weights. The torchvision models page says pretrained weights "may have their own licenses or terms and conditions derived from the dataset used for training" and that it is "your responsibility to determine whether you have permission to use the models for your use case". The weights were trained on ImageNet, whose terms were not checked. | [Model page](https://docs.pytorch.org/vision/main/models/mobilenetv3.html), [Models overview](https://docs.pytorch.org/vision/main/models.html) |

**Blocker for commercial or exam use:** the ImageNet base weights have no stated license and may carry ImageNet's terms. The fine-tuned `ears.onnx` is derived from them. Confirm before any commercial or exam use. (Fine-tuning on the owner's own photos does not remove the base weights' terms.)
