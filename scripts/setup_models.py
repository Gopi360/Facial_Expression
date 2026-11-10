"""Download the model files into models/ once. Existing files are skipped."""
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

FILES = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "emotion.onnx": "https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx",
    "sface.onnx": "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
    "yunet.onnx": "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
    "sample_face.png": "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.22.0/skimage/data/astronaut.png",
}


def download(name: str, url: str) -> None:
    target = MODELS_DIR / name
    if target.exists():
        print(f"skip  {name} (already there)")
        return
    tmp = MODELS_DIR / (name + ".part")
    print(f"fetch {name}")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        tmp.replace(target)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"could not download {name} from {url}: {exc}") from exc


def main() -> int:
    try:
        MODELS_DIR.mkdir(exist_ok=True)
        for name, url in FILES.items():
            download(name, url)
    except Exception as exc:
        print(f"Model setup failed: {exc}", file=sys.stderr)
        return 1
    print("All model files are in models/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
