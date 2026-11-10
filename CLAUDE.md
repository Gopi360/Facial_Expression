# Facial Expression Demo

Read `docs/superpowers/HANDOVER.md` first. It has the current status, pending work and open questions.

- Stage 1 is merged into `main`. Stage 2a work is on `feat/stage-2a-ears`.
- Run tests with `.venv\Scripts\python -m pytest -q`.
- All tunable numbers live in `app/core/config.py`. `app/core/` must not import cv2, mediapipe, onnxruntime or fastapi.
- Sleep thresholds come from real readings. Don't change them without new on-camera numbers.
- Nothing leaves the laptop: no network calls at runtime.
