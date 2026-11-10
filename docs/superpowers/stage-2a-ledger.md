# Stage 2a ledger: ear visibility

Plan: `docs/superpowers/plans/2026-11-10-stage-2a-ear-visibility.md`. Spec: `docs/superpowers/specs/2026-11-10-stage-2a-ear-visibility-design.md`. Branch `feat/stage-2a-ears`.

Rulings, fix rounds and deferred minor issues, copied from the working ledger and grouped by task. On-camera results are at the end.

## Task 1

- minor (deferred): no boundary tests for sides_to_check(±20), squares cut at top/bottom/left edges, or size<=0 branch; long line in types.py; RED reproduced after writing code

## Task 2

- minor (deferred): run id has 1 s resolution, two runs started in the same second overwrite each other (told owner to wait between runs)
- minor (deferred): collect buttons show pressed even if the socket send was dropped (before camera start); stale pressed state after session end
- minor (deferred): no Pipeline-level test of collector wiring (set_collect, offer, collect_counts, make_pipeline collect=True)
- minor (deferred): counts() globs dataset folders every frame in collect mode
- minor (deferred): every non-far face in frame is saved under the run's label (bystanders mislabelled; owner told to be alone in frame)
- Step 10: ear-size check passed (squares 44-52 px, clear vs covered visible by eye). Ruling: widened ear square, ear_offset_frac 0.08->0.15 and ear_square_frac 0.3->0.4, owner approved; geometry tests pinned to fixed values — the outer ear edge and top were cut off — costs a re-collect if the new crop overshoots. Test photos deleted with owner approval. Commit 5b89f73.
- Step 10: Ruling: ear square 0.15/0.4 overshot outward (inner ear edge clipped) -> 0.10/0.45, commit 376883d; second test photos deleted — costs one more owner check round
- Step 10: crop 0.10/0.45 fits the whole ear (squares 64-74 px). Ruling: ear_yaw_sign 1 -> -1, measured: turned frames cropped eye/glasses, not an ear; tests pinned to explicit sign — costs nothing if wrong beyond flipping back. Commit 6e5cc09. Third test photos deleted.

## Task 3

- review 1 — Important: frozen backbone BatchNorm updates in train mode; no guard for a label with 0 train/0 test photos. Ruling: keep all BatchNorm in eval in both phases — small dataset — costs slightly lower accuracy if wrong.
- minor (deferred): training resizes with PIL BILINEAR (antialiased); Task 5 must match — Ruling for Task 5: app resizes ear crops with cv2.INTER_AREA, the closest antialiased cv2 mode — costs a small train/serve skew if wrong
- minor (deferred): np.roll shift wraps edge pixels; one-run labels get no test photos; local _run helper in tests; dynamo=False deprecation warning (pinned torch); no class weighting; batch/lr hard-coded
- fix round 1 dispatched (FIX_BASE a293bd4)
- fix round 1/5 (3 addressed, 0 open; commits a293bd4..7f69a3b)
- minor (deferred): no automated test for the missing-label guard or for BN staying in eval
- complete (commits 6e5cc09..7f69a3b, review clean). Step 8 (owner photos + training) pending; running Task 4 meanwhile — Ruling: continue pure-logic tasks while the owner collects — Task 4 needs no photos — costs nothing if wrong.

## Task 4

- implementer concern (grace only applied on no-check calls; stale run survives a person leaving) — Ruling: apply grace at the start of every update — the pipeline does not update absent people — costs nothing if wrong. Sent back before review (54bd9dc).
- minor (deferred): mid-file import in tests/test_ears.py; no side-switch test once already hidden; no handling of now_ms going backwards

## Task 5

- Ruling (implementer, accepted at review): when ear state drops to unknown via grace, the #ears run gets _owed_snap instead of a silent pop — ear grace (2 s) ends before letter release (2 s) — costs a missing owed ear snap if wrong
- Ruling: test timings adjusted to 400 ms check spacing at 100 ms frames (5 checks in 2 s; visible-again 1.5 s; leaving test hidden min_red_ms-STEP_MS) — brief values were impossible at that spacing — none
- minor (deferred): no test for ear-red person going far (owed _A_ears.jpg, no snap while far); no pipeline test for covered returning {}; far ear-hidden person drawn red in others' snaps during grace; return within grace while hidden gives a second ear snap; hidden->unknown->hidden logs a second "ears are not visible"; ear checks outside the one-heavy-call budget (watch frame time); test_ear_model_missing_raises first raises lacks match

## Task 6

- implementer DONE fe0f934 (511 pass, 1 skip). Sent back before review: deferred-items pointer targeted git-ignored ledger; README lacked restart-after-training note — Ruling: commit docs/superpowers/stage-2a-ledger.md as the durable record — the sdd workspace is deleted at finish — costs nothing if wrong.
- review 1 approved, but the controller confirmed a real gap: `.strip-people` is a 2-column grid, so the ear line sat beside its person. Fix round 1: one wrapper div per person, plus the `REAL_JPEG` rename and an isolated corrupt-onnx test.
- fix round 1/5 (3 addressed, 0 open; commits 5f3c123..8f123cc)
- minor (deferred): test_missing_ear_model_runs_without_ears partly redundant, weak second assertion; README "Checks to run" lists 2 ear checks against the spec's 8
- complete for steps 1-5 (commits b17b44e..8f123cc, review clean)

## Photos, training and on-camera checks (2026-11-10)

- Photos: 1013 clear, 459 covered, over 7 covered runs, glasses on, the owner only.
- Some squares land on the eye and glasses lens instead of the ear, when the head is turned far or a hand or cloth covers the side of the face (MediaPipe's outline drifts inward). They are in both labels. Left in for now.
- Training on the laptop took 25 s (8 passes). Held-out runs: clear 98.3% (396/403, 7 called covered), covered 100% (79/79). The covered test set is small.
- On camera, the owner reports all checks pass: sitting normally and head turns not flagged, hand over an ear flags and clears, sleep still sounds. Per-check timings and the frame rate were not recorded.
- No numbers were tuned after training. Config stays: `ear_covered_at` 0.8, holds 1000/500 ms, grace 2000 ms, square 0.10/0.45, `ear_yaw_sign` -1. No `test_owner_ears_*` tests yet, since nothing was tuned from readings.
- Not done: the final whole-branch review (subagent-driven process) before merging `feat/stage-2a-ears`.
