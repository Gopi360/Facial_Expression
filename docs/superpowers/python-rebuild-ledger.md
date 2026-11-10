# SDD ledger — plan: docs/superpowers/plans/2026-11-06-python-rebuild.md

## Pre-flight scan
| Pair / task | Produces vs consumes | Finding |
|---|---|---|
| T1 -> all | types.py, CONFIG | Agree; snake_case of TS keys |
| T2 -> T6 | Baseline, PersonClassifier.update(emotion) | Agree |
| T3 -> T6 | Tracker ids, Identity methods | '?' letter collides; see ruling |
| T4 -> T6 | SessionFiles, SnapPolicy, diff_events, render_snap | '?' in file names; see ruling |
| T5 -> T6 | FaceFinder, EmotionModel, FaceSigner | Agree; num_faces 8 then Tracker keeps 4, consistent with "processed" |
| T6 -> T7 | Pipeline, FrameResult | Agree |
| T7 -> T8 | ?debug=1, server command | Agree |
| T1 self | Deletes TS, adds tooling | OK; TS stays on feat/expression-demo |
| T2 self | Tests vs rules | OK |
| T3 self | OK apart from '?' | See ruling |
| T4 self | OK | OK |
| T5 self | Needs real models | OK, marked tests |
| T6 self | OK | OK |
| T7 self | Page has no unit tests | Accepted; browser check |
| T8 self | Needs a human on camera | See ruling |

Ruling: People after Z keep separate state under an internal key such as ?<track id>, shown as Person ? — sharing one '?' key would merge different people's sleep state — cost if wrong: small extra key logic.
Ruling: Snap file names use "unknown" instead of '?' — Windows forbids '?' in file names — cost if wrong: none.
Ruling: Task 8 agent writes the README only; on-camera pass goes to the owner — an agent cannot act out expressions — cost if wrong: thresholds stay unproven until the owner tests.
Ruling: Task 1 keeps all old config keys until Task 2 shows which are unused — avoids guessing — cost if wrong: unused keys, pruned at final review.
Task 1: minor (deferred): setup_models has no size check on downloads; report misdescribed kept keys
Task 1: complete (commits fd3269f..9f2c147, review clean)
Ruling: Yawning displays about 2.0s after the mouth opens (1.5s yawn hold plus 0.5s label hold) — spec says "about 1.5 seconds" and exact rule is tuned on camera — cost if wrong: slightly slow label, one config value.
Ruling: Overrule plan: model-seen laughing (Happiness on top and jaw >= laugh_jaw_at) counts as a strong expression on the sleep path — spec section 5 anti-goal and the alarm is a hard gate — cost if wrong: a slack face misread as happy with open jaw delays the alarm up to 5s.
Task 2: minor (deferred): no boundary tests at yawn_jaw_at or smile_at; wide low-smile laugh can read as yawning; empty-scores dict with unknown keys gives angry 0; short matrix raises IndexError (Task 5 must catch)
Task 2: fix round 1/5 pending re-review (commits 748cc2e..3bc5b73)
Task 2: fix round 1/5 (1 addressed, 0 open; commits 748cc2e..3bc5b73)
Task 2: minor (deferred): happy-read drooping face with jaw over 0.25 can delay or hide alarm; check on camera. CALM known limit is a silent miss.
Task 2: complete (commits 9f2c147..3bc5b73, review clean)
Ruling: A lettered track stores a new signature only if it matches its own letter's stored signatures at identity_match_at — stops strangers and track swaps merging people — cost if wrong: a person's stored views refresh less after a big appearance change.
Ruling: Pending-letter timeout applies to visible tracks only, restarting when a letter is stripped — fixes phantom people — cost if wrong: none expected.
Task 3: minor (deferred): "clearly matches" has no margin; a look-alike can take the letter of someone dropped as a fifth face or missed for a frame; Task 6 must call sync before add_signature and pass tracker.active_ids()
Task 3: fix round 1/5 (2 addressed, 0 open; commits a5fc23f..0c867c4)
Task 3: complete (commits 3bc5b73..0c867c4, review clean)
Ruling: SnapPolicy gets drop(letter); caller updates every present person including hidden grace people, and drops on letter release — keeps "continuous" honest with the Review Focus 5 rule — cost if wrong: a person returning after release while still red gets a new snap, which is a new event anyway.
Task 4: minor (deferred): log wording strings not constants; save_snap catches OSError and cv2.error only; letter-position expression is hard to read; render_snap test doesn't check the letter; two manual snaps in one second overwrite each other
Task 4: fix round 1/5 (drop added, docstring claim was false; commits fc1dc4b..0da4dd8)
Task 4: fix round 2/5 (docstring added, comment only, checked directly by controller; commits 0da4dd8..baaf970)
Task 4: complete (commits 0c867c4..baaf970, review clean)
Ruling: EmotionModel.scores takes (frame_bgr, box) instead of a crop, so padding uses real frame pixels — the brief's crop-only signature could not do what it asked — cost if wrong: none, Task 6 passes the frame and box.
Task 5: minor (deferred): no discriminating transpose test; self-match test weak; corrupt model files give raw errors; non-square crop stretched to 224
Task 5: fix round 1/5 (1 addressed, 0 open; commits 7c077d0..f25daf5)
Task 5: complete (commits baaf970..f25daf5, review clean)
Ruling: Overrule SnapPolicy contract: a snap fires only while the person is visible; hidden updates keep the event open but can't fire — spec says the snap is proof showing them — cost if wrong: someone who goes red and never returns gets no snap, though the log still records the mark.
Ruling: Pipeline.close writes alarm stopped if the alarm is on — spec says the log records start and stop — cost if wrong: none.
Task 6: minor (deferred): person returning awake in the 2.0-2.5s gap shows sleeping about 0.7s; all-bad frames repeat last result; tracebacks at frame rate if detector stays broken; log sends whole panel each frame, Task 7 must replace not append; one tautological test
Task 6: fix round 1/5 (3 addressed, 0 open; commits 27ff252..59d7fb4)
Task 6: complete (commits f25daf5..59d7fb4, review clean)
Ruling: Add websockets to requirements — real uvicorn can't open /ws without it — cost if wrong: none.
Ruling: /ws rejects connections whose Origin isn't the app's own 127.0.0.1 or localhost page — any website in the same browser could otherwise connect — cost if wrong: opening the app by another host name is refused.
Ruling: Replace httpx with httpx2 2.16.0 for test client — Starlette warned on httpx and test output must be clean — cost if wrong: one dev dependency swap.
Ruling: Page watchdog ends the session and silences the alarm after 5s with no reply, and the server releases the session on a hung call — alarm must not stay on for someone awake — cost if wrong: a very slow frame ends the session.
Ruling: Page reads tunable numbers from GET /config instead of copies — config constraint, and snap and screen must match — cost if wrong: one extra local request.
Task 7: minor (deferred): failed alarm start only logged, user not told; fatal vs non-fatal errors inferred; port 80 Origin omitted; no real-uvicorn abrupt-drop test
Task 7: fix round 1/5 (3 addressed, 0 open; commits df58c7d..04d7c8b)
Task 7: minor (deferred): overflow message hardcodes 4; model load has no timeout; watchdog armed after send so a lost toBlob isn't caught; frame over 5s ends the session; queued snapshot at close can be lost; box colours duplicated; stuck thread can't be stopped
Ruling: Accept frame_finish_grace_ms 1000 (added by the implementer) — stops close overlapping a healthy frame on reload — cost if wrong: release delayed up to 1s on a stall.
Task 7: complete (commits 59d7fb4..04d7c8b, review clean)
Task 8: minor (deferred): README wording nits (port message quoted, redundant activate)
Task 8: complete (commits 04d7c8b..0c9bc65, review clean; on-camera pass left to owner per ruling)
Final review: With fixes. 3 Critical, 4 Important. One fix dispatch.
Ruling: Each letter keeps permanent anchor signatures plus rolling slots, and frontal also requires head near level and eyes open — dozing wiped the store and the owner's priority is keeping a letter — cost if wrong: a poor first view anchors weakly.
Ruling: A new track stays Identifying until a frontal view decides or identity_pending_ms runs out — stops one turned glance sticking a phantom letter — cost if wrong: up to 2s of Identifying for someone staying turned.
Ruling: After 2 consecutive frontal checks clearly failing its own letter, a track loses the letter and re-identifies — narrow exception to keep-by-position, so a stranger can't wear someone's letter — cost if wrong: a big appearance change costs a letter.
Ruling: Reverse the snap-only-while-visible cost: keep the last visible frame during a red run and save it at grace end if visible red reached min_red_ms — every red mark should have proof — cost if wrong: a snap may show a frame from up to 2s before.
Ruling: Accept the I3 change: drop counts the last interval, so a person lost around 1s gets a snap; someone red under that gets a log line and no snap — spec's flicker rule wins — cost if wrong: a very short red mark has no proof.
Ruling: Accept the I1 limit: below about 5 fps every frame counts as a gap, so sleep never shows; measured rate is 14.5 and the floor is 8; stated in README — cost if wrong: on a much slower machine the alarm fails silently.
Final: fix wave re-reviewed (all findings addressed; minors parked for the owner).
Final: parked — a letter given during a head-down return can stick until they leave; a stranger with a steady tilted pitch is never checked; README overstates recovery on these — Ruling: no second fix wave, surface to owner — cost if wrong: wrong letter for one visit.
Owner on-camera: sleep and sound confirmed working on owner and a colleague (03b2cbd). Far faces not detected.
Ruling: A red person needs letter_release_ms of far views before reading far — near sleeper's alarm must not stop or pulse — cost if wrong: alarm runs 2s after someone really moves away.
Ruling: Replace the I1 limit: separate frame_gap_limit_ms 500, plus a page warning under 6 fps — group sleep must not fail silently; old ruling rested on 14.5 fps — cost if wrong: gaps under 0.5s count as closed time.
Ruling: Far tracks don't hold a letter against a near frontal match — a wall picture must not cost a person their letter — cost if wrong: a real far person may briefly lose a letter to a look-alike.
Ruling: Non-frontal checks fail only on a clearly low score, under identity_check_fail_at 0.2, and the nod rule starts at frontal_pitch_delta_deg — protects a drowsy real person — cost if wrong: a stranger who scores 0.2-0.36 while tilted keeps a letter longer.
Far work: fix round 1 re-reviewed, all addressed (0689a41..6d7e9ac). Minors deferred: flickering far views can keep a red person's alarm on past 2s; 2.5s closure plus a short loss can flash sleeping; rare drowsy under 3 fps.
Ruling: Accept strongDip-style extras: far_max_width_px 100, far_same_face_overlap, far_every_frames 2 — cost if wrong: config tuning.
