# Phone demo — status (2026-11-09)

Three detection paths on a phone camera, all running the *same* ViTDet-B, plus a live dashboard.
Main scenario: three phones at once, one per mode, sharing one GPU server.

| Mode | Where | Port |
|---|---|---|
| On-device | ViTDet-B exported to ONNX, ONNX Runtime CPU on the phone | 65435 (model download + reports) |
| Naive offload | JPEG → server, every frame fully inferred | 65433 |
| Ouroboros | JPEG → server, registration + wrap-around + partial recompute | 65434 |
| Dashboard | any browser: `http://147.46.66.242:65432/` | 65432 |

## Layout
- `server/demo_server.py` — one process, one GPU model, four ports; per-mode sessions; CPU stages run in
  parallel, only the ViT forward is serialized (`timing.gpu_wait`); `/v1/gpu` probe (other tenants' GPU
  work + their request cycle); SSE dashboard feed; serves `build/vitdet_b.onnx`.
- `server/dashboard.html` — 3 lanes (live frame, e2e, FPS, breakdown), 60 s timeline, GPU share, prediction card.
- `server/sim_device.py` — stand-in phone (replays a video; `--net-ms` emulates Wi-Fi delay; `--mode ondevice` runs the ONNX on CPU).
- `server/test_client.py` — offline sweep (compute vs n_dirty).
- `tools/export_vitdet_onnx.py` — exports the server's ViTDet-B (box branch) to ONNX; verified vs torch (max box diff 6e-5).
- `phone/` — Android app (Kotlin, CameraX, ONNX Runtime 1.31.1, OpenCV 5.0.0).
  - `OuroborosMirror.kt` — phone-side port of the server's registration + dirtiness map; predicts the
    server's recomputed patch count in parallel with the upload; anchor advanced with the server's own
    affine/shift/dmap (from the response) so it never drifts.
  - `LatencyPredictor.kt` — online model: encode + network(KB) + GPU wait (probe + queue sim + residual) +
    pre + ViT(n) + post + other; refit every frame.
  - `DashboardReporter.kt` — fire-and-forget per-frame report to the server.

## How to run
```bash
# server (Ouroboros venv; Ouroboros checkout at ~/Ouroboros)
cd server && ~/Ouroboros/.venv/bin/python demo_server.py
# on-device model (434 MB, not in git):  ~/Ouroboros/.venv/bin/python tools/export_vitdet_onnx.py --out build/vitdet_b.onnx
#   (needs onnx onnxruntime onnxscript in the venv)
# app
cd phone && ./gradlew assembleDebug [-Pandroid.aapt2FromMavenOverride=$HOME/android-sdk/aapt2-wrapper/aapt2  # aarch64 host]
adb install -r -g app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.ouroboros.demo/.MainActivity --es mode ouroboros --ez predict true --ez autostart true --es server 147.46.66.242
# stand-in devices
cd server && python sim_device.py --mode naive --net-ms 15 ; python sim_device.py --mode ondevice --threads 4
```
App: enter only the host; the port follows the mode (`host:port` overrides).

## adb on this aarch64 container
The SDK's adb is x86_64; under qemu the adb *server* segfaults. Working setup: Ubuntu's arm64 `adb`
debs extracted (no sudo) into `~/.local/adb-arm64/root`, wrapper `~/.local/bin/adb` sets LD_LIBRARY_PATH
(libprotobuf32t64 had to be fetched from the ports pool directly — local apt index was stale).
Phone: Pixel 11 Pro XL, Android 17, `adb connect 147.46.118.129:25557` (key already authorized).

## Measured (Pixel 11 Pro XL on Wi-Fi, GH200 server)
Single device:
- Ouroboros 100 ms (10.0 FPS) vs Naive 132 ms (7.6 FPS); ViT 46 vs 85 ms; ~144/4096 patches recomputed.
- On-device ViTDet-B: 15–21 s per frame (~0.06 FPS), model load 1.2 s.
- Prediction: phone-predicted patch count == server's on 388/388 frames; e2e MAE 7 ms, 84% within ±10%.
  Prediction is ready ~+51 ms after send start (mirror ~44 ms) vs result at ~100 ms.

Three at once (Pixel = Ouroboros+prediction, sim naive with --net-ms 15, sim on-device):
- Ouroboros 103 ms vs Naive 140 ms; GPU share 31% vs 60% (util 91%); GPU wait median 0 vs 17.6 ms.
- Prediction: patch count exact 411/411; e2e MAE 14.8 ms, bias +0.3, 61% within ±10%, 86% within ±20%.
  Without the queue-aware GPU probe it was MAE 45 ms / 7% within ±10%.

## Findings worth knowing
- The create_dirtiness_map chain (>thr → blur15 → resize 1/16 linear → >0) equals "max of blurred
  residual over the 16×16 patch > thr" (verified on 1800 cases) — the phone uses a 64×64 max-pool.
- Upstream `estimate_affine_in_padded_anchor_fast` raises on <2 ORB matches (dark frames); demo server
  treats it as reg_fail → refresh.
- Upstream anchor composite in float64 cost ~8 ms; `np.copyto(where=mask)` gives identical output.
- Model declares RGB pixel_mean but offload code feeds cv2 BGR; no difference on the bear clip, phone
  matches the server (BGR). Worth checking on harder content.
- GH200: ViT ≈ 46 ms + 9 µs/patch, so threshold barely moves latency (fixed cost dominates).

## Open / next
- Dashboard was verified on a phone browser (stacked layout); desktop 3-column layout not yet screenshot-checked
  (headless Chromium was being installed when work stopped).
- Server runs in the foreground of a session; for the demo, run it under tmux/systemd.
- Only one physical phone tested; naive/on-device lanes were exercised with sim_device.py.
- Possible: send the phone's affine to skip server ORB (~10 ms) when prediction is on.
