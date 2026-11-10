# Phone demo — status (2026-11-10)

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
GPU: this container may only use **MIG-bb8cda48-86b2-549c-84c2-32299742a69b** (3g.48gb, 60 SMs, 46.5 GiB).
Always pass `CUDA_VISIBLE_DEVICES` explicitly to anything started in the background.
```bash
# server, detached in tmux (survives the launching shell); pins the MIG; prints the URLs
server/run_demo.sh            # restart = run again;  stop: server/run_demo.sh stop;  log: tmux attach -t ouroboros-demo
# or in the foreground:
cd server && CUDA_VISIBLE_DEVICES=MIG-bb8cda48-86b2-549c-84c2-32299742a69b ~/Ouroboros/.venv/bin/python demo_server.py
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

## On the MIG 3g.48gb slice (since 2026-11-10)
- ViT: full 152 ms (was 85 on the whole GH200); partial ≈ 54 ms + 25 µs/patch (was 46 + 9) → Ouroboros's
  relative saving is larger on the smaller slice.
- Pixel alone, Ouroboros + prediction: e2e 116 ms, **8.0 FPS**, prediction MAE 8 ms, patches exact 310/310.
- Three at once (Pixel Ouroboros, sim naive --net-ms 15, sim on-device): Ouroboros 162 ms / naive 209 ms e2e,
  GPU share 30% / 70% (util 100%); prediction MAE 6.0 ms, 97% within ±10%, 99% within ±20%, GPU-wait MAE 5 ms.
- App: mirror commit/reset moved to the mirror thread and the camera released as soon as the result is in,
  cutting per-frame dead time outside e2e from ~60 ms to ~6–9 ms.
- **GPU sharing policy, switchable live** (dashboard header buttons, `POST /v1/sched?policy=fifo|fair`,
  or `--sched`). Measured with the Pixel on Ouroboros+prediction, sim naive (--net-ms 15), sim on-device:

  | | FIFO | Fair-share |
  |---|---|---|
  | Ouroboros (Pixel) | 202 ms / 4.7 FPS | **132 ms / 6.1 FPS**, GPU wait median 0 |
  | Naive (sim) | 204 ms / 4.7 FPS | 444 ms / 2.3 FPS |
  | GPU share naive/ouro | 73% / 28% | 36% / 36% (+ ~30% anticipatory idle) |
  | Prediction MAE (±10% / ±20%) | 9.5 ms (94% / 99%) | 16.8 ms (67% / 89%) |
  | Patch count exact | 273/273 | 356/356 |

  FIFO: two always-ready clients just alternate on the GPU, so both get ≈1/(152+53 ms) and the GPU time
  Ouroboros saves is donated to naive. Fair: per-mode virtual GPU time, *anticipatory* — a plain
  work-conserving priority does nothing here (the light tenant is always in its network gap when the GPU
  frees), so the GPU is held idle briefly (≤150 ms, slack = 2×jitter of that tenant's request cycle) for a
  mode that is under its share. Lag capped at 300 ms of GPU time and reset on policy switch (otherwise
  credit banked under FIFO starved naive for seconds). Costs ~30% GPU idle.
- Phone predictor replays the server's scheduler: `/v1/gpu` returns, per other tenant, its running/queued
  job, its next expected arrival, and whether it would be served first (`ahead`, `ahead_next` = priority
  after its running job is charged).
- LMEDS RNG is seeded identically on both sides before registration (`REG_RNG_SEED`).
- Dashboard 3-column desktop layout verified in headless Chromium (no console errors).

## Visualisation options (2026-11-10)
- **Recomputed-patch green boost** (default ON; upstream's `img[...,1] = clip(img[...,1] + dmap*30, 0, 255)`).
  Drawn on the camera frame: each frame pixel looks its patch up through the server's affine
  (`server/vis.py recompute_mask`, `RecomputeVis.boostFrame`); verified pixel-exact against a direct
  per-pixel lookup (0 mismatches / 38 frames). A refresh (GOP) frame is all green.
  App: checkbox "재연산 패치 초록 표시" / `--ez boost false`; sim: `--no-boost`; server global view:
  `--no-boost` or dashboard button / `POST /v1/vis?boost=0`.
- **Shared global coordinate view** (wrap-around torus). Server returns `world` = cumulative roll of the
  canvas (mod 1024) and `world_reset`; on a GOP refresh it registers the new frame once against the outgoing
  anchor (~5 ms, refresh frames only) so the map stays continuous. Mosaic = each frame painted at
  canvas+world; outline/boxes drawn in canvas coords then rolled, so they wrap too.
  - Server: rendered off the request path (≤10 Hz) → `GET /v1/world.jpg`; dashboard Ouroboros lane
    "폰 화면 / 서버 global 좌표" switch. `--no-world-vis` / `POST /v1/vis?world=0` to disable.
  - App: checkbox "global 좌표 보기" / `--ez world true`; rendered on its own thread, not in e2e
    (Pixel: still 8.0 FPS, 96 ms).
  - Synthetic 14 px/frame pan: world step 14.0 ± 0.5 px per frame, continuous across every refresh, wraps.
- Finding: on the 40-frame bear clip (loops), upstream rigid registration sometimes returns garbage
  (e.g. 83° rotation after a GOP): `estimate_affine_in_padded_anchor_fast(mode="rigid")` normalises the
  LMEDS scale away, so a degenerate fit becomes a full-size rotation and the scale<0.95 refresh check never
  fires. Global view makes it visible. Possible fix (needs the phone mirror to match): reject when the raw
  LMEDS scale is far from 1 or rotation is large → reg_fail refresh.

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
- `server/run_demo.sh` (tmux) is written but its first launch from the Claude session was blocked by the
  permission classifier (persistent externally reachable service) — run it yourself or allow it.
- Only one physical phone tested; naive/on-device lanes were exercised with sim_device.py.
- Possible: send the phone's affine to skip server ORB (~10 ms) when prediction is on.
