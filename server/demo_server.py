"""
demo_server.py -- inference + dashboard server for the phone demo (phone/).

Main scenario: three phones at once, one per mode, all against this one GPU.

  port 65433  NAIVE      POST /v1/infer  every frame padded to 1024x1024 and fully
                                         inferred, independently of every other frame
  port 65434  OUROBOROS  POST /v1/infer  ORB registration against the anchor canvas,
                                         toroidal shift of anchor image + cached features,
                                         dirtiness map (?thres=), partial recompute of
                                         dirty 16x16 patches only, GOP refresh
  port 65435  ON-DEVICE  (no inference)  the on-device phone fetches its ONNX model and
                                         sends dashboard reports here
  port 65432  DASHBOARD  GET /           live dashboard for any browser

Every port serves the same routes (dashboard, reports, clock probe); the port a
request arrives on decides which mode /v1/infer runs. Each mode has its own
session state (one device per mode). CPU-side work (JPEG decode, ORB
registration, dirtiness map, box mapping) runs in parallel per request; only
the ViT forward pass is serialized on the GPU, and the time a request waits
for the GPU is reported separately (timing.gpu_wait) -- that is what the
other tenants cost it.

Response: detections already mapped back into the *sent frame's* normalized
coordinates, the number of recomputed patches, the recompute map itself
(64x64 bits, so the phone's mirror can follow the server's anchor exactly),
and a per-stage timing breakdown.

Runs against an Ouroboros checkout (OUROBOROS_ROOT, else ../third_party/Ouroboros,
../../Ouroboros or ~/Ouroboros) and its venv.
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
_root_candidates = [os.environ.get("OUROBOROS_ROOT"), HERE.parent / "third_party" / "Ouroboros",
                    HERE.parent.parent / "Ouroboros", Path.home() / "Ouroboros"]
OUROBOROS_ROOT = next(Path(p) for p in _root_candidates if p and (Path(p) / "offload" / "preprocessing.py").exists())
OFFLOAD_DIR = OUROBOROS_ROOT / "offload"
sys.path.insert(0, str(OFFLOAD_DIR))

import torch  # noqa: E402
from aiohttp import web  # noqa: E402

sys.path.insert(0, str(HERE))
from coco import COCO  # noqa: E402
from vis import WorldMosaic, canvas_footprint  # noqa: E402
from preprocessing import (  # noqa: E402
    shift_anchor_features, apply_affine_and_pad, get_padded_image,
    estimate_affine_in_padded_anchor_fast, create_dirtiness_map,
)

INPUT_SIZE = (1024, 1024)
BLOCK = 16
GRID = INPUT_SIZE[0] // BLOCK          # 64
N_PATCHES = GRID * GRID                 # 4096
PAD_COLOR = [0, 0, 0]
RESULT_THRES = 0.5
MAX_DIM = int(min(INPUT_SIZE) * 0.8)    # 819, same convention as webrtc_server.py
MODES = ("naive", "ouroboros", "ondevice")
REG_RNG_SEED = 0x0B0B   # must match OuroborosMirror.REG_RNG_SEED


model = None
device = "cuda"
gop = 20
registration_mode = "rigid"
port_mode = {}                       # local port -> mode


class GpuScheduler:
    """The one shared resource every mode contends for: a non-preemptive, work-conserving
    GPU queue with a switchable policy.

      fifo  first come, first served. Two always-ready phones then simply alternate, so
            both run at 1/(sum of their job times) -- GPU time Ouroboros saves goes to the other.
      fair  serve the mode that has used the least GPU time so far (per-mode virtual time =
            cumulative GPU ms; a mode returning from idle is caught up to the busy modes'
            minimum so it cannot bank credit). Plain work-conserving priority is not enough
            with stop-and-wait phones: the light tenant is always in its network/encode gap
            when the GPU frees, so the heavy one grabs it every time. So the scheduler is
            *anticipatory*: if a mode that is under its share is due back within MAX_HOLD_MS
            (from its measured request cycle), the GPU is held idle for it briefly.
    """
    POLICIES = ("fifo", "fair")
    MAX_HOLD_MS = 150.0
    MAX_LAG_MS = 300.0     # no mode may fall more than this much GPU time behind: no banked credit

    def __init__(self, policy="fifo"):
        self.policy = policy
        self.cv = threading.Condition()
        self.busy = False
        self.seq = 0
        self.waiting = []                 # [seq, mode]
        self.vtime = {}                   # mode -> GPU ms consumed (virtual time)
        self.last_active = {}
        self.last_release = {}            # mode -> wall s
        self.gap = {}                     # mode -> EWMA s from its release to its next acquire
        self.gap_dev = {}                 # mode -> EWMA |deviation| of that gap (Wi-Fi jitter)
        self.held_ms = 0.0                # GPU idle while a job was waiting (anticipatory hold)
        self._acct_t = time.time()

    def _catch_up(self, mode, now):
        others = [v for m, v in self.vtime.items() if m != mode and now - self.last_active.get(m, 0) < 2.0]
        if others:
            self.vtime[mode] = max(self.vtime.get(mode, 0.0), min(others))
        else:
            self.vtime.setdefault(mode, 0.0)

    def _pick(self):
        if not self.waiting:
            return None
        if self.policy == "fair":
            return min(self.waiting, key=lambda w: (self.vtime.get(w[1], 0.0), w[0]))
        return min(self.waiting, key=lambda w: w[0])

    def _hold_for(self, mode, now):
        """Seconds to keep the GPU idle for a cheaper tenant that is about to come back."""
        if self.policy != "fair":
            return 0.0
        waiting_modes = {w[1] for w in self.waiting}
        best = 0.0
        for m, rel in self.last_release.items():
            if m == mode or m in waiting_modes or m not in self.gap:
                continue
            if self.vtime.get(m, 0.0) >= self.vtime.get(mode, 0.0):
                continue                                   # not under its share
            due = rel + self.gap[m]
            slack = self.slack(m)
            if now - rel > 2.0 or now > due + slack:
                continue                                   # quiet, or overdue: stop waiting for it
            if due - now <= self.MAX_HOLD_MS / 1000:
                best = max(best, due + slack - now)
        return best

    def _account(self):
        """Call under cv at every state change: accrue time the GPU sat idle with a job waiting."""
        now = time.time()
        if not self.busy and self.waiting:
            self.held_ms += (now - self._acct_t) * 1000
        self._acct_t = now

    def slack(self, mode):
        """How long past its expected arrival a tenant is still waited for: 2x its gap jitter."""
        return min(0.050, max(0.010, 2 * self.gap_dev.get(mode, 0.005)))

    def acquire(self, mode):
        with self.cv:
            now = time.time()
            if now - self.last_active.get(mode, 0) > 2.0:
                self._catch_up(mode, now)
            if mode in self.last_release and now - self.last_release[mode] < 2.0:
                g = now - self.last_release[mode]
                if mode not in self.gap:
                    self.gap[mode], self.gap_dev[mode] = g, 0.005
                else:
                    self.gap_dev[mode] += 0.2 * (abs(g - self.gap[mode]) - self.gap_dev[mode])
                    self.gap[mode] += 0.3 * (g - self.gap[mode])
            self.last_active[mode] = now
            self.seq += 1
            me = [self.seq, mode]
            self._account()
            self.waiting.append(me)
            while True:
                if self.busy or self._pick() is not me:
                    self.cv.wait(); continue
                hold = self._hold_for(mode, time.time())
                if hold <= 0:
                    break
                self.cv.wait(hold)
            self._account()
            self.waiting.remove(me)
            self.busy = True

    def release(self, mode, gpu_ms):
        with self.cv:
            self._account()
            self.vtime[mode] = self.vtime.get(mode, 0.0) + gpu_ms
            self.last_active[mode] = self.last_release[mode] = time.time()
            top = max(self.vtime.values())
            for m in self.vtime:                      # bounded fairness window
                self.vtime[m] = max(self.vtime[m], top - self.MAX_LAG_MS)
            self.busy = False
            self.cv.notify_all()

    def ahead_of(self, other, mode, other_extra_ms=0.0):
        """Would a waiting `other` job be served before a waiting `mode` job?
        other_extra_ms: GPU time `other` will have added by then (its job running now)."""
        if self.policy == "fifo":
            return True                   # it got there first
        with self.cv:
            vo = self.vtime.get(other, 0.0) + other_extra_ms
            vm = self.vtime.get(mode, 0.0)
            if time.time() - self.last_active.get(mode, 0) > 2.0:
                act = [v for m, v in self.vtime.items() if m != mode]
                vm = max(vm, min(act)) if act else vm
        return vo <= vm

    def set_policy(self, policy):
        with self.cv:
            self.policy = policy
            top = max(self.vtime.values(), default=0.0)
            self.vtime = {m: top for m in self.vtime}   # start the new policy from a clean slate
            self.cv.notify_all()


gpu_sched = GpuScheduler()


class Session:
    def __init__(self):
        self.anchor_image = None
        self.anchor_features = None
        self.frame_idx = 0
        self.world = [0, 0]           # world position of the canvas on the torus (cumulative roll)
        self.lock = threading.Lock()  # one device per mode, but never let two requests interleave


sessions = {"naive": Session(), "ouroboros": Session()}


class GpuStats:
    """Sliding 5 s window of GPU occupancy / waits per mode, for the dashboard."""
    WINDOW = 5.0

    def __init__(self):
        self.ev = deque()            # (t_end, mode, busy_ms, wait_ms, compute_ms)
        self.lock = threading.Lock()

    def add(self, mode, busy_ms, wait_ms, compute_ms):
        with self.lock:
            self.ev.append((time.time(), mode, busy_ms, wait_ms, compute_ms))

    def snapshot(self):
        now = time.time()
        with self.lock:
            while self.ev and now - self.ev[0][0] > self.WINDOW:
                self.ev.popleft()
            ev = list(self.ev)
        out = {}
        for m in ("naive", "ouroboros"):
            xs = [e for e in ev if e[1] == m]
            out[m] = {
                "rps": len(xs) / self.WINDOW,
                "gpu_share": sum(e[2] for e in xs) / (self.WINDOW * 1000),
                "gpu_ms": float(np.mean([e[2] for e in xs])) if xs else None,
                "wait_ms": float(np.mean([e[3] for e in xs])) if xs else None,
                "compute_ms": float(np.mean([e[4] for e in xs])) if xs else None,
            }
        out["gpu_util"] = sum(e[2] for e in ev) / (self.WINDOW * 1000)
        return out


gpu_stats = GpuStats()


class GpuState:
    """Who holds the GPU right now, who is queued, and when each other tenant is due back.

    Exposed via GET /v1/gpu so a phone can fold the *other* tenants' GPU work into its
    latency prediction -- the one input it cannot compute itself. Besides the job that
    is running/queued now, it reports each tenant's request cycle (gap between the end of
    one of its GPU jobs and the arrival of its next one), so the phone can also account
    for a job that will grab the GPU before its own request gets there.
    Expected ViT time per job: a per-mode least-squares line in n_dirty.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.current = None              # (mode, t_start, expected_ms)
        self.waiting = {}                # token -> (mode, expected_ms)
        self.obs = {m: deque(maxlen=60) for m in ("naive", "ouroboros")}
        self.last_end = {}               # mode -> wall ms
        self.last_n = {}                 # mode -> n_dirty of its last job
        self.gap = {}                    # mode -> EWMA ms from job end to its next arrival at the GPU queue

    def expect(self, mode, n):
        with self.lock:
            xs = list(self.obs[mode])
        if len(xs) < 3:
            return 90.0
        a = np.array(xs)
        if a[:, 0].std() < 1:
            return float(a[:, 1].mean())
        b, c = np.polyfit(a[:, 0], a[:, 1], 1)
        return float(max(5.0, b * n + c))

    def queued(self, token, mode, expected):
        now = time.time() * 1000
        with self.lock:
            self.waiting[token] = (mode, expected)
            if mode in self.last_end:
                g = now - self.last_end[mode]
                if g < 5000:
                    self.gap[mode] = g if mode not in self.gap else self.gap[mode] + 0.2 * (g - self.gap[mode])

    def started(self, token, mode, expected):
        with self.lock:
            self.waiting.pop(token, None)
            self.current = (mode, time.time() * 1000, expected)

    def finished(self, mode, n, model_ms):
        with self.lock:
            self.current = None
            self.obs[mode].append((n, model_ms))
            self.last_end[mode] = time.time() * 1000
            self.last_n[mode] = n

    def others_ms(self, mode):
        """GPU work owed to other modes as of now, plus when each is next due back."""
        now = time.time() * 1000
        with self.lock:
            cur = self.current
            waiting = list(self.waiting.values())
            last_end = dict(self.last_end); gap = dict(self.gap); last_n = dict(self.last_n)
        rem = max(0.0, cur[1] + cur[2] - now) if cur and cur[0] != mode else 0.0
        ahead = {m: gpu_sched.ahead_of(m, mode) for m in ("naive", "ouroboros") if m != mode}
        q = sum(e for m, e in waiting if m != mode and ahead.get(m, True))
        tenants = []
        for m in ("naive", "ouroboros"):
            if m == mode or m not in last_end and not (cur and cur[0] == m):
                continue
            if now - last_end.get(m, now) > 3000 and not (cur and cur[0] == m):
                continue                                   # gone quiet: not a tenant right now
            dur = self.expect(m, last_n.get(m, N_PATCHES))
            g = gap.get(m)
            if cur and cur[0] == m:
                r = max(0.0, cur[1] + cur[2] - now)
                # its *next* job competes with its virtual time after this one has been charged
                tenants.append({"mode": m, "state": "running", "remaining": r, "dur": dur, "ahead": ahead[m],
                                "ahead_next": gpu_sched.ahead_of(m, mode, other_extra_ms=cur[2]),
                                "next_in": None if g is None else r + g})
            elif any(wm == m for wm, _ in waiting):
                tenants.append({"mode": m, "state": "queued", "remaining": 0.0, "dur": dur, "ahead": ahead[m],
                                "ahead_next": ahead[m], "next_in": None})
            else:
                tenants.append({"mode": m, "state": "idle", "remaining": 0.0, "dur": dur, "ahead": ahead[m],
                                "ahead_next": ahead[m],
                                "next_in": None if g is None else max(0.0, last_end[m] + g - now)})
        return {"t": now, "busy_remaining": rem, "queued": q, "others": rem + q, "sched": gpu_sched.policy,
                "busy_mode": cur[0] if cur else None, "tenants": tenants}


gpu_state = GpuState()


def _sync():
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _map_boxes_to_frame(boxes, frame_wh, scale, refresh, offset, affine):
    """Canvas-space boxes -> normalized [0,1] boxes in the client's sent frame.

    refresh: canvas = centre-padded resized frame (pure translation by offset).
    otherwise: canvas = affine(resized frame), so invert the 2x3 affine and
    take the axis-aligned hull of the 4 mapped corners (rigid registration
    can rotate a box slightly).
    """
    W, H = frame_wh
    out = []
    if len(boxes) == 0:
        return out
    if refresh:
        inv = np.array([[1, 0, -offset[0]], [0, 1, -offset[1]]], dtype=np.float64)
    else:
        A = np.vstack([affine, [0, 0, 1]]).astype(np.float64)
        inv = np.linalg.inv(A)[:2]
    for x0, y0, x1, y1 in boxes:
        pts = np.array([[x0, y0, 1], [x1, y0, 1], [x0, y1, 1], [x1, y1, 1]], dtype=np.float64)
        m = pts @ inv.T / scale
        bx0, by0 = m[:, 0].min() / W, m[:, 1].min() / H
        bx1, by1 = m[:, 0].max() / W, m[:, 1].max() / H
        out.append([float(np.clip(v, 0, 1)) for v in (bx0, by0, bx1, by1)])
    return out


def process(body, mode, thres, reset):
    """Whole request on a worker thread: CPU stages in parallel, GPU stage serialized."""
    s = sessions[mode]
    with s.lock:
        if reset:
            s.__init__()
        tm = {}
        t = time.perf_counter()
        frame_bgr = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise ValueError(f"undecodable image ({len(body)} B)")
        tm["decode"] = (time.perf_counter() - t) * 1000; t = time.perf_counter()

        H0, W0 = frame_bgr.shape[:2]
        scale = 1.0
        if max(H0, W0) > MAX_DIM:
            scale = MAX_DIM / max(H0, W0)
            frame_bgr = cv2.resize(frame_bgr, (int(W0 * scale), int(H0 * scale)), interpolation=cv2.INTER_LINEAR)
        h, w = frame_bgr.shape[:2]

        refresh = mode == "naive" or s.anchor_image is None or s.frame_idx % gop == 0
        affine, shift, dmap = None, (0, 0), None
        reason = "naive" if mode == "naive" else ("first" if s.anchor_image is None else "gop")
        tm["register"] = tm["dirty"] = 0.0

        if not refresh:
            try:
                # LMEDS samples from OpenCV's thread-local RNG; seed it so the result does not
                # depend on which worker thread got the request -- and so the phone's mirror,
                # seeded identically, reproduces it bit-for-bit
                cv2.setRNGSeed(REG_RNG_SEED)
                affine = estimate_affine_in_padded_anchor_fast(s.anchor_image, frame_bgr, mode=registration_mode)
            except cv2.error:
                # too few ORB keypoints/matches (dark or textureless frame): the upstream
                # helper raises instead of returning None -- treat it as a failed registration.
                affine = None
            if affine is None:
                refresh, reason = True, "reg_fail"
            else:
                target, shift, affine = apply_affine_and_pad(frame_bgr, affine, block_size=BLOCK, filling_color=PAD_COLOR)
                tm["register"] = (time.perf_counter() - t) * 1000; t = time.perf_counter()
                s.anchor_image = np.roll(s.anchor_image, shift=(-shift[0], -shift[1]), axis=(1, 0))
                dmap = create_dirtiness_map(s.anchor_image, target, dirty_thres=thres)
                if np.linalg.norm(affine[:2, :2]) < 0.95:
                    refresh, reason = True, "scale"
                tm["dirty"] = (time.perf_counter() - t) * 1000; t = time.perf_counter()

        offset = ((INPUT_SIZE[1] - w) // 2, (INPUT_SIZE[0] - h) // 2)
        if refresh:
            target = get_padded_image(frame_bgr, INPUT_SIZE, filling_color=PAD_COLOR)
            dmap = torch.ones(1, GRID, GRID, 1, device=device)
            shift = (0, 0)
        # ---- shared global coordinate: where this canvas sits on the torus ----
        world_reset = False
        if mode == "ouroboros":
            if not refresh:
                s.world = [(s.world[0] + shift[0]) % INPUT_SIZE[0], (s.world[1] + shift[1]) % INPUT_SIZE[1]]
            else:
                # a GOP refresh re-centres the frame on a fresh canvas; register it once against
                # the outgoing anchor so the world map stays continuous across the refresh
                delta = None
                if reason == "gop" and s.anchor_image is not None:
                    t_w = time.perf_counter()
                    try:
                        cv2.setRNGSeed(REG_RNG_SEED)
                        A = estimate_affine_in_padded_anchor_fast(s.anchor_image, frame_bgr, mode=registration_mode)
                    except cv2.error:
                        A = None
                    if A is not None and np.linalg.norm(A[:2, :2]) >= 0.95:
                        c0 = np.array([INPUT_SIZE[0] / 2, INPUT_SIZE[1] / 2])
                        old = A[:, :2] @ (c0 - np.array(offset)) + A[:, 2]   # new-canvas centre in old canvas
                        delta = np.rint(old - c0).astype(int)
                    tm["register"] += (time.perf_counter() - t_w) * 1000; t = time.perf_counter()
                if delta is None:
                    s.world, world_reset = [0, 0], True
                else:
                    s.world = [int(s.world[0] + delta[0]) % INPUT_SIZE[0], int(s.world[1] + delta[1]) % INPUT_SIZE[1]]
        dmap_np = dmap[0, :, :, 0].cpu().numpy() > 0
        n_dirty = int(dmap_np.sum())
        # The phone's mirror advances its copy of the anchor with exactly this map
        # (64x64 bits, 512 B), so it can never drift from the server's state.
        dmap_bits = None if refresh else base64.b64encode(np.packbits(dmap_np).tobytes()).decode("ascii")

        # ---- GPU: the only serialized stage ----
        token = object(); expected = gpu_state.expect(mode, n_dirty)
        gpu_state.queued(token, mode, expected)
        t_wait = time.perf_counter()
        gpu_sched.acquire(mode)
        try:
            gpu_state.started(token, mode, expected)
            t_gpu = time.perf_counter()
            with torch.no_grad():
                if s.anchor_features is not None and (shift[0] or shift[1]):
                    s.anchor_features = shift_anchor_features(s.anchor_features, shift[0] // BLOCK, shift[1] // BLOCK)
                if refresh:
                    (boxes, labels, scores), cache = model.forward_contexted(target)
                else:
                    (boxes, labels, scores), cache = model.forward_contexted(
                        target, anchor_features=s.anchor_features, dirtiness_map=dmap)
                keep = scores > RESULT_THRES
                boxes, labels, scores = boxes[keep], labels[keep], scores[keep]
            _sync()
            t_gpu_end = time.perf_counter()
            gpu_state.finished(mode, n_dirty, (t_gpu_end - t_gpu) * 1000)
        finally:
            gpu_sched.release(mode, (time.perf_counter() - t_gpu) * 1000 if "t_gpu" in locals() else 0.0)
        tm["gpu_wait"] = (t_gpu - t_wait) * 1000
        tm["model"] = (t_gpu_end - t_gpu) * 1000
        t = time.perf_counter()

        if mode == "ouroboros":
            s.anchor_features = cache
            if s.anchor_image is None or refresh:
                s.anchor_image = target
            else:
                # Same result as upstream's (d*target + (1-d)*anchor).astype(uint8) -- d is exactly
                # 0/1 -- without float64 math over the whole 1024x1024x3 canvas (~8 ms -> <1 ms).
                d = cv2.resize(dmap_np.astype(np.uint8), INPUT_SIZE, interpolation=cv2.INTER_NEAREST).astype(bool)
                s.anchor_image = np.ascontiguousarray(s.anchor_image)
                np.copyto(s.anchor_image, target, where=d[:, :, None])

        cboxes = boxes.tolist()
        nboxes = _map_boxes_to_frame(cboxes, (W0, H0), scale, refresh, offset, affine)
        tm["post"] = (time.perf_counter() - t) * 1000
        # what the phone's predictor models as "compute": everything except waiting for the GPU
        tm["compute"] = sum(v for k, v in tm.items() if k != "gpu_wait")
        gpu_stats.add(mode, tm["model"], tm["gpu_wait"], tm["compute"])

        resp = {
            "idx": s.frame_idx, "mode": mode, "refresh": bool(refresh), "refresh_reason": reason if refresh else None,
            "thres": thres, "gop": gop, "n_dirty": n_dirty, "n_total": N_PATCHES,
            "frame_wh": [W0, H0], "shift": [int(shift[0]), int(shift[1])],
            "affine": affine.tolist() if affine is not None else None,
            "boxes": nboxes, "scores": [float(x) for x in scores.tolist()],
            "labels": [int(x) for x in labels.tolist()],
            "names": [COCO[int(x)] if 0 <= int(x) < len(COCO) else str(int(x)) for x in labels.tolist()],
            "timing": {k: round(v, 3) for k, v in tm.items()},
            "dmap": dmap_bits,
            "world": list(s.world) if mode == "ouroboros" else None, "world_reset": world_reset,
        }
        if mode == "ouroboros" and vis_cfg["world"]:
            world_view.submit(target, (h, w), None if refresh else affine, offset, dmap_np, list(s.world),
                              cboxes, resp["names"], refresh, world_reset)
        s.frame_idx += 1
        return resp


# ---------------------------------------------------------------- server-side global view
vis_cfg = {"world": True, "boost": True}   # --no-world-vis / --no-boost, or POST /v1/vis


class WorldView:
    """Renders the Ouroboros session's toroidal world mosaic off the request path
    (one worker, latest frame wins) and tells dashboards a new image is ready."""
    MAX_HZ = 10.0

    def __init__(self):
        self.mosaic = WorldMosaic()
        self.lock = threading.Lock(); self.busy = False; self.pending = None; self.need_reset = False
        self.jpeg, self.v, self.last = None, 0, 0.0
        self.loop = None

    def submit(self, *job):
        with self.lock:
            if job[-1]:
                self.need_reset = True        # never drop a discontinuity
            self.pending = job
            if self.busy:
                return
            self.busy = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            with self.lock:
                job, self.pending = self.pending, None
                reset, self.need_reset = self.need_reset, False
                if job is None:
                    self.busy = False; return
            target, hw, affine, offset, dmap_np, world, cboxes, names, refresh, _ = job
            foot = canvas_footprint(hw, affine, offset)
            img = self.mosaic.update(target, foot, dmap_np, world, cboxes, names, boost=vis_cfg["boost"],
                                     refresh=refresh, reset=reset)
            now = time.time()
            if now - self.last >= 1.0 / self.MAX_HZ:
                self.last = now
                self.jpeg = cv2.imencode(".jpg", cv2.resize(img, (720, 720), interpolation=cv2.INTER_AREA),
                                         [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
                self.v += 1
                if self.loop:
                    self.loop.call_soon_threadsafe(_broadcast, "world", {"v": self.v})

    def clear(self):
        with self.lock:
            self.need_reset = True


world_view = WorldView()


async def world_jpg(request):
    if world_view.jpeg is None:
        return web.Response(status=404)
    return web.Response(body=world_view.jpeg, content_type="image/jpeg", headers={"Cache-Control": "no-store"})


async def vis_config(request):
    """GET: server-side visualisation flags.  POST ?world=0|1&boost=0|1: change them live."""
    if request.method == "POST":
        for k in vis_cfg:
            if k in request.query:
                vis_cfg[k] = request.query[k] in ("1", "true", "on")
        _broadcast("vis", dict(vis_cfg))
    return web.json_response(dict(vis_cfg))


def _request_mode(request):
    sock = request.transport.get_extra_info("sockname") if request.transport else None
    return port_mode.get(sock[1] if sock else None)


async def infer(request):
    t_recv = time.time() * 1000
    body = await request.read()
    t_body = time.time() * 1000
    # the port decides; ?mode= only matters on the dashboard port (handy for test_client.py)
    mode = _request_mode(request)
    if mode in (None, "dashboard"):
        mode = request.query.get("mode", "ouroboros")
    if mode not in ("naive", "ouroboros"):
        return web.json_response({"error": f"no inference on the {mode} port"}, status=400)
    thres = int(request.query.get("thres", "20"))
    reset = request.query.get("reset") == "1"
    try:
        resp = await asyncio.get_running_loop().run_in_executor(None, process, body, mode, thres, reset)
    except Exception as e:  # keep serving; report to the phone
        import traceback; traceback.print_exc()
        sessions[mode].__init__()
        return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=500)
    resp["t_recv"] = t_recv          # server wall clock (ms) when headers arrived
    resp["t_body"] = t_body          # ... when the full upload had arrived
    resp["t_send"] = time.time() * 1000
    resp["upload_bytes"] = len(body)
    tmg = resp["timing"]
    print(f"[{mode:9s}] #{resp['idx']:5d} {len(body)/1024:5.1f}KB dirty={resp['n_dirty']:4d} "
          f"{'R' if resp['refresh'] else ' '} compute={tmg['compute']:6.1f} gpu={tmg['model']:5.1f} "
          f"wait={tmg['gpu_wait']:5.1f} reg={tmg['register']:4.1f}", flush=True)
    return web.json_response(resp)


async def sched_policy(request):
    """GET: current GPU scheduling policy.  POST ?policy=fifo|fair: switch it live."""
    if request.method == "POST":
        pol = request.query.get("policy", "")
        if pol not in GpuScheduler.POLICIES:
            return web.json_response({"error": f"policy must be one of {GpuScheduler.POLICIES}"}, status=400)
        gpu_sched.set_policy(pol)
        print(f"[sched] policy -> {pol}", flush=True)
        _broadcast("server", {**gpu_stats.snapshot(), "now": time.time() * 1000, "sched": pol})
    return web.json_response({"sched": gpu_sched.policy})


async def gpu_probe(request):
    mode = _request_mode(request)
    if mode in (None, "dashboard"):
        mode = request.query.get("mode", "ouroboros")
    return web.json_response(gpu_state.others_ms(mode))


async def time_probe(request):
    return web.json_response({"t": time.time() * 1000})


async def reset(request):
    mode = _request_mode(request)
    for m, s in sessions.items():
        if mode in (None, "dashboard", m):
            s.__init__()
    world_view.clear()
    return web.json_response({"ok": True})


# ---------------------------------------------------------------- live dashboard
#
# Every phone POSTs one small JSON report per displayed frame (plus, rate-limited,
# a downscaled JPEG of what it is showing). The server keeps a short history per
# mode and fans every report out to dashboard browsers over Server-Sent Events,
# together with a once-a-second GPU-sharing summary.

DASH_HTML = HERE / "dashboard.html"
dash_history = deque(maxlen=900)
dash_clients = set()
dash_frames = {m: {"jpeg": None, "v": 0} for m in MODES}


def _broadcast(event, data):
    msg = f"event: {event}\ndata: {json.dumps(data)}\n\n"
    for q in list(dash_clients):
        if q.qsize() < 100:
            q.put_nowait(msg)


async def report(request):
    rec = await request.json()
    mode = str(rec.get("mode", "")).replace("_", "")
    if mode not in MODES:
        return web.json_response({"error": f"bad mode {mode!r}"}, status=400)
    rec["mode"] = mode
    img = rec.pop("image", None)
    if img:
        f = dash_frames[mode]
        f["jpeg"] = base64.b64decode(img)
        f["v"] += 1
        rec["image_v"] = f["v"]
    rec["server_t"] = time.time() * 1000
    rec["client_ip"] = request.remote
    dash_history.append(rec)
    _broadcast("frame", rec)
    return web.json_response({"ok": True})


async def events(request):
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache",
                                       "X-Accel-Buffering": "no"})
    await resp.prepare(request)
    q = asyncio.Queue()
    dash_clients.add(q)
    try:
        await resp.write(("event: snapshot\ndata: " + json.dumps(
            {"history": list(dash_history), "now": time.time() * 1000, "vis": vis_cfg,
             "world_v": world_view.v}) + "\n\n").encode())
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=15)
            except asyncio.TimeoutError:
                msg = ": keepalive\n\n"
            await resp.write(msg.encode())
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        dash_clients.discard(q)
    return resp


async def server_stats_loop(app):
    async def loop():
        prev_held = gpu_sched.held_ms
        while True:
            await asyncio.sleep(1.0)
            held, prev_held = gpu_sched.held_ms - prev_held, gpu_sched.held_ms
            if dash_clients:
                snap = gpu_stats.snapshot(); snap["now"] = time.time() * 1000; snap["sched"] = gpu_sched.policy
                snap["held_frac"] = min(1.0, held / 1000)   # anticipatory idle in the last second
                _broadcast("server", snap)
    task = asyncio.create_task(loop())
    yield
    task.cancel()


async def frame_jpg(request):
    mode = request.query.get("mode", "ouroboros")
    f = dash_frames.get(mode)
    if not f or f["jpeg"] is None:
        return web.Response(status=404)
    return web.Response(body=f["jpeg"], content_type="image/jpeg", headers={"Cache-Control": "no-store"})


async def dash_reset(request):
    dash_history.clear()
    _broadcast("reset", {})
    return web.json_response({"ok": True})


async def dashboard(request):
    return web.FileResponse(DASH_HTML, headers={"Cache-Control": "no-cache"})


MODEL_DIR = HERE.parent / "build"   # tools/export_vitdet_onnx.py writes here


async def model_file(request):
    """Lets the on-device phone fetch the ~430MB ONNX once instead of shipping it in the APK."""
    name = request.match_info["name"]
    path = (MODEL_DIR / name).resolve()
    if path.parent != MODEL_DIR.resolve() or not path.is_file():
        return web.json_response({"error": f"no model {name!r}"}, status=404)
    return web.FileResponse(path)


async def ports_info(request):
    return web.json_response({m: p for p, m in port_mode.items()})


def warmup():
    dummy = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    with torch.no_grad():
        for _ in range(5):
            _, cache = model.forward_contexted(dummy)
        for frac in (0.05, 0.3, 0.7):   # also warm the partial path at a few densities
            dm = (torch.rand(1, GRID, GRID, 1, device=device) < frac).float()
            for _ in range(3):
                _, cache = model.forward_contexted(dummy, anchor_features=cache, dirtiness_map=dm)
    _sync()


def main():
    global model, device, gop, registration_mode
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--dashboard-port", type=int, default=65432)
    ap.add_argument("--naive-port", type=int, default=65433)
    ap.add_argument("--ouroboros-port", type=int, default=65434)
    ap.add_argument("--ondevice-port", type=int, default=65435)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gop", type=int, default=20)
    ap.add_argument("--registration", default="rigid", choices=["rigid", "affine"])
    ap.add_argument("--sched", default="fifo", choices=list(GpuScheduler.POLICIES),
                    help="GPU sharing between modes (switchable live from the dashboard / POST /v1/sched)")
    ap.add_argument("--no-world-vis", action="store_true",
                    help="do not render the Ouroboros session on the shared global (toroidal) canvas")
    ap.add_argument("--no-boost", action="store_true",
                    help="server-side views: do not tint recomputed patches green")
    args = ap.parse_args()
    vis_cfg.update(world=not args.no_world_vis, boost=not args.no_boost)
    device, gop, registration_mode = args.device, args.gop, args.registration
    gpu_sched.set_policy(args.sched)
    port_mode.update({args.dashboard_port: "dashboard", args.naive_port: "naive",
                      args.ouroboros_port: "ouroboros", args.ondevice_port: "ondevice"})

    os.chdir(OFFLOAD_DIR)  # model code resolves some paths relative to offload/
    from models import MaskedRCNN_ViT_B_FPN_Contexted
    model = MaskedRCNN_ViT_B_FPN_Contexted(device=device)
    model.load_weight(str(OFFLOAD_DIR / "models" / "model_final_61ccd1.pkl"))
    model.eval()
    warmup()

    app = web.Application(client_max_size=16 * 1024 * 1024)
    app.router.add_post("/v1/infer", infer)
    app.router.add_get("/v1/time", time_probe)
    app.router.add_get("/v1/gpu", gpu_probe)
    app.router.add_get("/v1/sched", sched_policy)
    app.router.add_post("/v1/sched", sched_policy)
    app.router.add_post("/v1/reset", reset)
    app.router.add_get("/v1/model/{name}", model_file)
    app.router.add_post("/v1/report", report)
    app.router.add_get("/v1/events", events)
    app.router.add_get("/v1/frame.jpg", frame_jpg)
    app.router.add_get("/v1/world.jpg", world_jpg)
    app.router.add_get("/v1/vis", vis_config)
    app.router.add_post("/v1/vis", vis_config)
    app.router.add_post("/v1/dashboard/reset", dash_reset)
    app.router.add_get("/v1/ports", ports_info)
    app.router.add_get("/", dashboard)
    app.cleanup_ctx.append(server_stats_loop)

    async def serve():
        world_view.loop = asyncio.get_running_loop()
        runner = web.AppRunner(app, shutdown_timeout=2.0)
        await runner.setup()
        for port in port_mode:
            await web.TCPSite(runner, args.host, port).start()
        print(f"ready: Ouroboros={OUROBOROS_ROOT} gop={gop} reg={registration_mode} sched={gpu_sched.policy} | "
              + " ".join(f"{m}:{p}" for p, m in port_mode.items()), flush=True)
        await asyncio.Event().wait()

    asyncio.run(serve())


if __name__ == "__main__":
    main()
