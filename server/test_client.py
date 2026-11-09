"""Offline check of demo_server.py: replays a video as if it were the phone.

  python test_client.py --video ~/Ouroboros/offload/input.mp4 --out /tmp/x

Prints per-mode latency and, for ouroboros, compute time vs number of
recomputed patches across thresholds (what the phone's predictor fits).
Writes a few frames with the returned boxes drawn on the *sent* frame, to
verify the canvas->frame box mapping.
"""
import argparse, json, os, statistics, time, urllib.request
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:65432")
ap.add_argument("--video", required=True)
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--out", default=None)
ap.add_argument("--thres", default="20")
args = ap.parse_args()

cap = cv2.VideoCapture(args.video)
frames = []
while len(frames) < args.frames:
    ok, f = cap.read()
    if not ok:
        break
    h, w = f.shape[:2]
    frames.append(cv2.resize(f, (args.width, int(h * args.width / w))))


def post(path, data=b"", method="POST"):
    req = urllib.request.Request(args.url + path, data=data if method == "POST" else None, method=method,
                                 headers={"Content-Type": "image/jpeg"})
    return json.loads(urllib.request.urlopen(req).read())


def run(mode, thres):
    post("/v1/reset")
    rows = []
    for i, f in enumerate(frames):
        jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        t = time.perf_counter()
        r = post(f"/v1/infer?mode={mode}&thres={thres}", jpg)
        r["e2e"] = (time.perf_counter() - t) * 1000
        rows.append(r)
        if args.out and i in (0, 5, 25, len(frames) - 1):
            vis = f.copy(); H, W = vis.shape[:2]
            for (x0, y0, x1, y1), n, sc in zip(r["boxes"], r["names"], r["scores"]):
                cv2.rectangle(vis, (int(x0 * W), int(y0 * H)), (int(x1 * W), int(y1 * H)), (0, 255, 0), 2)
                cv2.putText(vis, f"{n} {sc:.2f}", (int(x0 * W), int(y0 * H) + 15), 0, 0.5, (0, 255, 0), 2)
            os.makedirs(args.out, exist_ok=True)
            cv2.imwrite(f"{args.out}/{mode}_t{thres}_{i:03d}.jpg", vis)
    return rows


def summary(tag, rows):
    w = rows[3:]
    c = [r["timing"]["compute"] for r in w]
    nr = [r for r in w if not r["refresh"]]
    print(f"{tag:18s} compute mean {statistics.mean(c):6.1f}ms  e2e {statistics.mean(r['e2e'] for r in w):6.1f}ms"
          + (f" | non-refresh: dirty {statistics.mean(r['n_dirty'] for r in nr):6.0f}/4096,"
             f" compute {statistics.mean(r['timing']['compute'] for r in nr):6.1f}ms"
             f" (reg {statistics.mean(r['timing']['register'] for r in nr):4.1f}"
             f" dirty {statistics.mean(r['timing']['dirty'] for r in nr):4.1f}"
             f" model {statistics.mean(r['timing']['model'] for r in nr):5.1f})" if nr else ""))


summary("naive", run("naive", 20))
pts = []
for th in [int(x) for x in args.thres.split(",")]:
    rows = run("ouroboros", th)
    summary(f"ouroboros thr={th}", rows)
    pts += [(r["n_dirty"], r["timing"]["compute"], r["timing"]["model"]) for r in rows[3:]]

n = np.array([p[0] for p in pts], float); c = np.array([p[1] for p in pts]); m = np.array([p[2] for p in pts])
A = np.vstack([n, np.ones_like(n)]).T
(bc, ac), *_ = np.linalg.lstsq(A, c, rcond=None)
(bm, am), *_ = np.linalg.lstsq(A, m, rcond=None)
resid = c - (A @ [bc, ac])
print(f"\nfit compute_ms = {ac:.1f} + {bc*1000:.3f}e-3 * n_dirty   (resid std {resid.std():.1f}ms, n={len(n)})")
print(f"fit model_ms   = {am:.1f} + {bm*1000:.3f}e-3 * n_dirty")
for lo, hi in [(0, 500), (500, 1500), (1500, 3000), (3000, 4097)]:
    sel = (n >= lo) & (n < hi)
    if sel.any():
        print(f"  n_dirty {lo:4d}-{hi:4d}: {sel.sum():3d} pts, compute {c[sel].mean():6.1f}ms model {m[sel].mean():6.1f}ms")
