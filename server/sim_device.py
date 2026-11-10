"""Stand-in for a phone: replays a video in one mode, stop-and-wait, and reports
to the dashboard exactly like the app does. For rehearsing the three-device
scenario with fewer phones, or load-testing the server.

  python sim_device.py --mode naive     --host 127.0.0.1
  python sim_device.py --mode ouroboros --host 127.0.0.1 --thres 20
  python sim_device.py --mode ondevice  --host 127.0.0.1 --threads 4   # ONNX on this CPU

Ports default to the server's (naive 65433, ouroboros 65434, ondevice 65435).
"""
import argparse, base64, json, os, statistics, threading, time, urllib.request
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from coco import COCO
from vis import boost_recomputed

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["naive", "ouroboros", "ondevice"])
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=None)
ap.add_argument("--video", default=str(Path.home() / "Ouroboros" / "offload" / "input.mp4"))
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--thres", type=int, default=20)
ap.add_argument("--threads", type=int, default=4, help="ondevice: CPU threads for onnxruntime")
ap.add_argument("--onnx", default=str(Path(__file__).resolve().parent.parent / "build" / "vitdet_b.onnx"))
ap.add_argument("--name", default=None)
ap.add_argument("--no-boost", action="store_true",
                help="ouroboros: do not tint the recomputed patches green (on by default)")
ap.add_argument("--seconds", type=float, default=0, help="stop after this long (0 = forever)")
ap.add_argument("--net-ms", type=float, default=0,
                help="emulated one-way network delay added before the upload and after the reply, "
                     "so a sim on the server behaves like a phone on Wi-Fi (e.g. 15)")
args = ap.parse_args()
port = args.port or {"naive": 65433, "ouroboros": 65434, "ondevice": 65435}[args.mode]
base = f"http://{args.host}:{port}"
name = args.name or f"sim-{args.mode}"
COLORS = {"naive": (255, 196, 64), "ouroboros": (174, 240, 105), "ondevice": (82, 82, 255)}  # BGR


def frames():
    while True:
        cap = cv2.VideoCapture(args.video)
        while True:
            ok, f = cap.read()
            if not ok:
                break
            h, w = f.shape[:2]
            yield cv2.resize(f, (args.width, int(h * args.width / w)))


def http(path, data=None, ctype="image/jpeg"):
    req = urllib.request.Request(base + path, data=data, method="POST" if data is not None else "GET",
                                 headers={"Content-Type": ctype})
    return urllib.request.urlopen(req, timeout=60).read()


def draw(vis, dets):
    H, W = vis.shape[:2]; c = COLORS[args.mode]
    for d in dets:
        x0, y0, x1, y1 = d["box"]
        cv2.rectangle(vis, (int(x0 * W), int(y0 * H)), (int(x1 * W), int(y1 * H)), c, 2)
        cv2.putText(vis, f"{d['name']} {d['score']:.2f}", (int(x0 * W) + 3, int(y0 * H) + 16), 0, 0.5, c, 2)
    cv2.putText(vis, args.mode.upper() + " (sim)", (8, 24), 0, 0.7, c, 2)
    return vis


report_q = deque(maxlen=1); last_img = [0.0]


def reporter():
    while True:
        if not report_q:
            time.sleep(0.01); continue
        rec, vis = report_q.pop()
        if vis is not None:
            small = cv2.resize(vis, (360, int(vis.shape[0] * 360 / vis.shape[1])))
            rec["image"] = base64.b64encode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])[1]).decode()
        try:
            http("/v1/report", json.dumps(rec).encode(), "application/json")
        except Exception as e:
            print("report failed:", e)


threading.Thread(target=reporter, daemon=True).start()

sess = None
if args.mode == "ondevice":
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = args.threads
    t = time.time(); sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])
    load_ms = (time.time() - t) * 1000

hist = deque(maxlen=30); done = deque(maxlen=30); n = 0; first = True; t_start = time.time()
for f in frames():
    if args.seconds and time.time() - t_start > args.seconds:
        break
    t0 = time.perf_counter()
    rec = {"mode": args.mode, "device": name}
    if args.mode == "ondevice":
        H, W = f.shape[:2]; canvas = np.zeros((1024, 1024, 3), np.uint8); ox, oy = (1024 - W) // 2, (1024 - H) // 2
        canvas[oy:oy + H, ox:ox + W] = f
        t1 = time.perf_counter(); b, s, c = sess.run(None, {"image": canvas}); t2 = time.perf_counter()
        k = s > 0.5
        dets = [{"name": str(int(ci)), "score": float(si),
                 "box": [float((bb[0] - ox) / W), float((bb[1] - oy) / H), float((bb[2] - ox) / W), float((bb[3] - oy) / H)]}
                for bb, si, ci in zip(b[k], s[k], c[k])]
        for d in dets: d["name"] = COCO[int(d["name"])]
        rec["ondevice"] = {"prep": (t1 - t0) * 1000, "infer": (t2 - t1) * 1000, "load": load_ms}
    else:
        jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        enc = (time.perf_counter() - t0) * 1000
        ts = time.perf_counter()
        if args.net_ms: time.sleep(args.net_ms / 1000)
        r = json.loads(http(f"/v1/infer?thres={args.thres}" + ("&reset=1" if first else ""), jpg))
        if args.net_ms: time.sleep(args.net_ms / 1000)
        http_ms = (time.perf_counter() - ts) * 1000
        first = False
        if "error" in r:
            print("server error:", r["error"]); time.sleep(0.5); first = True; continue
        dets = [{"name": nm, "score": sc, "box": bx} for nm, sc, bx in zip(r["names"], r["scores"], r["boxes"])]
        srv = r["t_send"] - r["t_body"]; net = http_ms - srv
        rec["offload"] = {"encode": enc, "upload": net * 0.75, "server": srv, "download": net * 0.25, "draw": 0.0,
                          "compute": r["timing"]["compute"], "timing": r["timing"], "n_dirty": r["n_dirty"],
                          "refresh": r["refresh"], "refresh_reason": r["refresh_reason"], "thres": args.thres,
                          "kb": len(jpg) / 1024}
        rec["predict_on"] = False
        rec["boost"] = args.mode == "ouroboros" and not args.no_boost
    vis = f.copy()
    if rec.get("boost"):
        boost_recomputed(vis, r)
    vis = draw(vis, dets)
    e2e = (time.perf_counter() - t0) * 1000
    n += 1; hist.append(e2e); done.append(time.perf_counter())
    fps = (len(done) - 1) / (done[-1] - done[0]) if len(done) > 1 else None
    rec.update({"frame": n, "e2e": e2e, "fps": fps, "avg30": statistics.mean(hist), "dets": dets})
    now = time.time()
    report_q.append((rec, vis if now - last_img[0] > 0.25 else None))
    if now - last_img[0] > 0.25: last_img[0] = now
    if n % 20 == 0 or args.mode == "ondevice":
        print(f"[{name}] #{n} e2e {e2e:.0f} ms avg {statistics.mean(hist):.0f} ms", flush=True)
