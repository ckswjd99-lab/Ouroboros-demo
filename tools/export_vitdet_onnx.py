"""Export the server's ViTDet-B (Mask R-CNN, box branch only) to ONNX for the phone.

Same weights and same 1024x1024 canvas as demo_server.py's naive mode, so the
phone's on-device mode runs *exactly* the model the server runs -- just on a
phone. Input: uint8 (1024, 1024, 3) canvas in the same channel order the
server feeds the model (cv2 BGR). Output: boxes (N,4) in canvas pixels,
scores (N,), classes (N,) -- before the 0.5 result threshold, which the app
applies (same as the server).

  python tools/export_vitdet_onnx.py --out phone/app/src/main/assets/vitdet_b.onnx [--fp16]
"""
import argparse, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(os.environ.get("OUROBOROS_ROOT", Path.home() / "Ouroboros"))
sys.path.insert(0, str(ROOT / "offload"))
os.chdir(ROOT / "offload")
from models import MaskedRCNN_ViT_B_FPN_Contexted  # noqa: E402
from models.ViTDet.structures import ImageList  # noqa: E402


class BoxOnly(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.m = base
        self.register_buffer("mean", base.pixel_mean.view(3, 1, 1).clone())
        self.register_buffer("std", base.pixel_std.view(3, 1, 1).clone())

    def forward(self, img_u8):
        x = img_u8.permute(2, 0, 1).float()
        x = (x - self.mean) / self.std
        images = ImageList(x[None], [(x.shape[1], x.shape[2])])
        feats = self.m.backbone(images.tensor)
        proposals, _ = self.m.proposal_generator(images, feats, None)
        inst = self.m.roi_heads._forward_box(feats, proposals)[0]
        return inst.pred_boxes.tensor, inst.scores, inst.pred_classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--check-image", default=str(ROOT / "offload" / "input.mp4"))
    args = ap.parse_args()
    out = Path(args.out).resolve()

    wrapper = MaskedRCNN_ViT_B_FPN_Contexted(device="cpu")
    wrapper.load_weight(str(ROOT / "offload" / "models" / "model_final_61ccd1.pkl"))
    wrapper.eval()
    model = BoxOnly(wrapper.base_model).eval()

    import cv2
    cap = cv2.VideoCapture(args.check_image); ok, frame = cap.read()
    frame = cv2.resize(frame, (640, 360))
    canvas = np.zeros((1024, 1024, 3), np.uint8)
    oy, ox = (1024 - 360) // 2, (1024 - 640) // 2
    canvas[oy:oy + 360, ox:ox + 640] = frame
    inp = torch.from_numpy(canvas)

    with torch.no_grad():
        ref = model(inp)
        out.parent.mkdir(parents=True, exist_ok=True)
        t = time.time()
        torch.onnx.export(model, (inp,), str(out), opset_version=args.opset, dynamo=False,
                          input_names=["image"], output_names=["boxes", "scores", "classes"],
                          do_constant_folding=True)
        print(f"exported {out} ({out.stat().st_size/1e6:.1f} MB) in {time.time()-t:.0f}s")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    t = time.time(); b, s, c = sess.run(None, {"image": canvas}); dt = time.time() - t
    keep = s > 0.5; rk = ref[1].numpy() > 0.5
    print(f"onnxruntime CPU (this host): {dt*1000:.0f} ms")
    print("torch :", ref[0].numpy()[rk].round(1).tolist(), ref[1].numpy()[rk].round(3).tolist(), ref[2].numpy()[rk].tolist())
    print("onnx  :", b[keep].round(1).tolist(), s[keep].round(3).tolist(), c[keep].tolist())
    if keep.sum() == rk.sum():
        print("max |box diff| =", float(np.abs(b[keep] - ref[0].numpy()[rk]).max()))


if __name__ == "__main__":
    main()
