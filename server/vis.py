"""Green-channel boost of the patches the server recomputed, drawn on the client's frame.

Same look as upstream Ouroboros (offload/server.py, on_device/evaluate_imvid.py):
    img[..., 1] = clip(img[..., 1] + dmap * 30, 0, 255)
but upstream draws on the 1024x1024 aligned canvas, while the demo shows the camera
frame. The dirtiness map lives in canvas space (canvas = affine(resized frame)), so each
frame pixel looks up its patch through that affine. A refresh recomputes every patch,
so the whole frame is boosted.
"""
import base64

import cv2
import numpy as np

CANVAS, BLOCK = 1024, 16
GRID = CANVAS // BLOCK
MAX_DIM = int(CANVAS * 0.8)   # demo_server.MAX_DIM
BOOST = 30


def recompute_mask(resp, frame_hw):
    """bool HxW mask of frame pixels whose patch the server recomputed (None if unknown)."""
    H, W = frame_hw
    if resp.get("refresh"):
        return np.ones((H, W), bool)
    bits, aff = resp.get("dmap"), resp.get("affine")
    if not bits or aff is None:
        return None
    grid = np.unpackbits(np.frombuffer(base64.b64decode(bits), np.uint8))[:GRID * GRID].reshape(GRID, GRID)
    W0, H0 = resp.get("frame_wh", [W, H])
    s = MAX_DIM / max(W0, H0) if max(W0, H0) > MAX_DIM else 1.0
    # frame pixel (x, y) of the displayed image -> sent frame -> resized -> canvas -> patch grid
    A = np.asarray(aff, np.float64)
    M = A.copy(); M[:, 0] *= s * W0 / W; M[:, 1] *= s * H0 / H; M /= BLOCK
    M[:, 2] -= 0.5   # INTER_NEAREST rounds the source coordinate; the patch index is a floor
    m = cv2.warpAffine(grid, M, (W, H), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return m.astype(bool)


def boost_recomputed(img_bgr, resp, amount=BOOST):
    """In place: + amount on the green channel of recomputed patches (saturating)."""
    m = recompute_mask(resp, img_bgr.shape[:2])
    if m is not None:
        g = img_bgr[:, :, 1]
        g[m] = np.minimum(g[m].astype(np.int16) + amount, 255).astype(np.uint8)
    return img_bgr


# ---------------------------------------------------------------- shared global coordinate view
#
# Ouroboros keeps one 1024x1024 canvas that is a *torus*: when the camera pans, the anchor
# (image + cached ViT features) is rolled by a whole number of patches instead of being
# re-padded, so content that leaves one edge comes back in at the other. The world position of
# the current canvas is therefore the cumulative roll  W = sum(shift) mod 1024  (the server
# returns it as resp["world"]); a canvas pixel c sits at world pixel (c + W) mod 1024.
# This mosaic paints every registered frame at its world position on that torus, so the frame
# visibly moves over a fixed map and wraps around its edges.

class WorldMosaic:
    DIM = 0.45                      # brightness of world content outside the current frame
    GRID_STEP = 128

    def __init__(self):
        self.img = np.zeros((CANVAS, CANVAS, 3), np.uint8)
        self.seen = np.zeros((CANVAS, CANVAS), bool)
        self.n = 0

    def clear(self):
        self.img[:] = 0; self.seen[:] = False

    def update(self, target, footprint, dmap_grid, world, boxes=(), names=(), boost=True,
               color=(105, 240, 174), refresh=False, reset=False):
        """target: canvas image (BGR, current canvas coords); footprint: bool canvas mask of the
        frame; dmap_grid: bool 64x64 recomputed patches; boxes: canvas-space xyxy. Returns BGR."""
        if reset:
            self.clear()
        wx, wy = int(world[0]) % CANVAS, int(world[1]) % CANVAS
        roll = lambda a: np.roll(a, shift=(wy, wx), axis=(0, 1))
        fw = roll(footprint)
        self.img[fw] = roll(target)[fw]; self.seen |= fw; self.n += 1   # the map keeps raw pixels

        out = (self.img * self.DIM).astype(np.uint8)
        out[fw] = self.img[fw]
        if boost:   # only the current frame shows what was recomputed for it
            d = cv2.resize(dmap_grid.astype(np.uint8), (CANVAS, CANVAS), interpolation=cv2.INTER_NEAREST).astype(bool)
            d = roll(d & footprint)
            g = out[:, :, 1]
            g[d] = np.minimum(g[d].astype(np.int16) + BOOST, 255).astype(np.uint8)
        for v in range(0, CANVAS, self.GRID_STEP):         # fixed world grid (torus seams at 0)
            c = (70, 70, 70) if v else (0, 160, 255)
            cv2.line(out, (v, 0), (v, CANVAS - 1), c, 1); cv2.line(out, (0, v), (CANVAS - 1, v), c, 1)

        # outline + boxes drawn in canvas coords, then rolled with the frame (so they wrap too)
        ov = np.zeros_like(out); om = np.zeros((CANVAS, CANVAS), np.uint8)
        cs, _ = cv2.findContours(footprint.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        yellow = (0, 255, 255)
        cv2.drawContours(ov, cs, -1, yellow, 3); cv2.drawContours(om, cs, -1, 255, 3)
        for (x0, y0, x1, y1), nm in zip(boxes, names):
            p0, p1 = (int(x0), int(y0)), (int(x1), int(y1))
            cv2.rectangle(ov, p0, p1, color, 3); cv2.rectangle(om, p0, p1, 255, 3)
            org = (int(x0) + 4, max(int(y0) - 8, 20))
            cv2.putText(ov, nm, org, 0, 0.8, color, 2); cv2.putText(om, nm, org, 0, 0.8, 255, 2)
        om = roll(om).astype(bool); out[om] = roll(ov)[om]
        tag = f"world ({wx},{wy})" + ("  REFRESH" if refresh else "")
        cv2.putText(out, tag, (12, CANVAS - 16), 0, 1.0, (0, 0, 0), 5)
        cv2.putText(out, tag, (12, CANVAS - 16), 0, 1.0, (255, 255, 255), 2)
        return out


def canvas_footprint(frame_hw, affine=None, offset=None):
    """bool canvas mask covered by the (resized) frame: affine-warped or centre-padded."""
    h, w = frame_hw
    if affine is not None:
        return cv2.warpAffine(np.ones((h, w), np.uint8), np.asarray(affine, np.float64), (CANVAS, CANVAS),
                              flags=cv2.INTER_NEAREST).astype(bool)
    m = np.zeros((CANVAS, CANVAS), bool)
    m[offset[1]:offset[1] + h, offset[0]:offset[0] + w] = True
    return m
