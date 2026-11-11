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
# Same as upstream (offload/server.py result thread, on_device/evaluate_imvid.py): the *anchor
# canvas* -- exactly the 1024x1024 image whose cached ViT features the next frame reuses -- shown
# rolled by the cumulative wrap-around shift, so content stays put while the canvas slides under it
# and wraps at the edges. The anchor is rebuilt from a centred frame at every refresh (GOP, failed
# registration, scale change), so the global view refreshes with it; the cumulative shift itself is
# never reset (upstream keeps cum_shift too), so the seam keeps its place.

class WorldView:
    """Renders anchor canvas + current frame outline + recomputed patches + boxes, rolled by `world`."""
    DIM = 0.6                       # anchor content outside the current frame (stale, still reused)

    def render(self, anchor, footprint, dmap_grid, world, boxes=(), names=(), boost=True,
               color=(105, 240, 174), refresh=False, frame_idx=None, gop=None):
        """anchor: canvas after this frame's composite (BGR); footprint: bool canvas mask of the current
        frame; dmap_grid: bool 64x64 recomputed patches; boxes: canvas-space xyxy."""
        wx, wy = int(world[0]) % CANVAS, int(world[1]) % CANVAS
        out = (anchor * self.DIM).astype(np.uint8)
        out[footprint] = anchor[footprint]
        if boost:
            d = cv2.resize(dmap_grid.astype(np.uint8), (CANVAS, CANVAS), interpolation=cv2.INTER_NEAREST).astype(bool)
            g = out[:, :, 1]
            g[d] = np.minimum(g[d].astype(np.int16) + BOOST, 255).astype(np.uint8)
        cv2.rectangle(out, (0, 0), (CANVAS - 1, CANVAS - 1), (0, 160, 255), 3)    # canvas edge = wrap seam
        cs, _ = cv2.findContours(footprint.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, (0, 255, 255), 3)
        for (x0, y0, x1, y1), nm in zip(boxes, names):
            cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), color, 3)
            cv2.putText(out, nm, (int(x0) + 4, max(int(y0) - 8, 20)), 0, 0.8, color, 2)
        out = np.roll(out, shift=(wy, wx), axis=(0, 1))
        tag = f"world ({wx},{wy})"
        if frame_idx is not None and gop:
            tag += f"  refresh in {gop - frame_idx % gop}" if not refresh else "  REFRESH"
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
