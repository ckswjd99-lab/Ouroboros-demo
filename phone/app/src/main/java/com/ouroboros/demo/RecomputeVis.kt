package com.ouroboros.demo

import android.graphics.Bitmap
import org.json.JSONObject
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.max

/**
 * Visualisations of what the Ouroboros server did with a frame (port of server/vis.py).
 *
 *  - [boostFrame]: green-channel boost (+30, saturating) of the patches the server recomputed,
 *    like upstream Ouroboros' `img[..., 1] = clip(img[..., 1] + dmap * 30, 0, 255)`, but on the
 *    camera frame: the dirtiness map is in canvas space (canvas = affine(resized frame)), so every
 *    frame pixel looks its patch up through the server's affine. A refresh recomputes everything.
 *  - [WorldView]: the shared global coordinate, as upstream draws it: the anchor canvas (the image
 *    whose cached features the server reuses) rolled by the cumulative wrap-around shift `world`.
 *    The anchor is rebuilt from a centred frame at every refresh, so the view refreshes with it.
 */
object RecomputeVis {
    const val CANVAS = OuroborosMirror.CANVAS
    const val BLOCK = OuroborosMirror.BLOCK
    const val GRID = OuroborosMirror.GRID
    private const val MAX_DIM = (CANVAS * 0.8).toInt()
    const val BOOST = 30.0

    /** Server response fields this needs, parsed once. */
    class Info(json: JSONObject) {
        val refresh = json.optBoolean("refresh", true)
        val affine: DoubleArray? = json.optJSONArray("affine")?.let { a ->
            DoubleArray(6) { a.getJSONArray(it / 3).getDouble(it % 3) } }
        val grid: ByteArray? = json.optString("dmap", "").takeIf { it.isNotEmpty() && it != "null" }?.let {
            val b = android.util.Base64.decode(it, android.util.Base64.DEFAULT)
            ByteArray(GRID * GRID) { i -> ((b[i shr 3].toInt() shr (7 - (i and 7))) and 1).toByte() }
        }
        val frameW: Int; val frameH: Int
        init { val wh = json.optJSONArray("frame_wh"); frameW = wh?.getInt(0) ?: 0; frameH = wh?.getInt(1) ?: 0 }
        val world: IntArray? = json.optJSONArray("world")?.let { intArrayOf(it.getInt(0), it.getInt(1)) }
        val shift: IntArray? = json.optJSONArray("shift")?.let { intArrayOf(it.getInt(0), it.getInt(1)) }
        val idx = json.optInt("idx", -1)
        val gop = json.optInt("gop", 0)
        val scale get() = if (max(frameW, frameH) > MAX_DIM) MAX_DIM.toDouble() / max(frameW, frameH) else 1.0
        /** frame (sent-frame pixels) -> canvas, as a 2x3 row-major array */
        fun frameToCanvas(): DoubleArray {
            val s = scale
            return if (!refresh && affine != null) {
                val a = affine
                doubleArrayOf(a[0] * s, a[1] * s, a[2], a[3] * s, a[4] * s, a[5])
            } else {
                val w = (frameW * s).toInt(); val h = (frameH * s).toInt()
                doubleArrayOf(s, 0.0, ((CANVAS - w) / 2).toDouble(), 0.0, s, ((CANVAS - h) / 2).toDouble())
            }
        }
    }

    private fun mat23(a: DoubleArray) = Mat(2, 3, CvType.CV_64F).also { it.put(0, 0, *a) }

    private fun gridMat(g: ByteArray) = Mat(GRID, GRID, CvType.CV_8U).also { it.put(0, 0, g) }

    /** In place on a mutable ARGB_8888 bitmap that shows the frame sent to the server. */
    fun boostFrame(bmp: Bitmap, info: Info) {
        val w = bmp.width; val h = bmp.height
        val mask: Mat
        if (info.refresh) {
            mask = Mat(h, w, CvType.CV_8U, Scalar(1.0))
        } else {
            val a = info.affine ?: return; val g = info.grid ?: return
            val sx = info.scale * info.frameW / w; val sy = info.scale * info.frameH / h
            // displayed pixel -> canvas -> patch index; INTER_NEAREST rounds, the patch index is a floor
            val m = mat23(doubleArrayOf(a[0] * sx / BLOCK, a[1] * sy / BLOCK, a[2] / BLOCK - 0.5,
                                        a[3] * sx / BLOCK, a[4] * sy / BLOCK, a[5] / BLOCK - 0.5))
            val gm = gridMat(g); mask = Mat()
            Imgproc.warpAffine(gm, mask, m, Size(w.toDouble(), h.toDouble()),
                Imgproc.INTER_NEAREST or Imgproc.WARP_INVERSE_MAP, Core.BORDER_CONSTANT, Scalar(0.0))
            m.release(); gm.release()
        }
        val rgba = Mat(); Utils.bitmapToMat(bmp, rgba)
        Core.add(rgba, Scalar(0.0, BOOST, 0.0, 0.0), rgba, mask)
        Utils.matToBitmap(rgba, bmp)
        rgba.release(); mask.release()
    }

    /**
     * The server's anchor canvas, kept in sync on the phone from what each response says (the same
     * update the server makes): refresh -> centred frame; otherwise roll by -shift and copy in the
     * warped frame where patches were recomputed. Shown rolled by the cumulative shift `world`.
     * Every frame must go through [update] in order; [render]=false skips only the drawing.
     */
    class WorldView(private val outSide: Int = 720) {
        private var anchor: Mat? = null
        /** Forget the anchor; the view resumes at the next refresh. */
        fun clear() { anchor?.release(); anchor = null }

        fun update(frame: Bitmap, info: Info, dets: List<Det>, boxColor: Scalar, boost: Boolean, render: Boolean): Bitmap? {
            val world = info.world ?: return null
            val f = Mat(); Utils.bitmapToMat(frame, f)
            val sendW = info.frameW.takeIf { it > 0 } ?: frame.width
            val sendH = info.frameH.takeIf { it > 0 } ?: frame.height
            val a = info.frameToCanvas()          // sent-frame px -> canvas
            val kx = sendW.toDouble() / frame.width; val ky = sendH.toDouble() / frame.height
            val A = doubleArrayOf(a[0] * kx, a[1] * ky, a[2], a[3] * kx, a[4] * ky, a[5])
            val am = mat23(A)
            val cs = Size(CANVAS.toDouble(), CANVAS.toDouble())
            val target = Mat(); Imgproc.warpAffine(f, target, am, cs, Imgproc.INTER_LINEAR)
            val d = Mat()
            if (info.refresh || info.grid == null) {
                d.create(CANVAS, CANVAS, CvType.CV_8U); d.setTo(Scalar(1.0))
            } else { val gm = gridMat(info.grid); Imgproc.resize(gm, d, cs, 0.0, 0.0, Imgproc.INTER_NEAREST); gm.release() }

            val prev = anchor
            if (info.refresh) {
                prev?.release(); anchor = target.clone()
            } else if (prev != null && info.shift != null) {
                val rolled = roll(prev, -info.shift[0], -info.shift[1]); prev.release()
                target.copyTo(rolled, d); anchor = rolled
            } else {                               // joined mid-GOP: wait for the next refresh
                for (m in listOf(f, am, target, d)) m.release(); return null
            }
            if (!render) { for (m in listOf(f, am, target, d)) m.release(); return null }

            val anc = anchor!!
            val ones = Mat(f.rows(), f.cols(), CvType.CV_8U, Scalar(255.0))
            val foot = Mat(); Imgproc.warpAffine(ones, foot, am, cs, Imgproc.INTER_NEAREST)
            val out = Mat(); Core.multiply(anc, Scalar(DIM, DIM, DIM, 1.0), out)
            anc.copyTo(out, foot)
            if (boost) Core.add(out, Scalar(0.0, BOOST, 0.0, 0.0), out, d)
            Imgproc.rectangle(out, Point(0.0, 0.0), Point(CANVAS - 1.0, CANVAS - 1.0), Scalar(255.0, 160.0, 0.0, 255.0), 3)
            fun map(x: Double, y: Double) = Point(A[0] * x + A[1] * y + A[2], A[3] * x + A[4] * y + A[5])
            val fw = frame.width.toDouble(); val fh = frame.height.toDouble()
            val outline = listOf(MatOfPoint(map(0.0, 0.0), map(fw, 0.0), map(fw, fh), map(0.0, fh)))
            Imgproc.polylines(out, outline, true, Scalar(255.0, 255.0, 0.0, 255.0), 3)
            for (det in dets) {
                val ps = listOf(map(det.x0 * fw, det.y0 * fh), map(det.x1 * fw, det.y0 * fh), map(det.x0 * fw, det.y1 * fh), map(det.x1 * fw, det.y1 * fh))
                val p0 = Point(ps.minOf { it.x }, ps.minOf { it.y }); val p1 = Point(ps.maxOf { it.x }, ps.maxOf { it.y })
                Imgproc.rectangle(out, p0, p1, boxColor, 3)
                Imgproc.putText(out, det.name, Point(p0.x + 4, max(p0.y - 8, 20.0)), Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, boxColor, 2)
            }
            val outW = roll(out, world[0], world[1])
            var tag = "world (${Math.floorMod(world[0], CANVAS)},${Math.floorMod(world[1], CANVAS)})"
            if (info.gop > 0 && info.idx >= 0) tag += if (info.refresh) "  REFRESH" else "  refresh in ${info.gop - info.idx % info.gop}"
            Imgproc.putText(outW, tag, Point(12.0, CANVAS - 16.0), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0.0, 0.0, 0.0, 255.0), 5)
            Imgproc.putText(outW, tag, Point(12.0, CANVAS - 16.0), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255.0, 255.0, 255.0, 255.0), 2)

            val small = Mat(); Imgproc.resize(outW, small, Size(outSide.toDouble(), outSide.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
            val bmp = Bitmap.createBitmap(outSide, outSide, Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(small, bmp)
            for (m in listOf(f, am, target, d, ones, foot, out, outW, small)) m.release()
            outline.forEach { it.release() }
            return bmp
        }

        companion object {
            private const val DIM = 0.6

            /** dst[(y+dy)%h, (x+dx)%w] = src[y, x]  (np.roll(src, (dy, dx), axis=(0, 1))) */
            fun roll(src: Mat, dx: Int, dy: Int): Mat {
                val w = src.cols(); val h = src.rows()
                val sx = ((dx % w) + w) % w; val sy = ((dy % h) + h) % h
                if (sx == 0 && sy == 0) return src.clone()
                val dst = Mat(src.size(), src.type())
                for ((srcX, dstX, wx) in listOf(Triple(0, sx, w - sx), Triple(w - sx, 0, sx))) {
                    if (wx == 0) continue
                    for ((srcY, dstY, hy) in listOf(Triple(0, sy, h - sy), Triple(h - sy, 0, sy))) {
                        if (hy == 0) continue
                        src.submat(Rect(srcX, srcY, wx, hy)).copyTo(dst.submat(Rect(dstX, dstY, wx, hy)))
                    }
                }
                return dst
            }
        }
    }
}
