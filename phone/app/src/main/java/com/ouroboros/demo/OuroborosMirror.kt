package com.ouroboros.demo

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.DMatch
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.features.BFMatcher
import org.opencv.features.ORB
import org.opencv.geometry.Geometry
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import kotlin.math.atan2
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Phone-side replica of the server's Ouroboros preprocessing
 * (Ouroboros/offload/preprocessing.py + demo_server.py's process()):
 *
 *   ORB registration of the frame against the anchor canvas (rigid),
 *   toroidal shift of the anchor (np.roll), absdiff -> gray -> blur ->
 *   threshold -> blur -> 64x64 -> ">0"  ==  the set of patches the server
 *   is about to recompute.
 *
 * It works on the *same JPEG bytes* the server receives (decoded with the
 * same OpenCV 5.0.0), and keeps its own copy of the anchor canvas in lock
 * step with the server's, so the patch count it predicts is the one the
 * server will actually use. That count is what the latency predictor
 * turns into a compute-time estimate before the server has even replied.
 *
 * Usage per frame (strictly sequential, like the stop-and-wait pipeline):
 *   val p = predict(jpeg, thres, whatIfThresholds)   // in parallel with the upload
 *   ...response arrives...
 *   commit(serverRefresh = resp.refresh)               // advance anchor like the server did
 */
class OuroborosMirror(var gop: Int = 20) {

    data class Prediction(
        val nDirty: Int,                 // for the threshold actually being sent
        val refresh: Boolean,            // mirror expects a full refresh (all 4096 patches)
        val reason: String?,
        val whatIf: List<Pair<Int, Int>>, // (threshold, nDirty) for the what-if table
        val shift: IntArray,
        val affine: DoubleArray?,        // 2x3 row-major, after the wrap-around shift adjustment
        val ms: Double,                  // time the mirror itself took on the phone
    )

    companion object {
        const val CANVAS = 1024
        const val BLOCK = 16
        const val GRID = CANVAS / BLOCK
        const val N_PATCHES = GRID * GRID
        private const val MAX_DIM = (CANVAS * 0.8).toInt()   // 819
    }

    private var anchor: Mat? = null
    private var frameIdx = 0

    // per-frame scratch carried from predict() to commit()
    private var pFrame: Mat? = null
    private var pTarget: Mat? = null
    private var pRolledAnchor: Mat? = null
    private var pDmap: Mat? = null          // 64x64 CV_8U, 1 = recompute
    private var pRefresh = true

    private val orb: ORB = ORB.create()
    private val matcher: BFMatcher = BFMatcher.create(Core.NORM_HAMMING, true)

    fun reset() {
        anchor?.release(); anchor = null
        frameIdx = 0
        releaseScratch()
    }

    fun predict(jpeg: ByteArray, thres: Int, whatIf: List<Int>): Prediction {
        val t0 = System.nanoTime()
        releaseScratch()
        var frame = Imgcodecs.imdecode(MatOfByte(*jpeg), Imgcodecs.IMREAD_COLOR)
        if (max(frame.cols(), frame.rows()) > MAX_DIM) {
            val s = MAX_DIM.toDouble() / max(frame.cols(), frame.rows())
            val r = Mat()
            Imgproc.resize(frame, r, Size((frame.cols() * s).toInt().toDouble(), (frame.rows() * s).toInt().toDouble()),
                0.0, 0.0, Imgproc.INTER_LINEAR)
            frame.release(); frame = r
        }
        pFrame = frame

        val a = anchor
        var refresh = a == null || frameIdx % gop == 0
        var reason: String? = if (a == null) "first" else if (refresh) "gop" else null
        var shift = intArrayOf(0, 0)
        var counts: List<Pair<Int, Int>> = whatIf.map { it to N_PATCHES }
        var nDirty = N_PATCHES
        var affOut: DoubleArray? = null

        if (!refresh) {
            val aff = estimateAffine(a!!, frame)
            if (aff == null) {
                refresh = true; reason = "reg_fail"
            } else {
                val (target, sh, affAdj) = applyAffineAndPad(frame, aff)
                shift = sh
                val rolled = roll(a, -sh[0], -sh[1])
                // residual -> gray -> blur, shared by every threshold
                val diff = Mat(); Core.absdiff(rolled, target, diff)
                val gray = Mat(); Imgproc.cvtColor(diff, gray, Imgproc.COLOR_BGR2GRAY); diff.release()
                Imgproc.GaussianBlur(gray, gray, Size(15.0, 15.0), 1.5)
                // create_dirtiness_map's  >thr -> GaussianBlur(15,1.5) -> resize(1/16, LINEAR) -> >0
                // is exactly "max of the blurred residual over the 16x16 patch > thr":
                // the 15x15 Gaussian support plus the 2x2 bilinear taps at 16i+7..8 cover
                // precisely that patch. (Verified identical to the server on 1800 cases.)
                // So one 64x64 max-pool serves every threshold.
                val pmax = patchMax(gray); gray.release()
                counts = whatIf.map { t -> t to pmax.count { (it.toInt() and 0xff) > t } }
                pDmap = Mat(GRID, GRID, CvType.CV_8U).also { m ->
                    m.put(0, 0, ByteArray(N_PATCHES) { if ((pmax[it].toInt() and 0xff) > thres) 1 else 0 })
                }
                nDirty = pmax.count { (it.toInt() and 0xff) > thres }
                pTarget = target; pRolledAnchor = rolled
                affOut = DoubleArray(6) { affAdj.get(it / 3, it % 3)[0] }
                if (frobenius2x2(affAdj) < 0.95) { refresh = true; reason = "scale" }
                affAdj.release()
            }
        }
        if (refresh) { nDirty = N_PATCHES; counts = whatIf.map { it to N_PATCHES }; shift = intArrayOf(0, 0) }
        pRefresh = refresh
        return Prediction(nDirty, refresh, reason, counts, shift, affOut, (System.nanoTime() - t0) / 1e6)
    }

    /**
     * Advance the anchor with what the *server* actually did for this frame
     * (its refresh decision, its affine incl. wrap-around shift, its
     * recompute map), not with the phone's own prediction. A prediction that
     * was off for one frame (e.g. a borderline ORB/LMEDS difference) then
     * costs exactly that frame, and the mirror never drifts from the server.
     */
    fun commit(serverRefresh: Boolean, serverAffine: DoubleArray?, serverShift: IntArray?, serverDmapBits: ByteArray?) {
        val frame = pFrame ?: return
        val a = anchor
        val newAnchor: Mat
        if (serverRefresh || a == null || serverAffine == null || serverShift == null || serverDmapBits == null) {
            newAnchor = padCentered(frame)
        } else {
            newAnchor = roll(a, -serverShift[0], -serverShift[1])
            val m = Mat(2, 3, CvType.CV_64F); m.put(0, 0, *serverAffine)
            val target = Mat()
            Imgproc.warpAffine(frame, target, m, Size(CANVAS.toDouble(), CANVAS.toDouble()))
            val bits = ByteArray(N_PATCHES) { i -> ((serverDmapBits[i shr 3].toInt() shr (7 - (i and 7))) and 1).toByte() }
            val dm = Mat(GRID, GRID, CvType.CV_8U); dm.put(0, 0, bits)
            val up = Mat()
            Imgproc.resize(dm, up, Size(CANVAS.toDouble(), CANVAS.toDouble()), 0.0, 0.0, Imgproc.INTER_NEAREST)
            target.copyTo(newAnchor, up)
            m.release(); target.release(); dm.release(); up.release()
        }
        anchor?.release(); anchor = newAnchor
        frameIdx++
        releaseScratch()
    }

    // ---- ports of Ouroboros/offload/preprocessing.py ----

    /** estimate_affine_in_padded_anchor_fast(mode="rigid") */
    private fun estimateAffine(anchorPadded: Mat, target: Mat): Mat? {
        val scaler = 2
        val a = Mat(); Imgproc.resize(anchorPadded, a, Size((anchorPadded.cols() / scaler).toDouble(), (anchorPadded.rows() / scaler).toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
        val t = Mat(); Imgproc.resize(target, t, Size((target.cols() / scaler).toDouble(), (target.rows() / scaler).toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
        val kp1 = MatOfKeyPoint(); val des1 = Mat(); orb.detectAndCompute(t, Mat(), kp1, des1)
        val kp2 = MatOfKeyPoint(); val des2 = Mat(); orb.detectAndCompute(a, Mat(), kp2, des2)
        a.release(); t.release()
        if (des1.empty() || des2.empty()) return null
        val mm = MatOfDMatch(); matcher.match(des1, des2, mm)
        val matches: List<DMatch> = mm.toList().sortedBy { it.distance }.take(100)
        if (matches.size < 2) return null   // server: cv2 raises below 2 points -> reg_fail
        val k1 = kp1.toArray(); val k2 = kp2.toArray()
        val src = MatOfPoint2f(*matches.map { k1[it.queryIdx].pt }.toTypedArray())
        val dst = MatOfPoint2f(*matches.map { k2[it.trainIdx].pt }.toTypedArray())
        val inl = Mat()
        val m = Geometry.estimateAffinePartial2D(src, dst, inl, Geometry.LMEDS)
        if (m == null || m.empty()) return null
        val a00 = m.get(0, 0)[0]; val a10 = m.get(1, 0)[0]
        val scale = hypot(a00, a10)
        if (scale > 1e-6) {
            val th = atan2(a10, a00)
            m.put(0, 0, cos(th)); m.put(0, 1, -sin(th))
            m.put(1, 0, sin(th)); m.put(1, 1, cos(th))
        }
        m.put(0, 2, m.get(0, 2)[0] * scaler)
        m.put(1, 2, m.get(1, 2)[0] * scaler)
        return m
    }

    /** apply_affine_and_pad(block_size=16, filling_color=[0,0,0]) -> (canvas, shift px, adjusted affine) */
    private fun applyAffineAndPad(target: Mat, aff: Mat): Triple<Mat, IntArray, Mat> {
        val w = target.cols().toDouble(); val h = target.rows().toDouble()
        val pts = arrayOf(Point(0.0, 0.0), Point(0.0, h), Point(w, 0.0), Point(w, h)).map { p ->
            Point(aff.get(0, 0)[0] * p.x + aff.get(0, 1)[0] * p.y + aff.get(0, 2)[0],
                  aff.get(1, 0)[0] * p.x + aff.get(1, 1)[0] * p.y + aff.get(1, 2)[0])
        }
        // python does this in float32 (cv2.transform on float32 points)
        val xs = pts.map { it.x.toFloat() }; val ys = pts.map { it.y.toFloat() }
        var sx = 0; var sy = 0
        if (xs.any { it < 0 } || ys.any { it < 0 } || xs.any { it > CANVAS } || ys.any { it > CANVAS }) {
            val sxm = floor(min(0f, xs.min() / BLOCK).toDouble()).toInt()
            val sxp = ceil(max(0f, (xs.max() - CANVAS) / BLOCK).toDouble()).toInt()
            val sym = floor(min(0f, ys.min() / BLOCK).toDouble()).toInt()
            val syp = ceil(max(0f, (ys.max() - CANVAS) / BLOCK).toDouble()).toInt()
            sx = sxm + sxp; sy = sym + syp
            aff.put(0, 2, aff.get(0, 2)[0] - sx * BLOCK)
            aff.put(1, 2, aff.get(1, 2)[0] - sy * BLOCK)
        }
        val out = Mat()
        Imgproc.warpAffine(target, out, aff, Size(CANVAS.toDouble(), CANVAS.toDouble()))
        // filling colour is black, so result_image[mask] = transformed[mask] is just `out`
        return Triple(out, intArrayOf(sx * BLOCK, sy * BLOCK), aff)
    }

    /** 1024x1024 CV_8U -> 64*64 per-patch maxima, row-major. */
    private fun patchMax(gray: Mat): ByteArray {
        val px = ByteArray(CANVAS * CANVAS); gray.get(0, 0, px)
        val out = IntArray(N_PATCHES)
        for (y in 0 until CANVAS) {
            val rowBase = y * CANVAS; val pRow = (y / BLOCK) * GRID
            for (x in 0 until CANVAS) {
                val v = px[rowBase + x].toInt() and 0xff
                val p = pRow + x / BLOCK
                if (v > out[p]) out[p] = v
            }
        }
        return ByteArray(N_PATCHES) { out[it].toByte() }
    }

    /** create_dirtiness_map, from the already blurred gray residual (reference path, kept for clarity) */
    private fun dirtinessFromBlurred(blurredGray: Mat, thres: Int): Mat {
        val bin = Mat(); Imgproc.threshold(blurredGray, bin, thres.toDouble(), 1.0, Imgproc.THRESH_BINARY)
        val f = Mat(); bin.convertTo(f, CvType.CV_32F); bin.release()
        Imgproc.GaussianBlur(f, f, Size(15.0, 15.0), 1.5)
        val small = Mat(); Imgproc.resize(f, small, Size(GRID.toDouble(), GRID.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR); f.release()
        val out = Mat(); Imgproc.threshold(small, out, 0.0, 1.0, Imgproc.THRESH_BINARY); small.release()
        val out8 = Mat(); out.convertTo(out8, CvType.CV_8U); out.release()
        return out8
    }

    /** np.roll(img, shift=(dx, dy), axis=(1, 0)) */
    private fun roll(src: Mat, dx: Int, dy: Int): Mat {
        val w = src.cols(); val h = src.rows()
        val sx = ((dx % w) + w) % w; val sy = ((dy % h) + h) % h
        if (sx == 0 && sy == 0) return src.clone()
        val dst = Mat(src.size(), src.type())
        // dst[(y+sy)%h, (x+sx)%w] = src[y, x], done as up to 4 rectangle copies
        for ((srcX, dstX, wx) in listOf(Triple(0, sx, w - sx), Triple(w - sx, 0, sx))) {
            if (wx == 0) continue
            for ((srcY, dstY, hy) in listOf(Triple(0, sy, h - sy), Triple(h - sy, 0, sy))) {
                if (hy == 0) continue
                src.submat(Rect(srcX, srcY, wx, hy)).copyTo(dst.submat(Rect(dstX, dstY, wx, hy)))
            }
        }
        return dst
    }

    /** get_padded_image(size=1024, filling_color=[0,0,0]) */
    private fun padCentered(frame: Mat): Mat {
        val c = Mat(CANVAS, CANVAS, frame.type(), Scalar(0.0, 0.0, 0.0))
        val ox = (CANVAS - frame.cols()) / 2; val oy = (CANVAS - frame.rows()) / 2
        frame.copyTo(c.submat(Rect(ox, oy, frame.cols(), frame.rows())))
        return c
    }

    private fun frobenius2x2(m: Mat): Double {
        var s = 0.0
        for (r in 0..1) for (c in 0..1) { val v = m.get(r, c)[0]; s += v * v }
        return sqrt(s)
    }

    private fun releaseScratch() {
        pFrame?.release(); pTarget?.release(); pRolledAnchor?.release(); pDmap?.release()
        pFrame = null; pTarget = null; pRolledAnchor = null; pDmap = null
        pRefresh = true
    }
}
