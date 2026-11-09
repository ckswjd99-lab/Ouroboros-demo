package com.ouroboros.demo

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Online end-to-end latency model for the Ouroboros offload mode.
 *
 *   e2e = encode                     measured before the prediction is made
 *       + upload + download          learned (line in upload KB, split by clock-synced legs)
 *       + gpuWait                    from the server's GPU probe: work other phones still have
 *                                    on the GPU, minus how long until our request gets there,
 *                                    plus a learned residual correction
 *       + pre(refresh)               server decode + registration + dirtiness map (learned)
 *       + vit(nDirty, refresh)       server ViT time, a line in nDirty (learned)
 *       + post + serverOther         server box mapping etc. / HTTP + JSON (learned)
 *       + clientPost                 parse + draw on the phone (learned)
 *
 * Everything is refit continuously from what the server reports for each frame,
 * so it tracks network, load and contention changes at runtime. What the phone
 * supplies at prediction time is nDirty (from [OuroborosMirror], computed in
 * parallel with the upload) and the GPU probe answer.
 */
class LatencyPredictor {

    /** Sliding-window least-squares line with a ridge pull toward a prior slope. */
    class RidgeLine(private val window: Int, private val priorSlope: Double, private val lambda: Double) {
        private val xs = ArrayDeque<Double>(); private val ys = ArrayDeque<Double>()
        fun add(x: Double, y: Double) {
            xs.addLast(x); ys.addLast(y)
            while (xs.size > window) { xs.removeFirst(); ys.removeFirst() }
        }
        val n get() = xs.size
        fun fit(): Pair<Double, Double> {
            if (xs.isEmpty()) return 0.0 to priorSlope
            val mx = xs.average(); val my = ys.average()
            var sxx = 0.0; var sxy = 0.0
            for (i in xs.indices) { val dx = xs[i] - mx; sxx += dx * dx; sxy += dx * (ys[i] - my) }
            val slope = (sxy + lambda * priorSlope) / (sxx + lambda)
            return (my - slope * mx) to slope
        }
        fun predict(x: Double): Double { val (a, b) = fit(); return a + b * x }
        fun residualStd(): Double {
            if (xs.size < 3) return Double.NaN
            val (a, b) = fit(); var s = 0.0
            for (i in xs.indices) { val r = ys[i] - (a + b * xs[i]); s += r * r }
            return sqrt(s / (xs.size - 2))
        }
    }

    class Ewma(private val alpha: Double) {
        var value = Double.NaN; private set
        fun add(x: Double) { if (!x.isNaN()) value = if (value.isNaN()) x else value + alpha * (x - value) }
        fun or(d: Double) = if (value.isNaN()) d else value
    }

    // ViT time: ~46 ms + ~9 us/patch on the GH200 when alone (server/test_client.py)
    private val vitPartial = RidgeLine(window = 80, priorSlope = 0.009, lambda = 2e5)
    private val vitRefresh = Ewma(0.2)
    private val prePartial = Ewma(0.2)
    private val preRefresh = Ewma(0.2)
    private val post = Ewma(0.2)
    private val network = RidgeLine(window = 40, priorSlope = 0.05, lambda = 2e3)   // up+down, ms per KB
    private val uploadShare = Ewma(0.1)          // upload / (upload + download)
    private val serverOther = Ewma(0.2)
    private val clientPost = Ewma(0.2)
    private val waitPlain = Ewma(0.2)            // fallback when no GPU probe
    private val waitResid = Ewma(0.15)           // actual wait - probe-based wait

    data class Estimate(val totalMs: Double, val encodeMs: Double, val networkMs: Double, val gpuWaitMs: Double,
                        val serverOverheadMs: Double, val computeMs: Double, val clientPostMs: Double,
                        val rawWaitMs: Double)

    data class Observation(
        val nDirty: Int, val refresh: Boolean, val uploadKb: Double,
        val preMs: Double, val vitMs: Double, val postMs: Double, val gpuWaitMs: Double,
        val serverProcMs: Double,       // server t_send - t_body
        val httpMs: Double,             // phone: request start -> response read
        val uploadMs: Double,           // clock-synced, may be NaN
        val clientPostMs: Double,
        val rawWaitPredMs: Double,      // probe-based wait we predicted (NaN if no probe)
    )

    /**
     * @param othersGpuMs GPU work other phones still had at the server when our probe was answered
     * @param probeRttMs round trip of that probe (its answer is ~half of it old relative to our send)
     */
    fun estimate(nDirty: Int, refresh: Boolean, uploadKb: Double, encodeMs: Double,
                 probe: OffloadClient.GpuProbe? = null): Estimate {
        val pre = if (refresh) preRefresh.or(1.0) else prePartial.or(15.0)
        val vit = if (refresh) vitRefresh.or(vitPartial.predict(OuroborosMirror.N_PATCHES.toDouble()))
                  else vitPartial.predict(nDirty.toDouble())
        val p = post.or(1.0)
        val net = network.predict(uploadKb).coerceAtLeast(0.0)
        val up = net * uploadShare.or(0.7)
        var raw = Double.NaN
        val wait = if (probe != null) {
            // the probe was answered ~rtt/2 after we started sending; our request reaches the
            // GPU queue after the upload and the server's CPU-side preprocessing
            raw = probe.simulateWait(max(0.0, up - probe.rttMs / 2 + pre))
            max(0.0, raw + waitResid.or(0.0))
        } else waitPlain.or(0.0)
        val so = serverOther.or(1.0); val cp = clientPost.or(2.0)
        val comp = pre + vit + p
        return Estimate(encodeMs + net + wait + so + comp + cp, encodeMs, net, wait, so, comp, cp, raw)
    }

    fun observe(o: Observation) {
        if (o.refresh) { vitRefresh.add(o.vitMs); preRefresh.add(o.preMs) }
        else { vitPartial.add(o.nDirty.toDouble(), o.vitMs); prePartial.add(o.preMs) }
        post.add(o.postMs)
        val net = (o.httpMs - o.serverProcMs).coerceAtLeast(0.0)
        network.add(o.uploadKb, net)
        if (!o.uploadMs.isNaN() && net > 1) uploadShare.add((o.uploadMs / net).coerceIn(0.0, 1.0))
        serverOther.add((o.serverProcMs - o.preMs - o.vitMs - o.postMs - o.gpuWaitMs).coerceAtLeast(0.0))
        clientPost.add(o.clientPostMs)
        waitPlain.add(o.gpuWaitMs)
        if (!o.rawWaitPredMs.isNaN()) waitResid.add(o.gpuWaitMs - o.rawWaitPredMs)
    }

    fun computeModelText(): String {
        val (a, b) = vitPartial.fit()
        return "ViT ≈ %.1f + %.2f×(n/1000) ms (σ %.1f, n=%d) · refresh ViT %.1f · pre %.1f · post %.1f".format(
            a, b * 1000, vitPartial.residualStd(), vitPartial.n, vitRefresh.or(Double.NaN),
            prePartial.or(Double.NaN), post.or(Double.NaN))
    }

    fun networkText(): String {
        val (a, b) = network.fit()
        return "net ≈ %.1f + %.3f×KB ms (σ %.1f, up %.0f%%) · GPU wait 보정 %+.1f ms".format(
            a, b, network.residualStd(), 100 * uploadShare.or(Double.NaN), waitResid.or(0.0))
    }

    /** Running accuracy of past predictions. */
    class ErrorStats {
        private val errs = ArrayDeque<Double>(); private val rel = ArrayDeque<Double>()
        private val nErr = ArrayDeque<Int>()
        fun add(pred: Double, actual: Double, nPred: Int, nActual: Int) {
            errs.addLast(actual - pred); rel.addLast(abs(actual - pred) / actual); nErr.addLast(nPred - nActual)
            while (errs.size > 100) { errs.removeFirst(); rel.removeFirst(); nErr.removeFirst() }
        }
        fun clear() { errs.clear(); rel.clear(); nErr.clear() }
        val count get() = errs.size
        fun mae() = errs.map { abs(it) }.average()
        fun bias() = errs.average()
        fun within(frac: Double) = rel.count { it <= frac }.toDouble() / rel.size.coerceAtLeast(1)
        fun nMae() = nErr.map { abs(it) }.average()
        fun nExact() = nErr.count { it == 0 }.toDouble() / nErr.size.coerceAtLeast(1)
    }
}
