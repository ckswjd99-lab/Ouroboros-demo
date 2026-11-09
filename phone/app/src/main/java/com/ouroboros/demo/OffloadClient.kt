package com.ouroboros.demo

import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

/** Talks to ouroboros-demo/server/demo_server.py. */
class OffloadClient(base: String) {
    val baseUrl = (if (base.startsWith("http")) base else "http://$base").trimEnd('/')

    /** server_clock - phone_clock (ms), from the lowest-RTT probe. */
    var clockOffsetMs = Double.NaN; private set
    var probeRttMs = Double.NaN; private set

    fun syncClock(probes: Int = 9) {
        var best = Double.MAX_VALUE; var off = Double.NaN
        repeat(probes) {
            val c0 = nowMs()
            val j = JSONObject(get("/v1/time").decodeToString())
            val c1 = nowMs()
            if (c1 - c0 < best) { best = c1 - c0; off = j.getDouble("t") - (c0 + c1) / 2 }
        }
        clockOffsetMs = off; probeRttMs = best
    }

    /** Another phone's job stream on the shared GPU, as the server sees it right now. */
    data class Tenant(val mode: String, val state: String, val remainingMs: Double, val durMs: Double,
                      val nextInMs: Double?, val ahead: Boolean, val aheadNext: Boolean)
    data class GpuProbe(val busyRemainingMs: Double, val rttMs: Double, val busyMode: String?,
                        val sched: String, val tenants: List<Tenant>) {
        /**
         * Expected GPU wait for a request that reaches the server's GPU queue [arriveInMs]
         * after the probe was answered. Replays the server's scheduler (demo_server.GpuScheduler)
         * over the next few jobs: the one running now, anything queued, and each other tenant's
         * next job if it is due before we get there.
         *  - an idle GPU starts whatever arrives (work-conserving) ...
         *  - ... except under "fair", where it is held for a cheaper tenant (us, when the other
         *    is not `ahead`) due within MAX_HOLD_MS
         *  - when both are waiting, `ahead` decides (always true under FIFO: it got there first)
         */
        fun simulateWait(arriveInMs: Double): Double {
            var free = busyRemainingMs
            val arrivals = tenants.flatMap { t ->
                listOfNotNull(if (t.state == "queued") Triple(0.0, t.durMs, t.ahead) else null,
                              t.nextInMs?.let { Triple(it, t.durMs, t.aheadNext) })
            }.sortedBy { it.first }
            for ((at, dur, ahead) in arrivals) {
                if (at >= arriveInMs) continue
                val heldForUs = sched == "fair" && !ahead && arriveInMs - at <= MAX_HOLD_MS
                free = when {
                    heldForUs -> free                       // server keeps the GPU for us
                    at >= free -> at + dur                  // GPU idle when it arrives: it starts
                    free <= arriveInMs -> free + dur        // GPU frees before we arrive: it runs
                    ahead -> free + dur                     // both waiting, it wins
                    else -> free                            // both waiting, we win
                }
            }
            return maxOf(0.0, free - arriveInMs)
        }
        companion object { const val MAX_HOLD_MS = 150.0 }   // demo_server.GpuScheduler.MAX_HOLD_MS
    }

    /** What the server's GPU still owes *other* phones right now (GET /v1/gpu). */
    fun gpuProbe(): GpuProbe {
        val c0 = nowMs()
        val j = JSONObject(get("/v1/gpu").decodeToString())
        val rtt = nowMs() - c0
        val ts = j.optJSONArray("tenants")
        val tenants = (0 until (ts?.length() ?: 0)).map { i ->
            val t = ts!!.getJSONObject(i)
            Tenant(t.getString("mode"), t.getString("state"), t.optDouble("remaining", 0.0), t.optDouble("dur", 0.0),
                if (t.isNull("next_in")) null else t.getDouble("next_in"), t.optBoolean("ahead", true),
                t.optBoolean("ahead_next", t.optBoolean("ahead", true)))
        }
        return GpuProbe(j.optDouble("busy_remaining", 0.0), rtt,
            j.optString("busy_mode").takeIf { it.isNotEmpty() && it != "null" }, j.optString("sched", "fifo"), tenants)
    }

    data class Response(
        val json: JSONObject, val dets: List<Det>,
        val tStart: Double, val tEnd: Double,     // phone clock, request start / body fully read
        val uploadMs: Double, val downloadMs: Double, val serverProcMs: Double,
    ) {
        val httpMs get() = tEnd - tStart
        val timing: JSONObject get() = json.getJSONObject("timing")
        val computeMs get() = timing.getDouble("compute")
        val nDirty get() = json.getInt("n_dirty")
        val refresh get() = json.getBoolean("refresh")
    }

    fun infer(jpeg: ByteArray, mode: String, thres: Int, reset: Boolean): Response {
        val q = "/v1/infer?mode=$mode&thres=$thres" + if (reset) "&reset=1" else ""
        val tStart = nowMs()
        val body = post(q, jpeg)
        val tEnd = nowMs()
        val j = JSONObject(body.decodeToString())
        if (j.has("error")) throw RuntimeException("server: ${j.getString("error")}")
        val boxes = j.getJSONArray("boxes"); val scores = j.getJSONArray("scores"); val names = j.getJSONArray("names")
        val dets = (0 until boxes.length()).map { i ->
            val b = boxes.getJSONArray(i)
            Det(b.getDouble(0).toFloat(), b.getDouble(1).toFloat(), b.getDouble(2).toFloat(), b.getDouble(3).toFloat(),
                scores.getDouble(i).toFloat(), names.getString(i))
        }
        val tBody = j.getDouble("t_body"); val tSend = j.getDouble("t_send")
        val serverProc = tSend - tBody
        var up = Double.NaN; var down = Double.NaN
        if (!clockOffsetMs.isNaN()) {
            up = (tBody - clockOffsetMs) - tStart
            down = tEnd - (tSend - clockOffsetMs)
            // clamp clock-sync noise while keeping up + server + down == http
            val net = (tEnd - tStart) - serverProc
            up = up.coerceIn(0.0, net); down = net - up
        }
        return Response(j, dets, tStart, tEnd, up, down, serverProc)
    }

    fun download(name: String, dest: File, progress: (Long, Long) -> Unit) {
        val c = URL("$baseUrl/v1/model/$name").openConnection() as HttpURLConnection
        c.connectTimeout = 5000; c.readTimeout = 30000
        if (c.responseCode != 200) throw RuntimeException("model download: HTTP ${c.responseCode}")
        val total = c.contentLengthLong
        val tmp = File(dest.path + ".part")
        c.inputStream.use { inp -> tmp.outputStream().use { out ->
            val buf = ByteArray(1 shl 20); var got = 0L; var lastReport = 0L
            while (true) {
                val n = inp.read(buf); if (n < 0) break
                out.write(buf, 0, n); got += n
                if (got - lastReport > (4 shl 20)) { progress(got, total); lastReport = got }
            }
        } }
        if (total > 0 && tmp.length() != total) throw RuntimeException("model download truncated (${tmp.length()}/$total)")
        tmp.renameTo(dest)
    }

    private fun get(path: String): ByteArray {
        val c = URL(baseUrl + path).openConnection() as HttpURLConnection
        c.connectTimeout = 3000; c.readTimeout = 5000
        return c.inputStream.use { it.readBytes() }
    }

    private fun post(path: String, data: ByteArray): ByteArray {
        val c = URL(baseUrl + path).openConnection() as HttpURLConnection
        c.connectTimeout = 3000; c.readTimeout = 20000
        c.requestMethod = "POST"; c.doOutput = true
        c.setRequestProperty("Content-Type", "image/jpeg")
        c.setFixedLengthStreamingMode(data.size)
        c.outputStream.use { it.write(data) }
        val code = c.responseCode
        val stream = if (code in 200..299) c.inputStream else c.errorStream
        return stream.use { it.readBytes() }
    }

    companion object { fun nowMs() = System.nanoTime() / 1e6 }
}
