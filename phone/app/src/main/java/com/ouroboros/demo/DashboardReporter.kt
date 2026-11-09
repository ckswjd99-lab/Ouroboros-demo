package com.ouroboros.demo

import android.graphics.Bitmap
import android.util.Base64
import android.util.Log
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.max

/**
 * Fire-and-forget per-frame status for the live dashboard (server GET /).
 *
 * Off the measurement path: reports are sent from their own thread after the
 * frame is already on screen, at most one in flight, latest-wins (a report
 * that arrives while one is sending replaces any queued one instead of
 * piling up). The displayed frame is attached at most [imageHz] times/s,
 * downscaled to [imageMaxSide] px.
 */
class DashboardReporter(private val baseUrl: String, private val imageHz: Double = 4.0,
                        private val imageMaxSide: Int = 360) {
    private val exec = Executors.newSingleThreadExecutor()
    private val pending = AtomicReference<Pair<JSONObject, Bitmap?>?>(null)
    @Volatile private var lastImageMs = 0.0
    @Volatile var enabled = true

    fun offer(rec: JSONObject, shown: Bitmap?) {
        if (!enabled) return
        val now = OffloadClient.nowMs()
        val img = if (shown != null && now - lastImageMs >= 1000.0 / imageHz) { lastImageMs = now; shown } else null
        if (pending.getAndSet(rec to img) == null) exec.execute { drain() }
    }

    private fun drain() {
        while (true) {
            val (rec, img) = pending.getAndSet(null) ?: return
            try {
                if (img != null) rec.put("image", encode(img))
                val c = URL("$baseUrl/v1/report").openConnection() as HttpURLConnection
                c.connectTimeout = 2000; c.readTimeout = 3000
                c.requestMethod = "POST"; c.doOutput = true
                c.setRequestProperty("Content-Type", "application/json")
                val body = rec.toString().toByteArray()
                c.setFixedLengthStreamingMode(body.size)
                c.outputStream.use { it.write(body) }
                c.inputStream.use { it.readBytes() }
            } catch (e: Exception) {
                Log.w("OuroDemo", "dashboard report failed: ${e.message}")
            }
        }
    }

    private fun encode(b: Bitmap): String {
        val s = imageMaxSide.toFloat() / max(b.width, b.height)
        val small = if (s < 1f) Bitmap.createScaledBitmap(b, (b.width * s).toInt(), (b.height * s).toInt(), true) else b
        val out = ByteArrayOutputStream()
        small.compress(Bitmap.CompressFormat.JPEG, 70, out)
        return Base64.encodeToString(out.toByteArray(), Base64.NO_WRAP)
    }

    fun shutdown() = exec.shutdown()
}

/** JSONObject that stores NaN/Infinity as null instead of throwing (Android's org.json rejects them). */
class J : JSONObject() {
    override fun put(name: String, value: Double): JSONObject =
        if (value.isFinite()) super.put(name, value) else super.put(name, JSONObject.NULL)
}
