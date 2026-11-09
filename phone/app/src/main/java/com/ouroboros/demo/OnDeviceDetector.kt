package com.ouroboros.demo

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * The server's own ViTDet-B (Mask R-CNN box branch), run on the phone via
 * ONNX Runtime. Input layout mirrors demo_server.py's naive path exactly:
 * the frame is centred on a black 1024x1024 canvas in cv2's BGR byte order,
 * and detections below 0.5 are dropped.
 */
class OnDeviceDetector(modelFile: File, threads: Int) : AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val canvas: ByteBuffer = ByteBuffer.allocateDirect(1024 * 1024 * 3).order(ByteOrder.nativeOrder())
    val loadMs: Double

    init {
        val t0 = System.nanoTime()
        val opts = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            if (threads > 0) setIntraOpNumThreads(threads)
        }
        session = env.createSession(modelFile.absolutePath, opts)
        loadMs = (System.nanoTime() - t0) / 1e6
    }

    data class Result(val dets: List<Det>, val prepMs: Double, val inferMs: Double)

    fun detect(frame: Bitmap): Result {
        val t0 = System.nanoTime()
        val w = frame.width; val h = frame.height
        val ox = (1024 - w) / 2; val oy = (1024 - h) / 2
        val px = IntArray(w * h); frame.getPixels(px, 0, w, 0, 0, w, h)
        canvas.clear()
        val zeroRow = ByteArray(1024 * 3)
        for (y in 0 until 1024) canvas.put(zeroRow)
        for (y in 0 until h) {
            canvas.position(((oy + y) * 1024 + ox) * 3)
            val row = y * w
            for (x in 0 until w) {
                val c = px[row + x]
                canvas.put((c and 0xff).toByte())          // B
                canvas.put(((c shr 8) and 0xff).toByte())   // G
                canvas.put(((c shr 16) and 0xff).toByte())  // R
            }
        }
        canvas.rewind()
        val t1 = System.nanoTime()
        val dets = ArrayList<Det>()
        OnnxTensor.createTensor(env, canvas, longArrayOf(1024, 1024, 3), OnnxJavaType.UINT8).use { input ->
            session.run(mapOf("image" to input)).use { out ->
                @Suppress("UNCHECKED_CAST")
                val boxes = out[0].value as Array<FloatArray>
                val scores = out[1].value as FloatArray
                val classes = out[2].value as LongArray
                for (i in scores.indices) {
                    if (scores[i] <= 0.5f) continue
                    val b = boxes[i]
                    dets += Det(
                        ((b[0] - ox) / w).coerceIn(0f, 1f), ((b[1] - oy) / h).coerceIn(0f, 1f),
                        ((b[2] - ox) / w).coerceIn(0f, 1f), ((b[3] - oy) / h).coerceIn(0f, 1f),
                        scores[i], Coco.name(classes[i].toInt()),
                    )
                }
            }
        }
        val t2 = System.nanoTime()
        return Result(dets, (t1 - t0) / 1e6, (t2 - t1) / 1e6)
    }

    override fun close() { session.close() }
}

data class Det(val x0: Float, val y0: Float, val x1: Float, val y1: Float, val score: Float, val name: String)

object Coco {
    private val names = listOf("person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
        "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush")
    fun name(i: Int) = names.getOrElse(i) { i.toString() }
}
