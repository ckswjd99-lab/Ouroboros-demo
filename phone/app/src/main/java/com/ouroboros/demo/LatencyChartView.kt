package com.ouroboros.demo

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.graphics.Path
import android.util.AttributeSet
import android.view.View

/** Rolling multi-series line chart (ms), shared y-axis, last [capacity] frames. */
class LatencyChartView @JvmOverloads constructor(ctx: Context, attrs: AttributeSet? = null) : View(ctx, attrs) {

    class Series(val label: String, color: Int, dashed: Boolean = false) {
        val values = ArrayDeque<Float>()
        val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            this.color = color; strokeWidth = 4f; style = Paint.Style.STROKE
            if (dashed) pathEffect = DashPathEffect(floatArrayOf(14f, 10f), 0f)
        }
    }

    private val capacity = 120
    val series = mutableListOf<Series>()
    private val grid = Paint().apply { color = Color.rgb(60, 60, 60); strokeWidth = 1f }
    private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.LTGRAY; textSize = 26f }

    fun setSeries(vararg s: Series) { series.clear(); series.addAll(s); invalidate() }

    /** One value per series per frame; NaN leaves a gap. */
    fun add(vararg v: Float) {
        for ((i, s) in series.withIndex()) {
            s.values.addLast(v.getOrElse(i) { Float.NaN })
            while (s.values.size > capacity) s.values.removeFirst()
        }
        postInvalidate()
    }

    fun clear() { series.forEach { it.values.clear() }; postInvalidate() }

    override fun onDraw(c: Canvas) {
        c.drawColor(Color.rgb(20, 20, 20))
        val top = 40f; val left = 70f; val w = width - left - 10f; val h = height - top - 10f
        val all = series.flatMap { it.values }.filter { !it.isNaN() }
        val maxV = niceCeil((all.maxOrNull() ?: 100f) * 1.1f)
        for (i in 0..4) {
            val y = top + h * i / 4
            c.drawLine(left, y, left + w, y, grid)
            c.drawText("%.0f".format(maxV * (4 - i) / 4), 4f, y + 9f, text)
        }
        var lx = left
        for (s in series) {
            val tp = Paint(text).apply { color = s.paint.color }
            c.drawText(s.label, lx, 28f, tp); lx += tp.measureText(s.label) + 30f
        }
        for (s in series) {
            val p = Path(); var pen = false
            s.values.forEachIndexed { i, v ->
                if (v.isNaN()) { pen = false; return@forEachIndexed }
                val x = left + w * i / (capacity - 1); val y = top + h * (1 - v / maxV)
                if (pen) p.lineTo(x, y) else p.moveTo(x, y); pen = true
            }
            c.drawPath(p, s.paint)
        }
    }

    private fun niceCeil(v: Float): Float {
        val steps = floatArrayOf(50f, 100f, 150f, 200f, 300f, 400f, 500f, 750f, 1000f, 1500f, 2000f, 3000f, 5000f, 10000f, 20000f, 50000f)
        return steps.firstOrNull { it >= v } ?: v
    }
}
