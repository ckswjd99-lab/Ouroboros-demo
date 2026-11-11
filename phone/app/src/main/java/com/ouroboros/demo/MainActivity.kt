package com.ouroboros.demo

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.ImageView
import android.widget.RadioGroup
import android.widget.SeekBar
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import org.json.JSONArray
import org.json.JSONObject
import org.opencv.android.OpenCVLoader
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.Future
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max

/**
 * Camera -> one of three detection paths -> captured frame + boxes on screen.
 *
 *  ON_DEVICE  the server's ViTDet-B, on the phone (ONNX Runtime, CPU)
 *  NAIVE      JPEG -> server, every frame fully inferred independently
 *  OUROBOROS  JPEG -> server, registration + wrap-around + partial recompute
 *
 * Stop-and-wait: the next camera frame is only taken once the previous
 * frame's result is on screen, so FPS = 1 / e2e latency for every mode.
 *
 * In OUROBOROS mode the optional predictor runs [OuroborosMirror] on the
 * phone *in parallel with the upload* to work out how many patches the
 * server is about to recompute, and [LatencyPredictor] turns that into a
 * predicted end-to-end latency before the server replies.
 */
class MainActivity : AppCompatActivity() {

    /** Each mode has its own server port (one device per mode, all three at once). */
    enum class Mode(val label: String, val color: Int, val port: Int) {
        ON_DEVICE("ON-DEVICE", Color.rgb(255, 82, 82), 65435),
        NAIVE("NAIVE OFFLOAD", Color.rgb(64, 196, 255), 65433),
        OUROBOROS("OUROBOROS", Color.rgb(105, 240, 174), 65434),
    }

    private lateinit var img: ImageView
    private lateinit var txtStatus: TextView
    private lateinit var txtStats: TextView
    private lateinit var txtPredict: TextView
    private lateinit var chart: LatencyChartView
    private lateinit var radioMode: RadioGroup
    private lateinit var seekThres: SeekBar
    private lateinit var txtThres: TextView
    private lateinit var checkPredict: CheckBox
    private lateinit var checkBoost: CheckBox
    private lateinit var checkWorld: CheckBox
    private lateinit var editServer: EditText
    private lateinit var btnStart: Button

    private val cameraExec = Executors.newSingleThreadExecutor()
    private val pipeExec = Executors.newSingleThreadExecutor()
    private val mirrorExec = Executors.newSingleThreadExecutor()
    private val probeExec = Executors.newSingleThreadExecutor()

    private val running = AtomicBoolean(false)
    private val busy = AtomicBoolean(false)
    private var cameraProvider: ProcessCameraProvider? = null

    @Volatile private var thres = 20
    @Volatile private var predictOn = false
    @Volatile private var boostOn = true        // green-boost the patches the server recomputed
    @Volatile private var worldOn = false       // show the shared global (toroidal) canvas instead of the frame
    private val worldView by lazy { RecomputeVis.WorldView() }
    private val visExec = Executors.newSingleThreadExecutor()
    private val worldQueued = java.util.concurrent.atomic.AtomicInteger(0)
    @Volatile private var lastWorld: Bitmap? = null
    @Volatile private var resetNext = true
    private var mode = Mode.OUROBOROS

    private var client: OffloadClient? = null
    private var reporter: DashboardReporter? = null
    private var detector: OnDeviceDetector? = null
    private val mirror by lazy { OuroborosMirror() }   // after OpenCVLoader.initLocal() in onCreate
    private val predictor = LatencyPredictor()
    private val errStats = LatencyPredictor.ErrorStats()
    private val errCompute = LatencyPredictor.ErrorStats()   // compute-only prediction vs server compute
    /** What to predict, chosen on the dashboard ("e2e" or "compute"); the server echoes it in every response. */
    @Volatile private var predTarget = "e2e"
    private var mirrorInSync = false

    private val whatIf = listOf(5, 10, 20, 30, 50, 80)

    // rolling stats
    private val e2eHist = ArrayDeque<Double>()
    private val doneTimes = ArrayDeque<Double>()
    private var frameNo = 0

    private val permLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        if (!ok) txtStatus.text = "카메라 권한이 필요합니다"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        img = findViewById(R.id.imgResult); txtStatus = findViewById(R.id.txtStatus)
        txtStats = findViewById(R.id.txtStats); txtPredict = findViewById(R.id.txtPredict)
        chart = findViewById(R.id.chart); radioMode = findViewById(R.id.radioMode)
        seekThres = findViewById(R.id.seekThres); txtThres = findViewById(R.id.txtThres)
        checkPredict = findViewById(R.id.checkPredict); editServer = findViewById(R.id.editServer)
        btnStart = findViewById(R.id.btnStart)
        checkBoost = findViewById(R.id.checkBoost); checkWorld = findViewById(R.id.checkWorld)
        checkBoost.setOnCheckedChangeListener { _, on -> boostOn = on }
        checkWorld.setOnCheckedChangeListener { _, on ->
            worldOn = on
            // the phone's copy of the anchor only stays exact if it sees every frame, so turning the
            // view on mid-run resumes at the next refresh (<= one GOP)
            if (on) { lastWorld = null; visExec.execute { worldView.clear() } }
        }

        if (!OpenCVLoader.initLocal()) Log.e(TAG, "OpenCV init failed")

        seekThres.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(s: SeekBar, p: Int, fromUser: Boolean) {
                thres = p.coerceAtLeast(1); txtThres.text = "threshold $thres"
            }
            override fun onStartTrackingTouch(s: SeekBar) {}
            override fun onStopTrackingTouch(s: SeekBar) {}
        })
        checkPredict.setOnCheckedChangeListener { _, on ->
            predictOn = on
            // The mirror only tracks the server's anchor while it runs, so
            // (re)enabling it forces a server-side refresh to resync both.
            if (on) { resetNext = true; mirrorInSync = false; errStats.clear(); errCompute.clear() }
            txtPredict.visibility = if (on) View.VISIBLE else View.GONE
        }
        radioMode.setOnCheckedChangeListener { _, id -> mode = modeFor(id); updateModeUi() }
        mode = modeFor(radioMode.checkedRadioButtonId); updateModeUi()
        btnStart.setOnClickListener { if (running.get()) stop() else start() }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED)
            permLauncher.launch(Manifest.permission.CAMERA)

        // allow `adb shell am start ... --es server host:port --es mode ouroboros --ez predict true --ez autostart true`
        intent.getStringExtra("server")?.let { editServer.setText(it) }
        when (intent.getStringExtra("mode")) {
            "ondevice" -> radioMode.check(R.id.modeOnDevice)
            "naive" -> radioMode.check(R.id.modeNaive)
            "ouroboros" -> radioMode.check(R.id.modeOuroboros)
        }
        if (intent.hasExtra("thres")) seekThres.progress = intent.getIntExtra("thres", 20)
        if (intent.hasExtra("predict")) checkPredict.isChecked = intent.getBooleanExtra("predict", false)
        if (intent.hasExtra("boost")) checkBoost.isChecked = intent.getBooleanExtra("boost", true)
        if (intent.hasExtra("world")) checkWorld.isChecked = intent.getBooleanExtra("world", false)
        if (intent.getBooleanExtra("autostart", false)) img.post { start() }
    }

    private fun modeFor(id: Int) = when (id) {
        R.id.modeOnDevice -> Mode.ON_DEVICE
        R.id.modeNaive -> Mode.NAIVE
        else -> Mode.OUROBOROS
    }

    private fun updateModeUi() {
        val ouro = mode == Mode.OUROBOROS
        findViewById<View>(R.id.rowThres).visibility = if (ouro) View.VISIBLE else View.GONE
        checkPredict.visibility = if (ouro) View.VISIBLE else View.GONE
        findViewById<View>(R.id.rowVis).visibility = if (ouro) View.VISIBLE else View.GONE
        txtPredict.visibility = if (ouro && predictOn) View.VISIBLE else View.GONE
        when (mode) {
            Mode.ON_DEVICE -> chart.setSeries(LatencyChartView.Series("e2e", Color.CYAN),
                LatencyChartView.Series("inference", Color.rgb(255, 82, 82)))
            Mode.NAIVE -> chart.setSeries(LatencyChartView.Series("e2e", Color.CYAN),
                LatencyChartView.Series("server compute", Color.rgb(105, 240, 174)))
            Mode.OUROBOROS -> chart.setSeries(LatencyChartView.Series("e2e", Color.CYAN),
                LatencyChartView.Series("server compute", Color.rgb(105, 240, 174)),
                LatencyChartView.Series("pred e2e", Color.rgb(255, 183, 77), dashed = true),
                LatencyChartView.Series("pred compute", Color.rgb(206, 147, 216), dashed = true))
        }
    }

    // ---------------------------------------------------------------- lifecycle

    private fun start() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            permLauncher.launch(Manifest.permission.CAMERA); return
        }
        setControlsEnabled(false); btnStart.text = "정지"
        e2eHist.clear(); doneTimes.clear(); frameNo = 0; chart.clear(); errStats.clear(); errCompute.clear()
        resetNext = true; mirrorInSync = false; mirror.reset(); lastWorld = null; visExec.execute { worldView.clear() }
        val m = mode
        val host = editServer.text.toString().trim().removePrefix("http://").trimEnd('/')
        val addr = if (host.contains(':')) host else "$host:${m.port}"   // explicit host:port overrides
        pipeExec.execute {
            try {
                client = OffloadClient(addr)
                reporter?.shutdown(); reporter = DashboardReporter(client!!.baseUrl)
                if (m == Mode.ON_DEVICE) prepareOnDevice(client!!)
                else {
                    ui { txtStatus.text = "서버 시계 동기화 중…" }
                    client!!.syncClock()
                }
                running.set(true)
                ui { txtStatus.text = "${m.label}: 실행 중 ($addr)"; bindCamera() }
            } catch (e: Exception) {
                Log.e(TAG, "start failed", e)
                ui { txtStatus.text = "시작 실패: ${e.message}"; stop() }
            }
        }
    }

    private fun stop() {
        running.set(false)
        cameraProvider?.unbindAll()
        btnStart.text = "시작"; setControlsEnabled(true)
    }

    private fun setControlsEnabled(on: Boolean) {
        for (i in 0 until radioMode.childCount) radioMode.getChildAt(i).isEnabled = on
        editServer.isEnabled = on
    }

    private fun prepareOnDevice(c: OffloadClient) {
        if (detector != null) return
        val f = File(filesDir, MODEL_NAME)
        if (!f.exists()) {
            ui { txtStatus.text = "모델 다운로드 중…" }
            c.download(MODEL_NAME, f) { got, total ->
                ui { txtStatus.text = "모델 다운로드 %d / %d MB".format(got shr 20, total shr 20) }
            }
        }
        ui { txtStatus.text = "모델 로딩 중 (ViTDet-B, %d MB)…".format(f.length() shr 20) }
        detector = OnDeviceDetector(f, threads = 0)
        Log.i(TAG, "on-device model loaded in ${detector!!.loadMs} ms")
    }

    override fun onDestroy() {
        running.set(false)
        cameraExec.shutdown(); pipeExec.shutdown(); mirrorExec.shutdown(); probeExec.shutdown(); visExec.shutdown()
        detector?.close(); reporter?.shutdown()
        super.onDestroy()
    }

    // ---------------------------------------------------------------- camera

    private fun bindCamera() {
        val f = ProcessCameraProvider.getInstance(this)
        f.addListener({
            val provider = f.get(); cameraProvider = provider
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(ResolutionSelector.Builder().setResolutionStrategy(
                    ResolutionStrategy(Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)).build())
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(cameraExec) { onFrame(it) }
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun onFrame(image: ImageProxy) {
        if (!running.get() || !busy.compareAndSet(false, true)) { image.close(); return }
        val bmp = try { toUpright(image) } catch (e: Exception) { busy.set(false); image.close(); return }
        image.close()
        pipeExec.execute {
            try { if (running.get()) process(bmp) }
            catch (e: Exception) {
                Log.e(TAG, "frame failed", e)
                ui { txtStatus.text = "오류: ${e.message}" }
                resetNext = true; mirrorInSync = false
                Thread.sleep(300)
            } finally { busy.set(false) }
        }
    }

    /** RGBA frame -> upright bitmap with the long side at most 640 px. */
    private fun toUpright(image: ImageProxy): Bitmap {
        val src = image.toBitmap()
        val rot = image.imageInfo.rotationDegrees
        val s = 640f / max(src.width, src.height)
        val m = Matrix().apply { if (s < 1f) postScale(s, s); postRotate(rot.toFloat()) }
        return Bitmap.createBitmap(src, 0, 0, src.width, src.height, m, true)
    }

    // ---------------------------------------------------------------- pipeline

    private fun process(frame: Bitmap) {
        when (mode) {
            Mode.ON_DEVICE -> processOnDevice(frame)
            Mode.NAIVE, Mode.OUROBOROS -> processOffload(frame)
        }
    }

    private fun processOnDevice(frame: Bitmap) {
        val t0 = OffloadClient.nowMs()
        val r = detector!!.detect(frame)
        val vis = draw(frame, r.dets, Mode.ON_DEVICE)
        val e2e = OffloadClient.nowMs() - t0
        recordFrame(e2e)
        chart.add(e2e.toFloat(), r.inferMs.toFloat())
        val stats = buildString {
            append(commonStats())
            append("canvas prep %6.1f ms | inference %8.1f ms\n".format(r.prepMs, r.inferMs))
            append("model: ViTDet-B (same weights as server), ONNX Runtime CPU, 1024×1024\n")
            append("load %.0f ms | dets ${r.dets.size}".format(detector!!.loadMs))
        }
        Log.i(TAG, "CSV,ondevice,$frameNo,%.1f,%.1f,%.1f,${r.dets.size}".format(e2e, r.prepMs, r.inferMs))
        ui { img.setImageBitmap(vis); setStatus(e2e); txtStats.text = stats }
        reporter?.offer(baseRecord(Mode.ON_DEVICE, e2e, r.dets).put("ondevice", J()
            .put("prep", r.prepMs).put("infer", r.inferMs).put("load", detector!!.loadMs)), vis)
    }

    private class Pending(val pred: OuroborosMirror.Prediction, val est: LatencyPredictor.Estimate,
                          val whatIfEst: List<Triple<Int, Int, Double>>, val whatIfCompute: List<Double>,
                          val readyAtMs: Double, val probe: OffloadClient.GpuProbe?, val target: String)

    private fun processOffload(frame: Bitmap) {
        val c = client!!
        val m = mode
        val modeStr = if (m == Mode.NAIVE) "naive" else "ouroboros"
        val th = thres
        val t0 = OffloadClient.nowMs()
        val jpeg = ByteArrayOutputStream(64 * 1024).also { frame.compress(Bitmap.CompressFormat.JPEG, 85, it) }.toByteArray()
        val encMs = OffloadClient.nowMs() - t0
        val kb = jpeg.size / 1024.0

        val wantPredict = m == Mode.OUROBOROS && predictOn
        val reset = resetNext; resetNext = false
        // mirror state is only ever touched on mirrorExec (single thread), so reset/commit
        // stay ordered with the predictions without blocking this loop
        if (reset) { mirrorExec.execute { mirror.reset() }; mirrorInSync = wantPredict }

        // Both run on their own threads while the upload is in flight: the mirror works out
        // how many patches the server will recompute, the probe asks how much GPU work other
        // phones still have queued there.
        val doPredict = wantPredict && mirrorInSync
        val target = predTarget
        // compute-only prediction needs no network or GPU-queue model, so no probe either
        val probeFut: Future<OffloadClient.GpuProbe?>? = if (doPredict && target == "e2e") probeExec.submit<OffloadClient.GpuProbe?> {
            try { c.gpuProbe() } catch (e: Exception) { null }
        } else null
        val fut: Future<Pending>? = if (doPredict) mirrorExec.submit<Pending> {
            val p = mirror.predict(jpeg, th, whatIf)
            val probe = try { probeFut?.get(300, java.util.concurrent.TimeUnit.MILLISECONDS) } catch (e: Exception) { null }
            val est: LatencyPredictor.Estimate
            val wi: List<Triple<Int, Int, Double>>
            val wic: List<Double>
            synchronized(predictor) {
                est = predictor.estimate(p.nDirty, p.refresh, kb, encMs, probe)
                val ests = p.whatIf.map { (_, n) -> predictor.estimate(n, p.refresh, kb, encMs, probe) }
                wi = p.whatIf.zip(ests) { (t, n), e -> Triple(t, n, e.totalMs) }
                wic = ests.map { it.computeMs }
            }
            val ready = OffloadClient.nowMs() - t0
            ui { txtStatus.text = if (target == "compute")
                    "#${frameNo + 1} 전송 중… 서버 연산 예측 %.0f ms (예측 완료 +%.0f ms)".format(est.computeMs, ready)
                 else "#${frameNo + 1} 전송 중… 예측 도착 +%.0f ms (예측 완료 +%.0f ms)".format(est.totalMs, ready) }
            Pending(p, est, wi, wic, ready, probe, target)
        } else null

        val r = c.infer(jpeg, modeStr, th, reset)
        val tPost0 = OffloadClient.nowMs()
        val info = if (m == Mode.OUROBOROS) RecomputeVis.Info(r.json) else null
        val vis = draw(frame, r.dets, m, if (boostOn) info else null)
        val showWorld = info != null && worldOn
        // The global view runs off the critical path, so it never changes the e2e latency being
        // measured. Every frame updates the phone's anchor copy, in order; only the newest is drawn.
        if (showWorld) {
            val dets = r.dets; val boost = boostOn; val col = m.color
            worldQueued.incrementAndGet()
            visExec.execute {
                val newest = worldQueued.decrementAndGet() == 0
                try {
                    val wb = worldView.update(frame, info!!, dets,
                        org.opencv.core.Scalar(Color.red(col).toDouble(), Color.green(col).toDouble(), Color.blue(col).toDouble(), 255.0),
                        boost, render = newest)
                    if (wb != null) { lastWorld = wb; ui { if (worldOn && mode == Mode.OUROBOROS) img.setImageBitmap(wb) } }
                } catch (e: Exception) { Log.e(TAG, "world view failed", e); worldView.clear() }
            }
        }
        val tDone = OffloadClient.nowMs()
        val e2e = tDone - t0
        val clientPost = tDone - tPost0
        r.json.optInt("gop", 0).takeIf { it > 0 }?.let { mirror.gop = it }
        r.json.optString("pred_target", "").takeIf { it == "e2e" || it == "compute" }?.let { predTarget = it }

        val pend = try { fut?.get() } catch (e: Exception) { Log.e(TAG, "mirror failed", e); mirrorInSync = false; null }
        // The result is in hand: let the camera thread grab and convert the next frame while
        // this one's bookkeeping (anchor update, stats, dashboard report) finishes.
        busy.set(false)
        var affDiff = Double.NaN
        if (pend != null) {
            val sAff = r.json.optJSONArray("affine")?.let { arr ->
                DoubleArray(6) { arr.getJSONArray(it / 3).getDouble(it % 3) } }
            val sShift = r.json.optJSONArray("shift")?.let { intArrayOf(it.getInt(0), it.getInt(1)) }
            val sDmap = r.json.optString("dmap", "").takeIf { it.isNotEmpty() && it != "null" }
                ?.let { android.util.Base64.decode(it, android.util.Base64.DEFAULT) }
            val refresh = r.refresh
            mirrorExec.execute { mirror.commit(refresh, sAff, sShift, sDmap) }
            affDiff = if (sAff != null && pend.pred.affine != null)
                sAff.indices.maxOf { kotlin.math.abs(sAff[it] - pend.pred.affine[it]) } else Double.NaN
            // e2e estimates made without the GPU probe (compute target) would skew the e2e stats
            if (pend.target == "e2e") errStats.add(pend.est.totalMs, e2e, pend.pred.nDirty, r.nDirty)
            errCompute.add(pend.est.computeMs, r.computeMs, pend.pred.nDirty, r.nDirty)
        }
        if (m == Mode.OUROBOROS) synchronized(predictor) {
            val tm = r.timing
            predictor.observe(LatencyPredictor.Observation(r.nDirty, r.refresh, kb,
                preMs = tm.optDouble("decode", 0.0) + tm.optDouble("register", 0.0) + tm.optDouble("dirty", 0.0),
                vitMs = tm.optDouble("model", 0.0), postMs = tm.optDouble("post", 0.0),
                gpuWaitMs = tm.optDouble("gpu_wait", 0.0), serverProcMs = r.serverProcMs, httpMs = r.httpMs,
                uploadMs = r.uploadMs, clientPostMs = clientPost, rawWaitPredMs = pend?.est?.rawWaitMs ?: Double.NaN))
        }

        recordFrame(e2e)
        chart.add(e2e.toFloat(), r.computeMs.toFloat(),
            pend?.takeIf { it.target == "e2e" }?.est?.totalMs?.toFloat() ?: Float.NaN,
            pend?.takeIf { it.target == "compute" }?.est?.computeMs?.toFloat() ?: Float.NaN)
        val t = r.timing
        val stats = buildString {
            append(commonStats())
            append("encode %5.1f | upload %6.1f | server %6.1f | download %6.1f | draw %4.1f ms\n".format(
                encMs, r.uploadMs, r.serverProcMs, r.downloadMs, clientPost))
            append("server: dec %.1f reg %.1f dirty %.1f ViT %.1f post %.1f = %.1f, GPU대기 %.1f ms\n".format(
                t.optDouble("decode"), t.optDouble("register"), t.optDouble("dirty"), t.optDouble("model"),
                t.optDouble("post"), r.computeMs, t.optDouble("gpu_wait")))
            append("recompute %4d / 4096 patches (%4.1f%%)%s | %.1f KB | dets %d".format(
                r.nDirty, 100.0 * r.nDirty / 4096, if (r.refresh) " [refresh: ${r.json.optString("refresh_reason")}]" else "",
                kb, r.dets.size))
        }
        val predText = if (pend != null && pend.target == "compute") buildString {
            val err = r.computeMs - pend.est.computeMs
            append("[서버 연산만 예측] 예측 %.0f ms → 실제 %.0f ms (오차 %+.0f ms, %+.0f%%)\n".format(
                pend.est.computeMs, r.computeMs, err, 100 * err / r.computeMs))
            append("  예측 시점: 전송 시작 후 +%.0f ms (mirror %.1f ms) · 네트워크/GPU 대기 제외\n".format(pend.readyAtMs, pend.pred.ms))
            append("  재연산 패치: 예측 %d / 실제 %d%s\n".format(pend.pred.nDirty, r.nDirty,
                if (pend.pred.refresh != r.refresh) "  (refresh 불일치)" else ""))
            if (errCompute.count > 0) append("누적(최근 %d): MAE %.1f ms, bias %+.1f ms, ±10%% 이내 %.0f%%, ±20%% 이내 %.0f%%\n".format(
                errCompute.count, errCompute.mae(), errCompute.bias(), 100 * errCompute.within(0.1), 100 * errCompute.within(0.2)))
            if (errCompute.count > 0) append("  패치 수 예측: MAE %.1f, 정확 일치 %.0f%%\n".format(errCompute.nMae(), 100 * errCompute.nExact()))
            synchronized(predictor) { append(predictor.computeModelText()).append('\n') }
            append("what-if (이 프레임, threshold별 서버 연산):\n")
            for ((i, w) in pend.whatIfEst.withIndex()) append("  thr %3d → %4d patches → %6.0f ms%s\n".format(
                w.first, w.second, pend.whatIfCompute[i], if (w.first == th) "  ◀" else ""))
        } else if (pend != null) buildString {
            val err = e2e - pend.est.totalMs
            append("이번 프레임: 예측 %.0f ms → 실제 %.0f ms (오차 %+.0f ms, %+.0f%%)\n".format(
                pend.est.totalMs, e2e, err, 100 * err / e2e))
            append("  예측 시점: 전송 시작 후 +%.0f ms (mirror %.1f ms)\n".format(pend.readyAtMs, pend.pred.ms))
            append("  재연산 패치: 예측 %d / 실제 %d%s\n".format(pend.pred.nDirty, r.nDirty,
                if (pend.pred.refresh != r.refresh) "  (refresh 불일치)" else ""))
            append("  분해: enc %.0f + net %.0f + GPU대기 %.0f(실제 %.0f) + compute %.0f + 기타 %.0f\n".format(
                pend.est.encodeMs, pend.est.networkMs, pend.est.gpuWaitMs, r.timing.optDouble("gpu_wait", 0.0),
                pend.est.computeMs, pend.est.serverOverheadMs + pend.est.clientPostMs))
            if (errStats.count > 0) append("누적(최근 %d): MAE %.1f ms, bias %+.1f ms, ±10%% 이내 %.0f%%, ±20%% 이내 %.0f%%\n".format(
                errStats.count, errStats.mae(), errStats.bias(), 100 * errStats.within(0.1), 100 * errStats.within(0.2)))
            if (errStats.count > 0) append("  패치 수 예측: MAE %.1f, 정확 일치 %.0f%%\n".format(errStats.nMae(), 100 * errStats.nExact()))
            synchronized(predictor) { append(predictor.computeModelText()).append('\n').append(predictor.networkText()).append('\n') }
            append("what-if (이 프레임, threshold별):\n")
            for ((wt, n, ms) in pend.whatIfEst) append("  thr %3d → %4d patches → %6.0f ms%s\n".format(wt, n, ms, if (wt == th) "  ◀" else ""))
        } else if (m == Mode.OUROBOROS && predictOn) "예측: 다음 프레임부터 (서버와 anchor 동기화 중)" else ""

        Log.i(TAG, "CSV,$modeStr,$frameNo,%.1f,%.1f,%.1f,%.1f,%.1f,%d,%d,%s,%s,%s,%d".format(
            e2e, encMs, r.uploadMs, r.computeMs, r.downloadMs, r.nDirty, th, r.refresh,
            pend?.est?.totalMs?.let { "%.1f".format(it) } ?: "", pend?.pred?.nDirty?.toString() ?: "", jpeg.size))
        ui {
            if (!showWorld || lastWorld == null) img.setImageBitmap(vis)   // until the view has synced
            setStatus(e2e); txtStats.text = stats
            if (m == Mode.OUROBOROS && predictOn) txtPredict.text = predText
        }
        val rec = baseRecord(m, e2e, r.dets).put("offload", J()
            .put("encode", encMs).put("upload", r.uploadMs).put("server", r.serverProcMs).put("download", r.downloadMs)
            .put("draw", clientPost).put("compute", r.computeMs).put("timing", t)
            .put("n_dirty", r.nDirty).put("refresh", r.refresh).put("refresh_reason", r.json.opt("refresh_reason"))
            .put("thres", th).put("kb", kb))
        rec.put("predict_on", m == Mode.OUROBOROS && predictOn)
        rec.put("boost", info != null && boostOn).put("world_view", showWorld)
        if (pend != null) {
            val wi = JSONArray(); pend.whatIfEst.forEachIndexed { i, (wt, n, ms) ->
                wi.put(JSONArray().put(wt).put(n).put(ms).put(pend.whatIfCompute[i])) }
            rec.put("pred", J()
                .put("target", pend.target).put("compute_est", pend.est.computeMs)
                .put("c_mae", errCompute.mae()).put("c_bias", errCompute.bias()).put("c_w10", errCompute.within(0.1))
                .put("c_w20", errCompute.within(0.2)).put("c_count", errCompute.count)
                .put("est", pend.est.totalMs).put("ready_at", pend.readyAtMs).put("mirror_ms", pend.pred.ms)
                .put("n_pred", pend.pred.nDirty).put("refresh_pred", pend.pred.refresh)
                .put("gpu_others", pend.probe?.busyRemainingMs ?: Double.NaN).put("sched", pend.probe?.sched).put("probe_rtt", pend.probe?.rttMs ?: Double.NaN)
                .put("gpu_wait_pred", pend.est.gpuWaitMs)
                .put("parts", J().put("encode", pend.est.encodeMs).put("network", pend.est.networkMs)
                    .put("server_overhead", pend.est.serverOverheadMs).put("compute", pend.est.computeMs)
                    .put("gpu_wait", pend.est.gpuWaitMs)
                    .put("post", pend.est.clientPostMs))
                .put("mae", errStats.mae()).put("bias", errStats.bias()).put("w10", errStats.within(0.1))
                .put("w20", errStats.within(0.2)).put("n_mae", errStats.nMae()).put("n_exact", errStats.nExact())
                .put("count", errStats.count).put("what_if", wi).put("aff_diff", affDiff)
                .put("compute_model", synchronized(predictor) { predictor.computeModelText() })
                .put("network_model", synchronized(predictor) { predictor.networkText() }))
        }
        reporter?.offer(rec, if (showWorld) lastWorld ?: vis else vis)
    }

    private fun baseRecord(m: Mode, e2e: Double, dets: List<Det>): JSONObject {
        val d = JSONArray()
        dets.forEach { d.put(J().put("name", it.name).put("score", it.score.toDouble())
            .put("box", JSONArray().put(it.x0.toDouble()).put(it.y0.toDouble()).put(it.x1.toDouble()).put(it.y1.toDouble()))) }
        return J().put("mode", m.name.lowercase()).put("frame", frameNo).put("e2e", e2e)
            .put("fps", fps()).put("avg30", e2eHist.average()).put("dets", d)
            .put("device", android.os.Build.MODEL)
    }

    // ---------------------------------------------------------------- display

    private fun recordFrame(e2e: Double) {
        frameNo++
        e2eHist.addLast(e2e); while (e2eHist.size > 30) e2eHist.removeFirst()
        doneTimes.addLast(OffloadClient.nowMs()); while (doneTimes.size > 30) doneTimes.removeFirst()
    }

    private fun fps(): Double =
        if (doneTimes.size < 2) Double.NaN else 1000.0 * (doneTimes.size - 1) / (doneTimes.last() - doneTimes.first())

    private fun commonStats() = "frame #%d | e2e last %.0f ms, avg(30) %.0f ms | %.2f FPS\n".format(
        frameNo, e2eHist.last(), e2eHist.average(), fps())

    private fun setStatus(e2e: Double) {
        txtStatus.text = "%s  #%d  %.0f ms  ·  %.2f FPS".format(mode.label, frameNo, e2e, fps())
        txtStatus.setTextColor(mode.color)
    }

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 4f }
    private val labelBg = Paint().apply { style = Paint.Style.FILL }
    private val labelText = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.BLACK; textSize = 22f; isFakeBoldText = true }
    private val bannerText = Paint(Paint.ANTI_ALIAS_FLAG).apply { textSize = 26f; isFakeBoldText = true; setShadowLayer(4f, 0f, 0f, Color.BLACK) }

    /** Boxes on the very frame that was captured/sent. */
    private fun draw(frame: Bitmap, dets: List<Det>, m: Mode, boost: RecomputeVis.Info? = null): Bitmap {
        val out = frame.copy(Bitmap.Config.ARGB_8888, true)
        if (boost != null) try { RecomputeVis.boostFrame(out, boost) } catch (e: Exception) { Log.e(TAG, "boost failed", e) }
        val cv = Canvas(out); val w = out.width; val h = out.height
        boxPaint.color = m.color; labelBg.color = m.color
        for (d in dets) {
            val r = RectF(d.x0 * w, d.y0 * h, d.x1 * w, d.y1 * h)
            cv.drawRect(r, boxPaint)
            val label = "%s %.2f".format(d.name, d.score)
            val tw = labelText.measureText(label)
            val ty = if (r.top > 28) r.top else r.top + 28
            cv.drawRect(r.left, ty - 26, r.left + tw + 8, ty, labelBg)
            cv.drawText(label, r.left + 4, ty - 5, labelText)
        }
        bannerText.color = m.color
        cv.drawText(m.label, 10f, 32f, bannerText)
        return out
    }

    private fun ui(block: () -> Unit) = runOnUiThread(block)

    companion object {
        const val TAG = "OuroDemo"
        const val MODEL_NAME = "vitdet_b.onnx"
    }
}
