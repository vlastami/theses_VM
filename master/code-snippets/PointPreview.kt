package com.littletrickster.scanner

import android.content.Context
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.Preview
import androidx.camera.view.PreviewView
import androidx.compose.animation.core.animateOffsetAsState
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.text.BasicText
import androidx.compose.foundation.clickable
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import org.opencv.core.Mat
import org.opencv.core.Point
import kotlin.math.min
import kotlin.math.roundToInt

// Auto-scan state machine (top-level)
sealed class AutoScanState { object Scanning: AutoScanState(); data class Holding(val startMs: Long): AutoScanState(); object Capturing: AutoScanState() }

/**
 * Compute a normalized quality score (0.0-1.0) from QualityMetrics for best-in-window selection.
 * Higher score = better quality. Combines sharpness (VoL), stability (motion), and coverage.
 */
fun computeFrameScore(metrics: QualityMetrics, th: GateThresholds): Double {
    // Normalize VoL: 0 at threshold, 1 at 2x threshold
    val volNorm = ((metrics.varianceOfLaplacian - th.minVoLRelaxed) / (th.minVoL - th.minVoLRelaxed + 1.0)).coerceIn(0.0, 1.0)

    // Normalize motion: 1 at 0 motion, 0 at threshold
    val motionNorm = (1.0 - metrics.motionPx / th.maxMotionPx).coerceIn(0.0, 1.0)

    // Normalize coverage: 0 at threshold, 1 at 100%
    val coverageNorm = ((metrics.coveragePct - th.minCoveragePct) / (100.0 - th.minCoveragePct)).coerceIn(0.0, 1.0)

    // Penalize glare
    val glarePenalty = (1.0 - metrics.glarePct / th.maxGlarePct).coerceIn(0.0, 1.0)

    // Weighted combination: VoL most important, then motion, then coverage
    return (volNorm * 0.4 + motionNorm * 0.3 + coverageNorm * 0.15 + glarePenalty * 0.15)
}

@Composable
fun PointPreview(
    modifier: Modifier = Modifier,
    imageCaptureConfig: ImageCapture,
    onAutoCaptureRequest: () -> Unit = {},
    captureInProgress: Boolean = false,
    frameCount: Int = 0,
    maxFrames: Int = 3,
    canCaptureMore: Boolean = true,
    isVideoScanMode: Boolean = false,
    videoScanActive: Boolean = false,
    onVideoScanComplete: () -> Unit = {},
    onVideoStitchResult: (Mat, Int) -> Unit = { _, _ -> },
    onVideoFrameCountChange: (Int) -> Unit = {}
) {
    var parentSize by remember { mutableStateOf(IntSize(1, 1)) }

    var imageWidth by remember { mutableStateOf(1) }
    var imageHeight by remember { mutableStateOf(1) }

    var points by remember { mutableStateOf(emptyList<Point>()) }
    var quality by remember { mutableStateOf<QualityMetrics?>(null) }
    var prevPointsForMotion by remember { mutableStateOf<List<Point>?>(null) }

    // Auto-scan state machine
    var autoState by remember { mutableStateOf<AutoScanState>(AutoScanState.Scanning) }
    var holdStartMs by remember { mutableStateOf<Long?>(null) }
    var analyzerEnabled by remember { mutableStateOf(true) }
    val scope = rememberCoroutineScope()

    val holdForMs = 500L
    val debounceMs = 200L

    // Hysteresis configuration - loaded from SharedPreferences "scanner"
    val scannerPrefs = rememberScannerSharedPrefs()
    val consecutivePassesRequired = remember { scannerPrefs.getInt("hysteresis_passes_required", 3).coerceIn(1, 10) }
    val consecutiveFailsToExit = remember { scannerPrefs.getInt("hysteresis_fails_to_exit", 2).coerceIn(1, 10) }

    // Best-in-window configuration
    val bestFrameThreshold = remember { scannerPrefs.getFloat("best_frame_threshold", 0.95f).toDouble().coerceIn(0.5, 1.0) }
    val dwellExtensionMs = remember { scannerPrefs.getLong("dwell_extension_ms", 150L).coerceIn(0L, 500L) }

    // Hysteresis state tracking
    var consecutivePassCount by remember { mutableStateOf(0) }
    var consecutiveFailCount by remember { mutableStateOf(0) }

    // Best-in-window state tracking
    var bestScoreInDwellWindow by remember { mutableStateOf(0.0) }
    var currentFrameScore by remember { mutableStateOf(0.0) }
    var dwellExtensionUntilMs by remember { mutableStateOf(0L) }

    val context = LocalContext.current
    val continuousStitcher = remember {
        ContinuousStitcher().apply {
            debugFrameDir = java.io.File(context.filesDir, "debug_frames")
        }
    }
    var stitcherFrameCount by remember { mutableStateOf(0) }
    var stitchProcessing by remember { mutableStateOf(false) }

    DisposableEffect(Unit) {
        onDispose {
            continuousStitcher.release()
        }
    }

    LaunchedEffect(videoScanActive) {
        if (!videoScanActive && continuousStitcher.isInitialized()) {
            stitchProcessing = true
            scope.launch(Dispatchers.IO) {
                try {
                    val result = continuousStitcher.finalizeStitch()
                    val rotation = continuousStitcher.getRotation()
                    continuousStitcher.release()
                    if (result != null) {
                        onVideoStitchResult(result, rotation)
                    }
                } finally {
                    stitchProcessing = false
                    stitcherFrameCount = 0
                    onVideoFrameCountChange(0)
                }
            }
        }
    }
    val th = remember(context) {
        val sp = context.getSharedPreferences("auto_scan", Context.MODE_PRIVATE)
        ThresholdPrefs.loadFromPreferences(ThresholdPrefs.SharedPreferencesAdapter(sp))
    }

    val mScale = remember(parentSize, imageWidth, imageHeight) {
        min(
            parentSize.height.toFloat() / imageHeight.toFloat(),
            parentSize.width.toFloat() / imageWidth.toFloat()
        )
    }

    val scaledWidth = remember(mScale, imageWidth) { imageWidth * mScale }

    val scaledHeight = remember(mScale, imageHeight) { imageHeight * mScale }


    val verticalOffset =
        remember(scaledHeight, parentSize) { (parentSize.height - scaledHeight) / 2 }
    val horizontalOffset =
        remember(scaledWidth, parentSize) { (parentSize.width - scaledWidth) / 2 }



    Box(modifier = modifier
        .onSizeChanged { parentSize = it }) {

        val surfaceProvider = previewView(modifier = Modifier.fillMaxSize(),
            builder = {
                scaleType = PreviewView.ScaleType.FIT_CENTER
                implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            })

        val effectiveEnabled = analyzerEnabled && !captureInProgress
        ImageAnalyser(
            enabled = effectiveEnabled,
            imageAnalysis = remember {
                ImageAnalysis.Builder().setResolutionSelector(highResolutionSelector())
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                    .build()
            },
            imageCapture = imageCaptureConfig,
            preview = remember {
                val preview: Preview = Preview.Builder()
                    .setResolutionSelector(defaultResolutionSelector())
                    .build()

                preview.setSurfaceProvider(surfaceProvider)
                preview
            },
            analyze = {
                if (!analyzerEnabled) {
                    return@ImageAnalyser
                }
                val mat = it.yuvToMat()
                // Downscale to grayscale for quality metrics at ~640px width
                val grayDown = mat.toGrayscaleDownscaled(640)

                if (isVideoScanMode && videoScanActive) {
                    val rotation = it.imageInfo.rotationDegrees
                    if (!continuousStitcher.isInitialized()) {
                        val colorClone = mat.clone()
                        val grayClone = grayDown.clone()
                        scope.launch(Dispatchers.IO) {
                            try {
                                continuousStitcher.tryInitialize(colorClone, grayClone, rotation)
                                stitcherFrameCount = continuousStitcher.frameCount
                                onVideoFrameCountChange(stitcherFrameCount)
                            } finally {
                                colorClone.release()
                                grayClone.release()
                            }
                        }
                    } else {
                        if (continuousStitcher.hasEnoughDisplacement(grayDown)) {
                            val colorClone = mat.clone()
                            scope.launch(Dispatchers.IO) {
                                try {
                                    continuousStitcher.captureFrame(colorClone)
                                    stitcherFrameCount = continuousStitcher.frameCount
                                    onVideoFrameCountChange(stitcherFrameCount)
                                } finally {
                                    colorClone.release()
                                }
                            }
                        }
                    }
                }

                val resized = Mat()
                val scale = mat.resizeMax(resized, 300.0)
                mat.release()

                // Detect quad with edge fit
                val detection = getPointsWithEdgeFit(resized)
                val foundPoints = detection.first.toMutableList()

                // Rotate points to match display orientation
                foundPoints.rotate(
                    it.imageInfo.rotationDegrees,
                    Point(resized.width() / 2.0, resized.height() / 2.0)
                )

                // Release resized after processing
                resized.release()

                // Scale points back to original frame size
                foundPoints *= scale

                // --- Compensate for Preview cropRect: shift points and use visible (cropped) rotated dimensions ---
                run {
                    val crop = it.cropRect
                    val srcW = it.width
                    val srcH = it.height

                    // Define crop corners in buffer space (pre-rotation)
                    val cropCorners = mutableListOf(
                        Point(crop.left.toDouble(),  crop.top.toDouble()),
                        Point(crop.right.toDouble(), crop.top.toDouble()),
                        Point(crop.right.toDouble(), crop.bottom.toDouble()),
                        Point(crop.left.toDouble(),  crop.bottom.toDouble())
                    )
                    // Rotate crop to display space (same as points)
                    cropCorners.rotate(
                        it.imageInfo.rotationDegrees,
                        Point(srcW / 2.0, srcH / 2.0)
                    )
                    // Bounding box of rotated crop
                    val minX = cropCorners.minOf { p -> p.x }
                    val minY = cropCorners.minOf { p -> p.y }
                    val maxX = cropCorners.maxOf { p -> p.x }
                    val maxY = cropCorners.maxOf { p -> p.y }

                    // Shift points so origin is top-left of the visible (cropped) image in display space
                    foundPoints.forEach { p ->
                        p.x -= minX
                        p.y -= minY
                    }

                    // Use the rotated crop dimensions for overlay scaling
                    imageWidth = (maxX - minX).toInt()
                    imageHeight = (maxY - minY).toInt()
                }

                // Compute quality metrics using grayscale and current points
                val qm = computeQuadQuality(
                    grayDown,
                    foundPoints,
                    imageWidth.toDouble(),
                    imageHeight.toDouble(),
                    prevPointsForMotion
                )
                // Prefer edge fit from detection (flags full-frame fallback)
                val qmFinal = qm.copy(edgeFit = detection.second)

                // Update states
                quality = qmFinal
                prevPointsForMotion = foundPoints.toList()
                points = foundPoints

                // Auto-scan state transitions with hysteresis
                quality?.let { metrics ->
                    val pass = passPolicy(metrics, th)
                    // Share latest metrics and geometry for capture/metadata
                    AutoScanStateStore.updateFromAnalyzer(
                        metrics = metrics,
                        points = foundPoints,
                        frameWidth = imageWidth,
                        frameHeight = imageHeight,
                        passed = pass
                    )
                    val now = System.currentTimeMillis()

                    // Update hysteresis counters
                    if (pass) {
                        consecutivePassCount++
                        consecutiveFailCount = 0
                    } else {
                        consecutiveFailCount++
                        consecutivePassCount = 0
                    }

                    // Compute current frame score for best-in-window tracking
                    currentFrameScore = computeFrameScore(metrics, th)

                    when (autoState) {
                        is AutoScanState.Scanning -> {
                            // Require N consecutive passes to enter Holding (hysteresis entry)
                            if (consecutivePassCount >= consecutivePassesRequired) {
                                holdStartMs = now
                                autoState = AutoScanState.Holding(now)
                                // Initialize best-in-window tracking
                                bestScoreInDwellWindow = currentFrameScore
                                dwellExtensionUntilMs = 0L
                                // Reset counters on state transition
                                consecutivePassCount = 0
                                consecutiveFailCount = 0
                            }
                        }
                        is AutoScanState.Holding -> {
                            // Track best score during dwell window
                            if (currentFrameScore > bestScoreInDwellWindow) {
                                bestScoreInDwellWindow = currentFrameScore
                            }

                            // Require M consecutive fails to exit Holding (hysteresis exit)
                            if (consecutiveFailCount >= consecutiveFailsToExit) {
                                autoState = AutoScanState.Scanning
                                holdStartMs = null
                                bestScoreInDwellWindow = 0.0
                                dwellExtensionUntilMs = 0L
                                // Reset counters on state transition
                                consecutivePassCount = 0
                                consecutiveFailCount = 0
                            } else if (pass) {
                                val start = holdStartMs ?: now
                                val dwellElapsed = now - start

                                // Check if basic dwell time is complete
                                if (dwellElapsed >= holdForMs) {
                                    val scoreThreshold = bestScoreInDwellWindow * bestFrameThreshold
                                    val isCurrentGoodEnough = currentFrameScore >= scoreThreshold

                                    // Set extension deadline if not already set
                                    if (dwellExtensionUntilMs == 0L) {
                                        dwellExtensionUntilMs = now + dwellExtensionMs
                                    }

                                    // Capture if current frame is good enough OR extension timeout reached
                                    val shouldCapture = isCurrentGoodEnough || now >= dwellExtensionUntilMs

                                    if (shouldCapture && canCaptureMore) {
                                        autoState = AutoScanState.Capturing
                                        analyzerEnabled = false
                                        scope.launch {
                                            onAutoCaptureRequest()
                                            delay(debounceMs)
                                            analyzerEnabled = true
                                            autoState = AutoScanState.Scanning
                                            holdStartMs = null
                                            bestScoreInDwellWindow = 0.0
                                            dwellExtensionUntilMs = 0L
                                            consecutivePassCount = 0
                                            consecutiveFailCount = 0
                                        }
                                    }
                                }
                            }
                        }
                        is AutoScanState.Capturing -> {
                            // wait until debounce completes
                        }
                    }

                }

                // Release gray
                grayDown.release()
            })

        repeat(4) {
            SimpleTargetCircle(
                getOffset = { points.getOrNull(it)?.toOffset() },
                horizontalOffset = horizontalOffset,
                verticalOffset = verticalOffset,
                scale = mScale
            )
        }

        val displayFrameCount = if (isVideoScanMode && videoScanActive) stitcherFrameCount else frameCount
        if (displayFrameCount > 0) {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.TopEnd) {
                BasicText(
                    text = if (isVideoScanMode && stitchProcessing) "Processing..." else if (isVideoScanMode) "$displayFrameCount captured" else "$displayFrameCount/$maxFrames",
                    modifier = Modifier
                        .padding(top = 16.dp, end = 16.dp)
                        .background(Color(0xFF4CAF50), shape = CircleShape)
                        .padding(horizontal = 12.dp, vertical = 6.dp),
                    style = TextStyle(color = Color.White, fontSize = 18.sp, fontWeight = FontWeight.Bold)
                )
            }
        }

        if (!canCaptureMore) {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                BasicText(
                    text = "Max frames reached\nTap Done to continue",
                    modifier = Modifier
                        .background(Color(0, 0, 0, 200), shape = CircleShape)
                        .padding(horizontal = 24.dp, vertical = 16.dp),
                    style = TextStyle(color = Color.White, fontSize = 18.sp, fontWeight = FontWeight.Bold)
                )
            }
        }

        val qmLocal = quality
        if (qmLocal != null) {
            val thresholds = th
            if (!captureInProgress && canCaptureMore) {
                Box(Modifier.fillMaxSize(), contentAlignment = Alignment.TopCenter) {
                    val tip = if (isVideoScanMode && videoScanActive) {
                        if (stitcherFrameCount == 0) "Hold steady to start"
                        else "Pan slowly downward"
                    } else if (isVideoScanMode && stitchProcessing) {
                        "Stitching $stitcherFrameCount frames..."
                    } else {
                        tipFor(qmLocal, thresholds, autoState)
                    }
                    BasicText(
                        text = tip,
                        modifier = Modifier
                            .padding(top = 16.dp)
                            .background(Color(0, 0, 0, 160), shape = CircleShape)
                            .padding(horizontal = 12.dp, vertical = 6.dp),
                        style = TextStyle(color = Color.White, fontSize = 20.sp)
                    )
                }
                Box(Modifier.fillMaxSize(), contentAlignment = Alignment.BottomCenter) {
                    GateRow(
                        metrics = qmLocal,
                        th = thresholds,
                        autoState = autoState,
                        holdStartMs = holdStartMs,
                        holdForMs = holdForMs
                    )
                }
            }
        }


    }
}

@Composable
private fun SimpleTargetCircle(
    getOffset: () -> Offset?,
    horizontalOffset: Float,
    verticalOffset: Float,
    scale: Float,
    circleColor: Color = Color.Green,
) {

    val offset = getOffset() ?: return


    val animatedOffset by animateOffsetAsState(
        targetValue = offset,
        animationSpec = offsetAnim
    )


    Box(
        Modifier
            .offset {
                IntOffset(
                    (horizontalOffset + animatedOffset.x * scale - 20.dp.toPx()).roundToInt(),
                    (verticalOffset + animatedOffset.y * scale - 20.dp.toPx()).roundToInt()
                )
            }

            .background(Color(255, 255, 255, 40), shape = CircleShape)
            .size(37.dp)
            .border(3.dp, circleColor, shape = CircleShape))
}

// ---- Guidance overlay helpers ----
private fun tipFor(metrics: QualityMetrics, th: GateThresholds, state: AutoScanState): String {
    if (state is AutoScanState.Capturing) return "Capturing..."
    val pass = passPolicy(metrics, th)
    if (pass) {
        return if (state is AutoScanState.Holding) "Hold steady" else "Ready"
    }
    val hasQuad = metrics.edgeFit.hasQuad
    val aspectOk = metrics.edgeFit.aspectRatio in th.minAspect..th.maxAspect
    val exposureOk = metrics.lowTailPct <= th.maxLowTailPct && metrics.highTailPct <= th.maxHighTailPct
    val glareOk = metrics.glarePct <= th.maxGlarePct
    val skewOk = metrics.skewDeg <= th.maxSkewDeg
    val coverageOk = metrics.coveragePct >= th.minCoveragePct
    val motionOk = metrics.motionPx <= th.maxMotionPx
    val volStrong = metrics.varianceOfLaplacian >= th.minVoL

    return when {
        !hasQuad -> "Find document"
        !coverageOk -> "Move closer"
        !motionOk -> "Hold steady"
        !volStrong -> "Hold steady"
        !glareOk -> "Avoid glare"
        !exposureOk -> "Better light"
        !skewOk || !aspectOk -> "Align edges"
        else -> "Adjust"
    }
}


@Composable
private fun GateRow(
    metrics: QualityMetrics,
    th: GateThresholds,
    autoState: AutoScanState,
    holdStartMs: Long?,
    holdForMs: Long
) {
    val volOk = metrics.varianceOfLaplacian >= th.minVoL
    val exposureOk = metrics.lowTailPct <= th.maxLowTailPct && metrics.highTailPct <= th.maxHighTailPct
    val glareOk = metrics.glarePct <= th.maxGlarePct
    val skewOk = metrics.skewDeg <= th.maxSkewDeg
    val coverageOk = metrics.coveragePct >= th.minCoveragePct
    val motionOk = metrics.motionPx <= th.maxMotionPx
    val quadOk = metrics.edgeFit.hasQuad
    val aspectOk = metrics.edgeFit.aspectRatio in th.minAspect..th.maxAspect

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(bottom = 16.dp)
            .padding(horizontal = 16.dp)
            .background(Color(0, 0, 0, 120), shape = CircleShape)
            .padding(horizontal = 12.dp, vertical = 8.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly, verticalAlignment = Alignment.CenterVertically) {
            Dot("Focus", volOk)
            Dot("Exposure", exposureOk)
            Dot("Glare", glareOk)
            Dot("Skew", skewOk)
        }
        Spacer(Modifier.height(6.dp))
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly, verticalAlignment = Alignment.CenterVertically) {
            Dot("Coverage", coverageOk)
            Dot("Motion", motionOk)
            Dot("Quad", quadOk)
            Dot("Aspect", aspectOk)
        }
        if (autoState is AutoScanState.Holding && holdStartMs != null) {
            val now = System.currentTimeMillis()
            val p = ((now - holdStartMs).toFloat() / holdForMs.toFloat()).coerceIn(0f, 1f)
            Spacer(Modifier.height(6.dp))
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(4.dp)
                    .background(Color(80, 80, 80, 200), CircleShape)
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth(p)
                        .height(4.dp)
                        .background(Color(0xFF, 0xC1, 0x07, 0xFF), CircleShape)
                )
            }
        }
    }
}

@Composable
private fun Dot(label: String, ok: Boolean) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(
            modifier = Modifier
                .size(10.dp)
                .background(if (ok) Color(0xFF4CAF50) else Color(0xFFF44336), CircleShape)
        )
        Spacer(Modifier.width(4.dp))
        BasicText(text = label, style = TextStyle(color = Color.White))
    }
}

@Composable
private fun ThresholdsDebugOverlay(th: GateThresholds) {
    Column(
        modifier = Modifier
            .background(Color(0, 0, 0, 200), shape = CircleShape)
            .padding(horizontal = 12.dp, vertical = 10.dp)
    ) {
        BasicText(text = "Thresholds", style = TextStyle(color = Color.White))
        Spacer(Modifier.height(6.dp))
        // List key thresholds compactly
        BasicText(text = "VoL ≥ ${formatD(th.minVoL)} (relaxed ${formatD(th.minVoLRelaxed)})", style = TextStyle(color = Color.White))
        BasicText(text = "Exposure tails ≤ L:${formatD(th.maxLowTailPct)}% H:${formatD(th.maxHighTailPct)}%", style = TextStyle(color = Color.White))
        BasicText(text = "Glare ≤ ${formatD(th.maxGlarePct)}%", style = TextStyle(color = Color.White))
        BasicText(text = "Skew ≤ ${formatD(th.maxSkewDeg)}°", style = TextStyle(color = Color.White))
        BasicText(text = "Coverage ≥ ${formatD(th.minCoveragePct)}%", style = TextStyle(color = Color.White))
        BasicText(text = "Motion ≤ ${formatD(th.maxMotionPx)} px", style = TextStyle(color = Color.White))
        BasicText(text = "Aspect ∈ [${formatD(th.minAspect)}, ${formatD(th.maxAspect)}]", style = TextStyle(color = Color.White))
    }
}

private fun formatD(d: Double): String =
    if (d % 1.0 == 0.0) d.toInt().toString() else String.format("%.1f", d)