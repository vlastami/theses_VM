package com.littletrickster.scanner

import android.graphics.Bitmap
import android.util.Log
import com.OctopodSystems.CheckChecker.BuildConfig
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

// Data classes for quality metrics and gate results

/** Dev logging helper for auto-scan gates (enabled only when BuildConfig.DEBUG and explicitly toggled). */
object AutoScanLog {
    @Volatile private var enabled: Boolean = false
    private const val TAG = "AutoScan"

    fun setEnabled(e: Boolean) { enabled = e }
    // Only enabled when explicitly toggled AND on debug builds
    fun isEnabled(): Boolean = enabled && BuildConfig.DEBUG

    fun d(msg: String) {
        if (!isEnabled()) return
        try { Log.d(TAG, msg) } catch (_: Throwable) {}
        // Avoid printing to stdout in production builds
        if (BuildConfig.DEBUG) {
            try { println("$TAG: $msg") } catch (_: Throwable) {}
        }
    }
}

data class EdgeFit(
    val hasQuad: Boolean,
    val aspectRatio: Double,
    val edgeConfidence: Double // 0..1 where 1 means perfect right angles
)

/**
 * Extension: return a grayscale (CV_8UC1) Mat downscaled to maxWidth while preserving aspect ratio.
 * Caller owns the returned Mat and must release it.
 */
fun Mat.toGrayscaleDownscaled(maxWidth: Int = 640): Mat {
    val src = this
    val tmpGray = Mat()
    try {
        if (src.channels() == 1) {
            // Already grayscale; clone to avoid mutating input
            src.copyTo(tmpGray)
        } else if (src.channels() == 3) {
            Imgproc.cvtColor(src, tmpGray, Imgproc.COLOR_RGB2GRAY)
        } else if (src.channels() == 4) {
            Imgproc.cvtColor(src, tmpGray, Imgproc.COLOR_RGBA2GRAY)
        } else {
            // Fallback: try convert assuming RGB
            Imgproc.cvtColor(src, tmpGray, Imgproc.COLOR_RGB2GRAY)
        }

        val w = tmpGray.width()
        val h = tmpGray.height()
        if (w <= 0 || h <= 0) return tmpGray

        if (w <= maxWidth) {
            return tmpGray
        }
        val scale = maxWidth.toDouble() / w.toDouble()
        val newSize = Size((w * scale).toInt().toDouble(), (h * scale).toInt().toDouble())
        val dst = Mat()
        Imgproc.resize(tmpGray, dst, newSize, 0.0, 0.0, Imgproc.INTER_AREA)
        tmpGray.release()
        return dst
    } catch (e: Throwable) {
        // In case of failure, return what we have
        return tmpGray
    }
}

/**
 * Defines the set of thresholds used to determine if a detected document image is of high enough
 * quality to be automatically captured. Each field corresponds to a specific quality metric.
 *
 * @property minVoL The minimum acceptable Variance of Laplacian. This is a measure of image sharpness.
 *           A higher value means a sharper, more in-focus image. Used to reject blurry photos.
 * @property minVoLRelaxed A more lenient sharpness threshold that is only applied if all other
 *           quality metrics are strong. This prevents rejecting an otherwise perfect shot that is
 *           only slightly soft.
 * @property maxLowTailPct The maximum allowed percentage of pixels that are in the 'dark tail' of
 *           the histogram (i.e., almost pure black). Used to detect underexposed images.
 * @property maxHighTailPct The maximum allowed percentage of pixels that are in the 'bright tail'
 *           of the histogram (i.e., almost pure white). Used to detect overexposed images.
 * @property maxGlarePct The maximum allowed percentage of the document area that is considered to be
 *           specular glare (e.g., from a light source reflecting off the page). This is a more
 *           aggressive check for bright spots than `maxHighTailPct`.
 * @property maxSkewDeg The maximum allowed skew angle of the document in degrees. This ensures the
 *           document is captured from a relatively straight-on angle, minimizing perspective distortion.
 *           A value of 0 means the document edges are perfectly parallel to the camera frame.
 * @property minCoveragePct The minimum percentage of the camera view that the document must occupy.
 *           This ensures the user is close enough to the document to get a high-resolution image.
 *           This check can be bypassed if the document meets `minHeightCoveragePct`.
 * @property minHeightCoveragePct The minimum percentage of the camera view's height that the document
 *           must occupy. This is an alternative to `minCoveragePct` for long, narrow documents like
 *           receipts, which may not cover a large area but can span the full height of the view.
 * @property maxMotionPx The maximum average motion of the document's corners between frames, measured
 *           in pixels. This ensures the camera is being held steady, preventing motion blur.
 * @property minAspect The minimum aspect ratio (width/height or height/width) of the detected document
 *           quad. This helps filter out false detections that are not shaped like typical documents.
 * @property maxAspect The maximum aspect ratio of the detected document quad. Along with `minAspect`,
 *           this defines the range of acceptable document shapes.
 */
data class GateThresholds(
    val minVoL: Double = 100.0,
    val minVoLRelaxed: Double = 60.0,
    val maxLowTailPct: Double = 5.0,
    val maxHighTailPct: Double = 5.0,
    val maxGlarePct: Double = 8.0,
    val maxSkewDeg: Double = 12.0,
    val minCoveragePct: Double = 10.0,
    val minHeightCoveragePct: Double = 75.0,
    val maxMotionPx: Double = 16.0,
    val minAspect: Double = 1.5,
    val maxAspect: Double = 12.0
)

/**
 * Evaluate whether current QualityMetrics passes gate thresholds.
 * Allows a small slack: if all other gates are strong, accept VoL down to minVoLRelaxed.
 */
fun passPolicy(metrics: QualityMetrics, th: GateThresholds = GateThresholds()): Boolean {
    val edge = metrics.edgeFit
    val hasQuad = edge.hasQuad
    val aspectOk = edge.aspectRatio in th.minAspect..th.maxAspect
    val glareOk = metrics.glarePct <= th.maxGlarePct
    val exposureOk = metrics.lowTailPct <= th.maxLowTailPct && metrics.highTailPct <= th.maxHighTailPct
    val skewOk = metrics.skewDeg <= th.maxSkewDeg
    // New logic: pass if either area coverage is good OR height coverage is good (for long receipts)
    val sizeOk = (metrics.coveragePct >= th.minCoveragePct) || (metrics.heightCoveragePct >= th.minHeightCoveragePct)
    val motionOk = metrics.motionPx <= th.maxMotionPx

    val volOk = metrics.varianceOfLaplacian >= th.minVoL
    val volRelaxOk = metrics.varianceOfLaplacian >= th.minVoLRelaxed

    val othersStrong = hasQuad && aspectOk && glareOk && exposureOk && skewOk && sizeOk && motionOk
    val pass = if (volOk) othersStrong else (volRelaxOk && othersStrong)

    // Dev log with raw values and decisions
    if (AutoScanLog.isEnabled()) {
        AutoScanLog.d(
            "gates pass=$pass | " +
            "vol=${format(metrics.varianceOfLaplacian)} (min=${format(th.minVoL)}, relax=${format(th.minVoLRelaxed)}) ok=$volOk relaxOk=$volRelaxOk | " +
            "expo L=${format(metrics.lowTailPct)}%≤${format(th.maxLowTailPct)} H=${format(metrics.highTailPct)}%≤${format(th.maxHighTailPct)} ok=$exposureOk | " +
            "glare=${format(metrics.glarePct)}%≤${format(th.maxGlarePct)} ok=$glareOk | " +
            "skew=${format(metrics.skewDeg)}°≤${format(th.maxSkewDeg)} ok=$skewOk | " +
            "size: area=${format(metrics.coveragePct)}%≥${format(th.minCoveragePct)} OR height=${format(metrics.heightCoveragePct)}%≥${format(th.minHeightCoveragePct)} ok=$sizeOk | " +
            "motion=${format(metrics.motionPx)}≤${format(th.maxMotionPx)} ok=$motionOk | " +
            "quad=$hasQuad aspect=${format(edge.aspectRatio)}∈[${format(th.minAspect)},${format(th.maxAspect)}] ok=$aspectOk"
        )
    }

    return pass
}

fun passPolicyScan(metrics: QualityMetrics, th: GateThresholds = GateThresholds()): Boolean {
    val glareOk = metrics.glarePct <= th.maxGlarePct
    val exposureOk = metrics.lowTailPct <= th.maxLowTailPct && metrics.highTailPct <= th.maxHighTailPct
    val skewOk = metrics.skewDeg <= th.maxSkewDeg
    val motionOk = metrics.motionPx <= th.maxMotionPx

    val volOk = metrics.varianceOfLaplacian >= th.minVoL
    val volRelaxOk = metrics.varianceOfLaplacian >= th.minVoLRelaxed

    val othersStrong = glareOk && exposureOk && skewOk && motionOk
    return if (volOk) othersStrong else (volRelaxOk && othersStrong)
}

data class QualityMetrics(
    val varianceOfLaplacian: Double,
    val lowTailPct: Double,
    val highTailPct: Double,
    val glarePct: Double,
    val skewDeg: Double,
    val coveragePct: Double,
    val heightCoveragePct: Double,
    val motionPx: Double,
    val edgeFit: EdgeFit
)

/**
 * Compute variance of Laplacian (focus metric) on a grayscale Mat (CV_8UC1).
 */
fun computeVarianceOfLaplacian(gray: Mat): Double {
    val lap = Mat()
    val mean = MatOfDouble()
    val std = MatOfDouble()
    try {
        Imgproc.Laplacian(gray, lap, CvType.CV_64F)
        Core.meanStdDev(lap, mean, std)
        val s = std.toArray().firstOrNull() ?: 0.0
        return s * s
    } finally {
        lap.release()
        mean.release()
        std.release()
    }
}

/**
 * Compute percentage of pixels in low and high histogram tails on grayscale image.
 * lowRange: [0, lowCut), highRange: (highCut, 255]
 */
fun computeHistogramTails(gray: Mat, lowCut: Int = 10, highCut: Int = 245): Pair<Double, Double> {
    require(gray.channels() == 1) { "computeHistogramTails expects CV_8UC1 mat" }
    val total = gray.rows().toLong() * gray.cols().toLong()
    if (total <= 0L) return 0.0 to 0.0
    val data = ByteArray(total.toInt())
    gray.get(0, 0, data)
    var low = 0L
    var high = 0L
    val lc = lowCut.coerceIn(0, 255)
    val hc = highCut.coerceIn(0, 255)
    for (b in data) {
        val v = b.toInt() and 0xFF
        if (v < lc) low++
        if (v > hc) high++
    }
    return (low * 100.0 / total) to (high * 100.0 / total)
}

/**
 * Simple glare estimator: percent of pixels above threshold.
 */
fun computeGlarePct(gray: Mat, threshold: Int = 250): Double {
    require(gray.channels() == 1) { "computeGlarePct expects CV_8UC1 mat" }
    val total = gray.rows().toLong() * gray.cols().toLong()
    if (total <= 0L) return 0.0
    val data = ByteArray(total.toInt())
    gray.get(0, 0, data)
    var bright = 0L
    val th = threshold.coerceIn(0, 255)
    for (b in data) {
        val v = b.toInt() and 0xFF
        if (v >= th) bright++
    }
    return bright * 100.0 / total
}

/**
 * Compute aspect ratio and right-angle confidence from quad points (TL, TR, BR, BL order expected).
 * Confidence = 1 - maxAbsCosine of interior angles. 1 means perfect rectangle.
 */
fun computeEdgeFit(points: List<Point>): EdgeFit {
    if (points.size != 4) return EdgeFit(false, 0.0, 0.0)
    val tl = points[0]
    val tr = points[1]
    val br = points[2]
    val bl = points[3]

    fun dist(a: Point, b: Point): Double = hypot(a.x - b.x, a.y - b.y)

    val wTop = dist(tl, tr)
    val wBot = dist(bl, br)
    val hLeft = dist(tl, bl)
    val hRight = dist(tr, br)
    val width = max(wTop, wBot)
    val height = max(hLeft, hRight)
    val aspect = if (width > 0 && height > 0) max(width, height) / max(1e-6, minOf(width, height)) else 0.0

    fun cosAngle(a: Point, b: Point, c: Point): Double {
        // angle at b formed by a-b and c-b
        val v1x = a.x - b.x
        val v1y = a.y - b.y
        val v2x = c.x - b.x
        val v2y = c.y - b.y
        val dot = v1x * v2x + v1y * v2y
        val n1 = hypot(v1x, v1y)
        val n2 = hypot(v2x, v2y)
        val denom = (n1 * n2).coerceAtLeast(1e-9)
        return dot / denom
    }

    val cos1 = abs(cosAngle(tr, tl, bl))
    val cos2 = abs(cosAngle(tl, tr, br))
    val cos3 = abs(cosAngle(tr, br, bl))
    val cos4 = abs(cosAngle(tl, bl, br))
    val maxAbsCos = listOf(cos1, cos2, cos3, cos4).maxOrNull() ?: 1.0
    val confidence = (1.0 - maxAbsCos).coerceIn(0.0, 1.0)

    return EdgeFit(true, aspect, confidence)
}

/**
 * Compute skew in degrees as the minimal tilt of horizontal edges relative to the nearest axis.
 * Returns values in [0, 45], so both landscape and portrait documents can be "aligned" when edges are parallel
 * to either the horizontal or vertical axes. Uses average of top and bottom edge skews.
 */
fun computeSkewDeg(points: List<Point>): Double {
    if (points.size != 4) return 0.0
    val tl = points[0]
    val tr = points[1]
    val br = points[2]
    val bl = points[3]

    fun deg(a: Point, b: Point): Double {
        val ang = atan2(b.y - a.y, b.x - a.x)
        return Math.toDegrees(ang)
    }
    fun skewToNearestAxis(angleDeg: Double): Double {
        // Normalize angle into [0, 180)
        val a = ((angleDeg % 180.0) + 180.0) % 180.0
        // Distance to nearest multiple of 90°
        val mod90 = a % 90.0
        return min(mod90, 90.0 - mod90)
    }

    val topSkew = skewToNearestAxis(deg(tl, tr))
    val botSkew = skewToNearestAxis(deg(bl, br))
    return (topSkew + botSkew) / 2.0
}

/**
 * Compute coverage = quad area / frame area in percent.
 */
fun computeCoveragePct(points: List<Point>, frameWidth: Double, frameHeight: Double): Double {
    if (points.size != 4) return 0.0
    val mop = MatOfPoint()
    try {
        mop.fromList(points)
        val area = Imgproc.contourArea(mop)
        val frameArea = frameWidth * frameHeight
        if (frameArea <= 0) return 0.0
        return (area * 100.0) / frameArea
    } finally {
        mop.release()
    }
}

/**
 * Compute height coverage = quad bounding box height / frame height in percent.
 */
fun computeHeightCoveragePct(points: List<Point>, frameHeight: Double): Double {
    if (points.size != 4 || frameHeight <= 0) return 0.0
    val minY = points.minOfOrNull { it.y } ?: 0.0
    val maxY = points.maxOfOrNull { it.y } ?: 0.0
    val quadHeight = maxY - minY
    return (quadHeight * 100.0) / frameHeight
}

/**
 * Compute motion in pixels between two quads as average per-corner distance.
 */
fun computeMotionPx(prev: List<Point>?, curr: List<Point>?): Double {
    if (prev == null || curr == null || prev.size != 4 || curr.size != 4) return 0.0
    fun dist(a: Point, b: Point): Double = hypot(a.x - b.x, a.y - b.y)
    var sum = 0.0
    for (i in 0 until 4) sum += dist(prev[i], curr[i])
    return sum / 4.0
}


/**
 * Ring buffer for recent quads with timestamps to compute stability within a window.
 */
data class TimedQuad(val points: List<Point>, val timestampMs: Long)

class QuadHistory(private val capacity: Int = 32, private val windowMs: Long = 500) {
    private val deque: ArrayDeque<TimedQuad> = ArrayDeque()

    fun add(points: List<Point>, nowMs: Long) {
        if (points.size != 4) return
        // copy points to avoid external mutation
        val copy = points.map { Point(it.x, it.y) }
        deque.addLast(TimedQuad(copy, nowMs))
        trim(nowMs)
        while (deque.size > capacity) deque.removeFirst()
    }

    private fun trim(nowMs: Long) {
        while (deque.isNotEmpty()) {
            val oldest = deque.first()
            if (nowMs - oldest.timestampMs > windowMs) {
                deque.removeFirst()
            } else break
        }
    }

    fun size(): Int = deque.size

    fun last(): List<Point>? = deque.lastOrNull()?.points

    fun oldestWithinWindow(nowMs: Long): List<Point>? {
        trim(nowMs)
        return deque.firstOrNull()?.points
    }

    /**
     * Average per-corner motion between the oldest sample still in window and the latest sample.
     */
    fun motionFromOldest(nowMs: Long): Double {
        val oldest = oldestWithinWindow(nowMs)
        val latest = last()
        return computeMotionPx(oldest, latest)
    }
}

/**
 * Convenience function to package core metrics when a quad is known.
 */
fun computeQuadQuality(
    gray: Mat?,
    points: List<Point>,
    frameWidth: Double,
    frameHeight: Double,
    prevPoints: List<Point>? = null
): QualityMetrics {
    val vol = gray?.let { computeVarianceOfLaplacian(it) } ?: 0.0
    val (lowTail, highTail) = gray?.let { computeHistogramTails(it) } ?: (0.0 to 0.0)
    val glare = gray?.let { computeGlarePct(it) } ?: 0.0
    val edge = computeEdgeFit(points)
    val skew = computeSkewDeg(points)
    val coverage = computeCoveragePct(points, frameWidth, frameHeight)
    val heightCoverage = computeHeightCoveragePct(points, frameHeight)
    val motion = computeMotionPx(prevPoints, points)
    return QualityMetrics(vol, lowTail, highTail, glare, skew, coverage, heightCoverage, motion, edge)
}

/**
 * Helper to detect if the returned points are the full-frame fallback rectangle.
 */
fun isFullFrameFallback(points: List<Point>, frameWidth: Double, frameHeight: Double): Boolean {
    if (points.size != 4) return false
    val expected = setOf(
        Point(0.0, 0.0),
        Point(frameWidth, 0.0),
        Point(0.0, frameHeight),
        Point(frameWidth, frameHeight)
    )
    return points.toSet() == expected
}

/**
 * Post-capture blur check on the final high-resolution Bitmap.
 * Converts Bitmap -> Mat -> Grayscale -> downscaled, then computes VoL.
 * @return true if the image is blurry (VoL is below the threshold).
 */
fun isPostCaptureBlurry(bitmap: Bitmap, minVoL: Double): Boolean {
    val mat = Mat()
    val gray = Mat()
    val grayDown = Mat()
    var isBlurry = true // Assume blurry by default
    try {
        // 1. Convert Bitmap to Mat
        val bmp32 = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        Utils.bitmapToMat(bmp32, mat)
        bmp32.recycle()

        // 2. Convert to grayscale
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_RGBA2GRAY)

        // 3. Downscale for performance (blur detection doesn't need full res)
        val scale = 1280.0 / gray.width().toDouble()
        val newSize = Size((gray.width() * scale), (gray.height() * scale))
        Imgproc.resize(gray, grayDown, newSize, 0.0, 0.0, Imgproc.INTER_AREA)

        // 4. Compute sharpness (Variance of Laplacian)
        val vol = computeVarianceOfLaplacian(grayDown)

        // 5. Compare to threshold
        isBlurry = vol < minVoL

        if (AutoScanLog.isEnabled()) {
            AutoScanLog.d("Post-capture blur check: VoL=${format(vol)}, minVoL=${format(minVoL)}, blurry=$isBlurry")
        }

    } finally {
        // 6. Release all Mats
        mat.release()
        gray.release()
        grayDown.release()
    }
    return isBlurry
}


/** Simple formatter for compact numeric output in logs. */
private fun format(d: Double): String =
    if (d % 1.0 == 0.0) d.toInt().toString() else String.format("%.1f", d)
