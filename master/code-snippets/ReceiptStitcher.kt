package com.littletrickster.scanner

import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.features2d.BFMatcher
import org.opencv.features2d.SIFT
import org.opencv.imgproc.Imgproc
import android.util.Log
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

data class StitchResult(
    val stitchedMat: Mat,
    val success: Boolean,
    val method: String
)

fun stitchReceipts(frames: List<Mat>): StitchResult {
    if (frames.isEmpty()) throw IllegalArgumentException("No frames to stitch")
    if (frames.size == 1) return StitchResult(frames[0].clone(), true, "single")

    val normalized = normalizeWidth(frames)
    Log.d("Stitcher", "stitchReceipts: ${frames.size} frames, width=${normalized[0].cols()}px")

    var result = normalized[0].clone()
    var success = true

    try {
        for (i in 1 until normalized.size) {
            Log.d("Stitcher", "Pair ${i - 1}->$i:")
            val stitched = stitchPair(result, normalized[i])
            if (stitched != null) {
                val prev = result
                result = stitched
                prev.release()
                Log.d("Stitcher", "  Result: ${result.cols()}x${result.rows()}")
            } else {
                Log.d("Stitcher", "  FAILED - keeping current result")
                success = false
            }
        }
    } finally {
        normalized.forEach { it.release() }
    }

    Log.d("Stitcher", "Stitch complete: ${result.cols()}x${result.rows()} success=$success")
    return StitchResult(result, success, "sift-homography")
}

private fun getSiftMatches(imgTop: Mat, imgBottom: Mat): Pair<MatOfPoint2f, MatOfPoint2f>? {
    val grayTop = Mat()
    val grayBottom = Mat()
    toGray(imgTop, grayTop)
    toGray(imgBottom, grayBottom)

    try {
        val h1 = grayTop.rows()
        val h2 = grayBottom.rows()
        val roiStartY = h1 / 3
        val roi2H = 2 * h2 / 3

        val roi1 = Mat(grayTop, Rect(0, roiStartY, grayTop.cols(), h1 - roiStartY))
        val roi2 = Mat(grayBottom, Rect(0, 0, grayBottom.cols(), roi2H))

        val sift = SIFT.create(5000)
        val kp1 = MatOfKeyPoint()
        val kp2 = MatOfKeyPoint()
        val desc1 = Mat()
        val desc2 = Mat()
        val emptyMask = Mat()

        sift.detectAndCompute(roi1, emptyMask, kp1, desc1)
        sift.detectAndCompute(roi2, emptyMask, kp2, desc2)
        emptyMask.release()

        Log.d("Stitcher", "  SIFT: top=${desc1.rows()} bottom=${desc2.rows()}")

        if (desc1.empty() || desc2.empty() || desc1.rows() < 2 || desc2.rows() < 2) {
            desc1.release(); desc2.release()
            kp1.release(); kp2.release()
            return null
        }

        val kp1Array = kp1.toArray()
        for (kp in kp1Array) {
            kp.pt.y += roiStartY
        }
        val kp2Array = kp2.toArray()

        val bf = BFMatcher.create()
        val knnMatches = ArrayList<MatOfDMatch>()
        bf.knnMatch(desc1, desc2, knnMatches, 2)
        desc1.release(); desc2.release()

        val good = knnMatches.filter { matchPair ->
            val matches = matchPair.toArray()
            matches.size >= 2 && matches[0].distance < 0.6f * matches[1].distance
        }.map { it.toArray()[0] }

        Log.d("Stitcher", "  SIFT: goodMatches=${good.size}")

        if (good.size < 10) {
            kp1.release(); kp2.release()
            return null
        }

        val srcPts = MatOfPoint2f()
        val dstPts = MatOfPoint2f()
        srcPts.fromList(good.map { Point(kp1Array[it.queryIdx].pt.x, kp1Array[it.queryIdx].pt.y) })
        dstPts.fromList(good.map { Point(kp2Array[it.trainIdx].pt.x, kp2Array[it.trainIdx].pt.y) })

        kp1.release(); kp2.release()
        return Pair(srcPts, dstPts)
    } finally {
        grayTop.release()
        grayBottom.release()
    }
}

private fun stitchPair(imgTop: Mat, imgBottom: Mat): Mat? {
    val matchResult = getSiftMatches(imgTop, imgBottom) ?: return null
    val (srcPts, dstPts) = matchResult

    try {
        val mask = Mat()
        val H = Calib3d.findHomography(dstPts, srcPts, Calib3d.RANSAC, 3.0, mask)

        if (H.empty()) {
            Log.d("Stitcher", "  Homography failed")
            mask.release()
            return null
        }

        val inlierCount = Core.countNonZero(mask)
        Log.d("Stitcher", "  Homography: inliers=$inlierCount/${srcPts.rows()}")
        mask.release()

        val h1 = imgTop.rows()
        val w1 = imgTop.cols()
        val h2 = imgBottom.rows()
        val w2 = imgBottom.cols()

        val corners = MatOfPoint2f(
            Point(0.0, 0.0), Point(w2.toDouble(), 0.0),
            Point(w2.toDouble(), h2.toDouble()), Point(0.0, h2.toDouble())
        )
        val warpedCorners = MatOfPoint2f()
        Core.perspectiveTransform(corners, warpedCorners, H)
        corners.release()

        val warpedPts = warpedCorners.toArray()
        val topCorners = arrayOf(
            Point(0.0, 0.0), Point(w1.toDouble(), 0.0),
            Point(w1.toDouble(), h1.toDouble()), Point(0.0, h1.toDouble())
        )
        val allPts = warpedPts + topCorners

        var xMin = allPts.minOf { it.x }.toInt()
        var yMin = allPts.minOf { it.y }.toInt()
        val xMax = allPts.maxOf { it.x }.toInt()
        val yMax = allPts.maxOf { it.y }.toInt()
        xMin = min(xMin, 0)
        yMin = min(yMin, 0)
        warpedCorners.release()

        val outW = xMax - xMin
        val outH = yMax - yMin

        if (outW > w1 * 3 || outH > h1 + h2 * 2 || outW <= 0 || outH <= 0) {
            Log.d("Stitcher", "  Canvas size out of range: ${outW}x${outH}, skipping")
            H.release()
            return null
        }

        val translate = Mat.eye(3, 3, CvType.CV_64FC1)
        translate.put(0, 2, (-xMin).toDouble())
        translate.put(1, 2, (-yMin).toDouble())

        val M = Mat()
        val zeroMat = Mat.zeros(3, 3, CvType.CV_64FC1)
        Core.gemm(translate, H, 1.0, zeroMat, 0.0, M)
        H.release()
        translate.release()
        zeroMat.release()

        val warpedBottom = Mat()
        Imgproc.warpPerspective(imgBottom, warpedBottom, M, Size(outW.toDouble(), outH.toDouble()))
        M.release()

        val warpedGray = Mat()
        toGray(warpedBottom, warpedGray)
        val bottomMask = Mat()
        Imgproc.threshold(warpedGray, bottomMask, 0.0, 255.0, Imgproc.THRESH_BINARY)
        warpedGray.release()

        val tx = -xMin
        val ty = -yMin

        val canvas = Mat.zeros(outH, outW, imgTop.type())
        val topMask = Mat.zeros(outH, outW, CvType.CV_8UC1)

        if (ty + h1 <= outH && tx + w1 <= outW && ty >= 0 && tx >= 0) {
            imgTop.copyTo(Mat(canvas, Rect(tx, ty, w1, h1)))
            Mat(topMask, Rect(tx, ty, w1, h1)).setTo(Scalar(255.0))
        }

        val overlap = Mat()
        Core.bitwise_and(topMask, bottomMask, overlap)

        val overlapTop = findFirstNonZeroRow(overlap)
        val overlapBot = findLastNonZeroRow(overlap)

        if (overlapTop < 0 || overlapBot < 0 || overlapBot <= overlapTop) {
            Log.d("Stitcher", "  No overlap found, combining directly")
            val result = canvas.clone()
            val notTop = Mat()
            Core.bitwise_not(topMask, notTop)
            val bottomOnly = Mat()
            Core.bitwise_and(notTop, bottomMask, bottomOnly)
            warpedBottom.copyTo(result, bottomOnly)
            notTop.release(); bottomOnly.release()
            overlap.release(); topMask.release(); bottomMask.release()
            canvas.release(); warpedBottom.release()
            return cropBlack(result)
        }

        val overlapH = overlapBot - overlapTop
        Log.d("Stitcher", "  Overlap: y=$overlapTop-$overlapBot (${overlapH}px)")

        val seamY = findBestSeam(canvas, warpedBottom, overlap, overlapTop, overlapBot)
        Log.d("Stitcher", "  Best seam at y=$seamY")

        val result = canvas.clone()
        val blendRadius = 8

        val notTop = Mat()
        Core.bitwise_not(topMask, notTop)
        val bottomOnly = Mat()
        Core.bitwise_and(notTop, bottomMask, bottomOnly)
        warpedBottom.copyTo(result, bottomOnly)
        notTop.release(); bottomOnly.release()

        val belowStart = min(seamY + blendRadius + 1, outH)
        if (belowStart < outH) {
            val belowMask = Mat.zeros(outH, outW, CvType.CV_8UC1)
            Mat(belowMask, Rect(0, belowStart, outW, outH - belowStart)).setTo(Scalar(255.0))
            val belowWithBottom = Mat()
            Core.bitwise_and(belowMask, bottomMask, belowWithBottom)
            warpedBottom.copyTo(result, belowWithBottom)
            belowMask.release(); belowWithBottom.release()
        }

        val blendStart = max(0, seamY - blendRadius)
        val blendEnd = min(outH - 1, seamY + blendRadius)
        for (y in blendStart..blendEnd) {
            val rowOverlap = overlap.row(y)
            if (Core.countNonZero(rowOverlap) == 0) continue

            val alpha = (y - seamY + blendRadius).toDouble() / (2.0 * blendRadius)
            val topRow = canvas.row(y)
            val botRow = warpedBottom.row(y)
            val blended = Mat()
            Core.addWeighted(topRow, 1.0 - alpha, botRow, alpha, 0.0, blended)
            val resultRow = result.row(y)
            blended.copyTo(resultRow, rowOverlap)
            blended.release()
        }

        overlap.release(); topMask.release(); bottomMask.release()
        canvas.release(); warpedBottom.release()

        return cropBlack(result)
    } finally {
        srcPts.release()
        dstPts.release()
    }
}

private fun findBestSeam(imgTop: Mat, imgBottom: Mat, overlap: Mat, yStart: Int, yEnd: Int): Int {
    val margin = max(20, ((yEnd - yStart) * 0.2).toInt())
    val searchStart = yStart + margin
    val searchEnd = yEnd - margin

    if (searchStart >= searchEnd) return (yStart + yEnd) / 2

    val grayTop = Mat()
    val grayBottom = Mat()
    toGray(imgTop, grayTop)
    toGray(imgBottom, grayBottom)

    val topFloat = Mat()
    val botFloat = Mat()
    grayTop.convertTo(topFloat, CvType.CV_32FC1)
    grayBottom.convertTo(botFloat, CvType.CV_32FC1)
    grayTop.release(); grayBottom.release()

    var bestY = searchStart
    var bestScore = Double.MAX_VALUE

    for (y in searchStart until searchEnd) {
        val overlapRow = overlap.row(y)
        if (Core.countNonZero(overlapRow) < 10) continue

        val topRow = topFloat.row(y)
        val botRow = botFloat.row(y)
        val diff = Mat()
        Core.absdiff(topRow, botRow, diff)

        val score = Core.mean(diff, overlapRow).`val`[0]
        diff.release()

        if (score < bestScore) {
            bestScore = score
            bestY = y
        }
    }

    topFloat.release(); botFloat.release()
    Log.d("Stitcher", "  Seam search: bestScore=%.1f".format(bestScore))
    return bestY
}

private fun findFirstNonZeroRow(mask: Mat): Int {
    for (y in 0 until mask.rows()) {
        if (Core.countNonZero(mask.row(y)) > 0) return y
    }
    return -1
}

private fun findLastNonZeroRow(mask: Mat): Int {
    for (y in mask.rows() - 1 downTo 0) {
        if (Core.countNonZero(mask.row(y)) > 0) return y
    }
    return -1
}

private fun cropBlack(img: Mat): Mat {
    val gray = Mat()
    toGray(img, gray)
    val nonZero = Mat()
    Core.findNonZero(gray, nonZero)
    gray.release()

    if (nonZero.empty()) {
        nonZero.release()
        return img
    }

    val rect = Imgproc.boundingRect(nonZero)
    nonZero.release()

    if (rect.width <= 0 || rect.height <= 0) return img

    val cropped = Mat(img, rect).clone()
    img.release()
    return cropped
}

internal fun toGray(src: Mat, dst: Mat) {
    val code = if (src.channels() == 4) Imgproc.COLOR_RGBA2GRAY else Imgproc.COLOR_RGB2GRAY
    Imgproc.cvtColor(src, dst, code)
}

fun normalizeWidth(frames: List<Mat>): List<Mat> {
    if (frames.isEmpty()) return emptyList()
    val targetWidth = frames[0].cols()
    if (targetWidth <= 0) return frames.map { it.clone() }
    return frames.map { frame ->
        if (frame.cols() == targetWidth || frame.cols() <= 0) {
            frame.clone()
        } else {
            val scale = targetWidth.toDouble() / frame.cols()
            val newHeight = max(1, (frame.rows() * scale).roundToInt())
            val resized = Mat()
            Imgproc.resize(frame, resized, Size(targetWidth.toDouble(), newHeight.toDouble()))
            resized
        }
    }
}
