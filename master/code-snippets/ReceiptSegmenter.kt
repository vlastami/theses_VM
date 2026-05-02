package com.littletrickster.scanner

import android.util.Log
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Rect
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

private const val TAG = "ReceiptSegmenter"
private const val MIN_AREA_RATIO = 0.20
private const val MIN_WIDTH_RATIO = 0.40

fun isValidCrop(cropW: Int, cropH: Int, frameW: Int, frameH: Int): Boolean {
    if (frameW <= 0 || frameH <= 0) return false
    val frameArea = frameW.toLong() * frameH.toLong()
    val cropArea = cropW.toLong() * cropH.toLong()
    val areaRatio = cropArea.toDouble() / frameArea
    val widthRatio = cropW.toDouble() / frameW
    return areaRatio >= MIN_AREA_RATIO && widthRatio >= MIN_WIDTH_RATIO
}

fun segmentReceipt(mat: Mat): Rect {
    val frameW = mat.cols()
    val frameH = mat.rows()
    val fullFrame = Rect(0, 0, frameW, frameH)

    val gray = Mat()
    val blurred = Mat()
    val binary = Mat()
    val kernel = Mat()
    val hierarchy = Mat()
    val contours = mutableListOf<MatOfPoint>()

    try {
        when (mat.channels()) {
            1 -> mat.copyTo(gray)
            3 -> Imgproc.cvtColor(mat, gray, Imgproc.COLOR_RGB2GRAY)
            4 -> Imgproc.cvtColor(mat, gray, Imgproc.COLOR_RGBA2GRAY)
            else -> return fullFrame
        }

        Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)

        Imgproc.threshold(blurred, binary, 0.0, 255.0, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU)

        val structElem = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(7.0, 7.0))
        try {
            Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_CLOSE, structElem)
        } finally {
            structElem.release()
        }

        Imgproc.findContours(binary, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        if (contours.isEmpty()) {
            Log.d(TAG, "No contours found, using full frame")
            return fullFrame
        }

        var largestArea = -1.0
        var largestIdx = 0
        for (i in contours.indices) {
            val area = Imgproc.contourArea(contours[i])
            if (area > largestArea) {
                largestArea = area
                largestIdx = i
            }
        }

        val boundingRect = Imgproc.boundingRect(contours[largestIdx])
        Log.d(TAG, "Largest contour: ${boundingRect.width}x${boundingRect.height} at (${boundingRect.x},${boundingRect.y})")

        return if (isValidCrop(boundingRect.width, boundingRect.height, frameW, frameH)) {
            Log.d(TAG, "Valid crop, using segmented bounds")
            boundingRect
        } else {
            Log.d(TAG, "Crop too small, using full frame")
            fullFrame
        }
    } catch (e: Exception) {
        Log.e(TAG, "Segmentation failed: ${e.message}")
        return fullFrame
    } finally {
        gray.release()
        blurred.release()
        binary.release()
        kernel.release()
        hierarchy.release()
        contours.forEach { it.release() }
    }
}
