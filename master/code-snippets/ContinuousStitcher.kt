package com.littletrickster.scanner

import android.util.Log
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File
import kotlin.math.abs

class ContinuousStitcher {
    private val savedFrames = ArrayList<Mat>()
    private var prevGrayFloat: Mat? = null
    private var hanningWindow: Mat? = null
    private var _frameCount = 0
    private var cumulativeDisplacement = 0.0
    private var rotationDegrees = 0

    val frameCount: Int @Synchronized get() = _frameCount

    var debugFrameDir: File? = null

    @Volatile
    private var initializing = false

    private val maxFrames = 25
    private val displacementThresholdFraction = 0.15
    private val minConfidence = 0.15

    @Synchronized
    fun isInitialized() = savedFrames.isNotEmpty()

    fun tryInitialize(colorFrame: Mat, grayFrame: Mat, rotation: Int): Boolean {
        if (initializing) return false
        initializing = true
        try {
            synchronized(this) {
                if (savedFrames.isNotEmpty()) return false
                release()

                rotationDegrees = rotation

                val rotatedColor = rotateFrame(colorFrame, rotation)
                savedFrames.add(rotatedColor)
                _frameCount = 1

                val rotatedGray = rotateFrame(grayFrame, rotation)
                val grayFloat = Mat()
                rotatedGray.convertTo(grayFloat, CvType.CV_32FC1)
                if (rotatedGray !== grayFrame) rotatedGray.release()

                prevGrayFloat = grayFloat

                val window = Mat()
                Imgproc.createHanningWindow(window, Size(grayFloat.cols().toDouble(), grayFloat.rows().toDouble()), CvType.CV_32FC1)
                hanningWindow = window

                cumulativeDisplacement = 0.0
                saveDebugFrame(rotatedColor, 1, 0.0, 0.0)
                Log.d("ContStitcher", "Initialized: ${rotatedColor.cols()}x${rotatedColor.rows()} rotation=$rotation")
                return true
            }
        } finally {
            initializing = false
        }
    }

    fun hasEnoughDisplacement(grayFrame: Mat): Boolean {
        val prev: Mat
        val window: Mat
        val currentRotation: Int
        val currentCount: Int
        synchronized(this) {
            prev = prevGrayFloat ?: return false
            window = hanningWindow ?: return false
            currentRotation = rotationDegrees
            currentCount = savedFrames.size
        }

        if (currentCount >= maxFrames) return false

        val rotated = rotateFrame(grayFrame, currentRotation)
        val currentFloat = Mat()
        rotated.convertTo(currentFloat, CvType.CV_32FC1)
        if (rotated !== grayFrame) rotated.release()

        if (currentFloat.rows() != prev.rows() || currentFloat.cols() != prev.cols()) {
            currentFloat.release()
            return false
        }

        val prevRows = prev.rows()
        val response = doubleArrayOf(0.0)
        val shift = Imgproc.phaseCorrelate(prev, currentFloat, window, response)

        synchronized(this) {
            prevGrayFloat?.release()
            prevGrayFloat = currentFloat
        }

        if (response[0] < minConfidence) return false
        if (abs(shift.x) > abs(shift.y)) return false

        cumulativeDisplacement += abs(shift.y)
        val threshold = prevRows * displacementThresholdFraction
        if (cumulativeDisplacement >= threshold) {
            cumulativeDisplacement = 0.0
            return true
        }
        return false
    }

    fun captureFrame(colorFrame: Mat) {
        val currentRotation: Int
        synchronized(this) {
            currentRotation = rotationDegrees
        }

        val rotated = rotateFrame(colorFrame, currentRotation)
        val frameNum: Int
        synchronized(this) {
            if (savedFrames.size >= maxFrames) {
                if (rotated !== colorFrame) rotated.release()
                return
            }
            savedFrames.add(rotated)
            _frameCount = savedFrames.size
            frameNum = _frameCount
        }
        saveDebugFrame(rotated, frameNum, 0.0, 0.0)
        Log.d("ContStitcher", "Captured frame $frameNum")
    }

    fun finalizeStitch(): Mat? {
        val frames: List<Mat>
        synchronized(this) {
            if (savedFrames.isEmpty()) return null
            frames = ArrayList(savedFrames)
        }

        Log.d("ContStitcher", "Finalizing stitch with ${frames.size} frames")

        if (frames.size == 1) {
            return frames[0].clone()
        }

        return try {
            val result = stitchReceipts(frames)
            Log.d("ContStitcher", "Stitch complete: method=${result.method} success=${result.success} size=${result.stitchedMat.cols()}x${result.stitchedMat.rows()}")
            result.stitchedMat
        } catch (e: Exception) {
            Log.e("ContStitcher", "Stitch failed", e)
            null
        }
    }

    @Synchronized
    fun getRotation(): Int = rotationDegrees

    @Synchronized
    fun release() {
        savedFrames.forEach { it.release() }
        savedFrames.clear()
        prevGrayFloat?.release()
        hanningWindow?.release()
        prevGrayFloat = null
        hanningWindow = null
        _frameCount = 0
        cumulativeDisplacement = 0.0
    }

    private fun saveDebugFrame(frame: Mat, frameNum: Int, dy: Double, response: Double) {
        val dir = debugFrameDir ?: return
        try {
            dir.mkdirs()
            val file = File(dir, "frame_%03d.png".format(frameNum))
            val bgr = Mat()
            if (frame.channels() == 4) {
                Imgproc.cvtColor(frame, bgr, Imgproc.COLOR_RGBA2BGR)
            } else if (frame.channels() == 3) {
                Imgproc.cvtColor(frame, bgr, Imgproc.COLOR_RGB2BGR)
            } else {
                frame.copyTo(bgr)
            }
            Imgcodecs.imwrite(file.absolutePath, bgr)
            bgr.release()
            val meta = File(dir, "frame_%03d.meta".format(frameNum))
            meta.writeText("%.1f,%.3f".format(dy, response))
        } catch (e: Exception) {
            Log.e("ContStitcher", "Failed to save debug frame $frameNum", e)
        }
    }

    private fun rotateFrame(frame: Mat, degrees: Int): Mat {
        if (degrees == 0) return frame
        val code = when (degrees) {
            90 -> Core.ROTATE_90_CLOCKWISE
            180 -> Core.ROTATE_180
            270 -> Core.ROTATE_90_COUNTERCLOCKWISE
            else -> return frame
        }
        val rotated = Mat()
        Core.rotate(frame, rotated, code)
        return rotated
    }
}
