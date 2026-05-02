package com.littletrickster.scanner

import android.util.SparseArray
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt


/**
 * Returns points in order 0 top left 1 top right 2 bottom right 3 bottom left
 *
 * @param original mat
 *
 */
fun getPointsDownscaled(original: Mat, maxDim: Int = 1024): List<Point> {
    val longestSide = max(original.cols(), original.rows())
    if (longestSide <= maxDim) return getPoints(original)

    val scale = maxDim.toDouble() / longestSide
    val smallMat = Mat()
    try {
        Imgproc.resize(
            original, smallMat,
            org.opencv.core.Size(original.cols() * scale, original.rows() * scale)
        )
        val smallPoints = getPoints(smallMat)
        return smallPoints.map { Point(it.x / scale, it.y / scale) }
    } finally {
        smallMat.release()
    }
}

fun getPoints(original: Mat): List<Point> {

    val width: Double
    val height: Double

    original.size().also {
        width = it.width
        height = it.height
    }

    val blurred = Mat()
    val srcGray = Mat()
    // value to modify for rectangle search adjustments
    Imgproc.medianBlur(original, blurred, 9)
    val threshOutput = Mat(blurred.size(), CvType.CV_8U)
    val squares = ArrayList<MatOfPoint>()
    val threshSquares: ArrayList<MatOfPoint> = ArrayList()
    val cannySquares = SparseArray<MatOfPoint>()
    val indices = ArrayList<Int>()

    val lBlurred = listOf(blurred)
    val lOutput = listOf(threshOutput)

    // value to modify for rectangle search adjustments
    for (c in 0..2) {

        Core.mixChannels(lBlurred, lOutput, MatOfInt(c, 0))

        val thresholdLevel = 3

        for (l in 0 until thresholdLevel) {

            if (l == 0) {
                for (t in 10..60 step 10) {
                    Imgproc.Canny(threshOutput, srcGray, t.toDouble(), (t * 2).toDouble())
                    Imgproc.dilate(srcGray, srcGray, Mat(), Point(-1.0, -1.0), 2)
                    findCannySquares(
                        srcGray = srcGray,
                        scaledWidth = width,
                        scaledHeight = height,
                        cannySquares = cannySquares,
                        (c + t),
                        indices = indices
                    )
                }


            } else {

                // values ​​and calculation to modify for rectangle search adjustments
                Imgproc.threshold(threshOutput, srcGray, (200 - 175 / (l + 2f)).toDouble(), 256.0, Imgproc.THRESH_BINARY)

                findThreshSquares(srcGray, width, height, threshSquares)
            }


//                }
        }

    }
    val indiceMax: Int = maxi(indices)
    if (indiceMax != -1) squares.add(cannySquares[indiceMax])

    // Removal of unlikely quadrilaterals i.e. those that touch the edges
    val squaresProba: ArrayList<MatOfPoint> = ArrayList()
    var pointsProba: MatOfPoint
    var pointsList: List<Point>
    val marge = (srcGray.size().width * 0.01f).toInt()
    var probable: Boolean
    for (i in threshSquares.indices) {
        probable = true
        pointsProba = threshSquares[i]
        pointsList = pointsProba.toList()
        for (p in pointsList) {
            if (p.x < marge || p.x >= srcGray.size().width - marge || p.y < marge || p.y >= srcGray.size().height - marge) {
                probable = false
                break
            }
        }
        if (probable) {
            squaresProba.add(pointsProba)
        }
    }

    // selection of the largest quadrilateral
    var largestContourIndex = 0
    var points = MatOfPoint()
    if (squaresProba.size != 0) {
        var largestArea = -1.0
        for (i in squaresProba.indices) {
            val a = Imgproc.contourArea(squaresProba[i], false)
            if (a > largestArea && a < srcGray.size().height * srcGray.size().width) {
                largestArea = a
                largestContourIndex = i
            }
        }
        if (squaresProba.size > 0) {
            points = squaresProba[largestContourIndex]
        } else {
            val pts: MutableList<Point> = ArrayList()
            pts.add(Point(0.0, 0.0))
            pts.add(Point(width, 0.0))
            pts.add(Point(0.0, height))
            pts.add(Point(width, height))
            points.fromList(pts)
        }
    } else {
        var largestArea = -1.0
        for (i in threshSquares.indices) {
            val a = Imgproc.contourArea(threshSquares[i], false)
            if (a > largestArea && a < srcGray.size().height * srcGray.size().width) {
                largestArea = a
                largestContourIndex = i
            }
        }
        if (threshSquares.size > 0) {
            points = threshSquares[largestContourIndex]
        } else {
            val pts: MutableList<Point> = ArrayList()
            pts.add(Point(0.0, 0.0))
            pts.add(Point(width, 0.0))
            pts.add(Point(0.0, height))
            pts.add(Point(width, height))
            points.fromList(pts)
        }
    }
    squares.add(points)

    // Addition of other contours after the 2 most important
    for (id in indices) {
        if (id != indiceMax) squares.add(cannySquares[id])
    }
    for (id in threshSquares.indices) {
        if (id != largestContourIndex) {
            squares.add(threshSquares[id])
        }
    }
    val unsorted = squares[0].toArray()


    return getOrderedPoints(unsorted)
}


private fun findCannySquares(
    srcGray: Mat,
    scaledWidth: Double,
    scaledHeight: Double,
    cannySquares: SparseArray<MatOfPoint>,
    indice: Int,
    indices: ArrayList<Int>
) {
    // contours search
    val contours: List<MatOfPoint> = ArrayList()
    Imgproc.findContours(srcGray, contours, Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE)
    val approx = MatOfPoint2f()
    for (i in contours.indices) {
        val contour = MatOfPoint2f()
        contour.fromArray(*contours[i].toArray())
        // detection of geometric shapes
        Imgproc.approxPolyDP(contour, approx, Imgproc.arcLength(contour, true) * 0.03, true)
        val approx1f = MatOfPoint()
        val approxArray = approx.toArray()
        approx1f.fromArray(*approxArray)
        // detection of quadrilaterals among geometric shapes
        if (approx.total() == 4L && abs(Imgproc.contourArea(approx)) > scaledWidth / 5 * (scaledHeight / 5) && Imgproc.isContourConvex(approx1f)) {
            var maxCosine = 0.0
            for (j in 2..4) {
                val cosine = abs(angle(approxArray[j % 4], approxArray[j - 2], approxArray[j - 1]))
                maxCosine = maxCosine.coerceAtLeast(cosine)
            }
            // selection of quadrilaterals with large enough angles
            if (maxCosine < 0.5) {
                cannySquares.put(indice, approx1f)
                indices.add(indice)
            }
        }
    }
}


private fun findThreshSquares(
    srcGray: Mat,
    scaledWidth: Double,
    scaledHeight: Double,
    threshSquares: MutableList<MatOfPoint>
) {
    // contours search
    val contours: List<MatOfPoint> = ArrayList()
    Imgproc.findContours(srcGray, contours, Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE)
    val approx = MatOfPoint2f()
    for (i in contours.indices) {
        val contour = MatOfPoint2f()
        contour.fromArray(*contours[i].toArray())
        // detection of geometric shapes
        Imgproc.approxPolyDP(contour, approx, Imgproc.arcLength(contour, true) * 0.03, true)
        val approxArray = approx.toArray()

        val approx1f = MatOfPoint()
        approx1f.fromArray(*approxArray)
        // detection of quadrilaterals among geometric shapes
        if (approx.total() == 4L && abs(Imgproc.contourArea(approx)) > scaledWidth / 5 * (scaledHeight / 5) && Imgproc.isContourConvex(
                approx1f
            )
        ) {
            var maxCosine = 0.0
            for (j in 2..4) {
                val cosine = abs(angle(approxArray[j % 4], approxArray[j - 2], approxArray[j - 1]))
                maxCosine = max(maxCosine, cosine)
            }
            //selection of quadrilaterals with large enough angles
            if (maxCosine < 0.5) {
                threshSquares.add(approx1f)
            }
        }
    }
}

private fun maxi(indices: List<Int>): Int {
    var max = -1
    for (i in indices) {
        if (i > max) max = i
    }
    return max
}

fun getOrderedPoints(points: Array<Point>): List<Point> {
    require(points.size == 4) { "getOrderedPoints expects 4 points" }
    // Sort by x to split into left and right pairs
    val byX = points.sortedBy { it.x }
    val left = byX.take(2).sortedBy { it.y }   // top (min y), bottom (max y)
    val right = byX.takeLast(2).sortedBy { it.y }
    val tl = left[0]
    val bl = left[1]
    val tr = right[0]
    val br = right[1]
    return listOf(tl, tr, br, bl)
}

fun angle(pt1: Point, pt2: Point, pt0: Point): Double {
    val dx1 = pt1.x - pt0.x
    val dy1 = pt1.y - pt0.y
    val dx2 = pt2.x - pt0.x
    val dy2 = pt2.y - pt0.y
    return (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10)
}

// Non-breaking API to get quad points along with edge-fit confidence
fun getPointsWithEdgeFit(original: Mat): Pair<List<Point>, EdgeFit> {
    val pts = getPoints(original)
    val size = original.size()
    val w = size.width
    val h = size.height
    var edge = computeEdgeFit(pts)
    if (isFullFrameFallback(pts, w, h)) {
        edge = edge.copy(hasQuad = false)
    }
    return Pair(pts, edge)
}


// Data class representing a detected quadrilateral with basic confidence metrics
data class QuadDetection(
    val quad: List<Point>?,
    val area: Double,
    val maxCosine: Double
)

/**
 * Detect quad and compute simple confidence metrics.
 * - quad will be null if detection is a full-frame fallback (i.e., no real quad found).
 * - area is the contour area of the detected quad (0.0 if none).
 * - maxCosine is the maximum absolute corner cosine (lower is more rectangular).
 */
fun detectQuadWithConfidence(original: Mat): QuadDetection {
    val points = getPoints(original)
    val size = original.size()
    val w = size.width
    val h = size.height

    // If the returned points represent the full-frame fallback, treat as no-quad
    if (isFullFrameFallback(points, w, h)) {
        return QuadDetection(null, 0.0, Double.POSITIVE_INFINITY)
    }

    // Compute area using OpenCV contourArea
    val mop = MatOfPoint()
    mop.fromArray(*points.toTypedArray())
    val area = Imgproc.contourArea(mop, false)

    // Compute maxCosine similar to internal detection helpers
    val arr = points.toTypedArray()
    var maxCosine = 0.0
    for (j in 2..4) {
        val cosine = kotlin.math.abs(angle(arr[j % 4], arr[j - 2], arr[j - 1]))
        if (cosine > maxCosine) maxCosine = cosine
    }

    return QuadDetection(points, area, maxCosine)
}
