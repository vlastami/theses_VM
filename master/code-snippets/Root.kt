package com.littletrickster.scanner

import android.app.Activity
import android.content.Context
import android.graphics.Bitmap
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.material.*
import androidx.compose.runtime.*
import androidx.compose.runtime.snapshots.SnapshotStateList
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.res.stringResource
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.launch
import org.opencv.android.Utils
import org.opencv.core.Mat
import android.util.Log
import java.io.File
import java.util.*


data class ScanOutput(val originalPath: String, val dewarpedPath: String, val qualityJson: String?)

data class CapturedFrame(
    val bitmap: Bitmap,
    val rotation: Int,
    val timestamp: Long = System.currentTimeMillis()
)

const val MAX_FRAMES = 3

@Composable
fun Root(
    onImageProcessed: ((ScanOutput) -> Unit)? = null
) {
    val scope = rememberCoroutineScope()

    val context = LocalContext.current
    val tempFolder = remember { context.tempFolder() }
    val originalImageFolder = remember { context.getImageFolder() }
    val unwrappedImageFolder = remember { context.getUnwrappedImageFolder() }
    val effectImageFolder = remember { context.getEffectImageFolder() }

    var ocSaveTask by remember { mutableStateOf<Deferred<File>?>(null) }

    val capturedFrames = remember { mutableStateListOf<CapturedFrame>() }
    val frameCount by remember { derivedStateOf { capturedFrames.size } }
    val canCaptureMore by remember { derivedStateOf { capturedFrames.size < MAX_FRAMES } }
    var stitchingInProgress by remember { mutableStateOf(false) }
    var isVideoScanMode by remember { mutableStateOf(false) }
    var videoScanActive by remember { mutableStateOf(false) }
    var videoFrameCount by remember { mutableStateOf(0) }


    LaunchedEffect(null) {
        val ocFiles = originalImageFolder.listFiles()!!
        val uwFiles = unwrappedImageFolder.listFiles()!!
        val efFiles = effectImageFolder.listFiles()!!
        val temp = File(tempFolder, "temp-image.jpg")
        temp.delete()
        if (ocFiles.size != uwFiles.size) {
            ocFiles.forEach(File::delete)
            uwFiles.forEach(File::delete)
            efFiles.forEach(File::delete)

        }
    }

    var bitmapAndRotation by remember {
        mutableStateOf<Pair<Bitmap, Int>?>(null)
    }

    val finalBitmap by remember { derivedStateOf { bitmapAndRotation?.first } }
    val rotation by remember { derivedStateOf { bitmapAndRotation?.second } }




    BackHandler {
        (context as Activity).finish()
    }

    var currentTab by remember { mutableStateOf(0) }


    finalBitmap?.also { finalBitmap ->

        PolygonSet(
            originalBitmap = finalBitmap,
            rotation = rotation!!,
            back = {
                val t = bitmapAndRotation
                bitmapAndRotation = null
                t?.first?.recycle()
            },
            setPoints = { points ->

                scope.launch(Dispatchers.IO) {
                    val date = Date().time

                    val originalMat = Mat()

                    Utils.bitmapToMat(finalBitmap, originalMat)

                    val unwrappedMat = unwrap(originalMat = originalMat, points = points)
                    originalMat.release()

                    val unwrappedBitmap = unwrappedMat.toBitmap()
                    unwrappedMat.release()

                    val unwrappedFile = File(unwrappedImageFolder, "$date.jpg")
                    unwrappedFile.saveJPEG(bitmap = unwrappedBitmap, rotation = rotation!!)
                    unwrappedBitmap.recycle()

                    val originalFile = File(originalImageFolder, "$date.jpg")
                    val tempFile = ocSaveTask!!.await()
                    tempFile.renameTo(originalFile)

                    // Build metadata JSON with thresholds (from prefs), device info, scores and resolution
                    val sp = context.getSharedPreferences("auto_scan", Context.MODE_PRIVATE)
                    val thresholds = ThresholdPrefs.loadFromPreferences(ThresholdPrefs.SharedPreferencesAdapter(sp))
                    val scores = AutoScanStateStore.buildScoresPayload(thresholds)
                    val extra = mapOf(
                        "resolution" to mapOf(
                            "width" to finalBitmap.width,
                            "height" to finalBitmap.height,
                            "rotation" to rotation!!
                        )
                    )
                    val metaMap = buildQualityMetadata(thresholds, scores = scores, extra = extra)
                    val metaJson = toJson(metaMap)

                    // Notify with file paths and metadata
                    onImageProcessed?.let { callback ->
                        callback(ScanOutput(
                            originalPath = originalFile.absolutePath,
                            dewarpedPath = unwrappedFile.absolutePath,
                            qualityJson = metaJson
                        ))
                    }

                    bitmapAndRotation = null

                }
            }
        )


    } ?: run {
        Box(modifier = Modifier.fillMaxSize()) {
            Column {
                when (currentTab) {
                    0 -> {
                        TakeImage(
                            frameCount = frameCount,
                            maxFrames = MAX_FRAMES,
                            canCaptureMore = canCaptureMore,
                            onDoneClick = if (frameCount > 0 && !stitchingInProgress) {
                                {
                                    if (frameCount == 1) {
                                        val firstFrame = capturedFrames.firstOrNull()
                                        if (firstFrame != null) {
                                            bitmapAndRotation = Pair(firstFrame.bitmap, firstFrame.rotation)
                                            ocSaveTask = scope.async(Dispatchers.IO) {
                                                val tempFile = File(tempFolder, "temp-image.jpg")
                                                tempFile.saveJPEG(
                                                    bitmap = firstFrame.bitmap,
                                                    quality = 98,
                                                    rotation = firstFrame.rotation
                                                )
                                                tempFile
                                            }
                                            capturedFrames.removeAt(0)
                                            capturedFrames.forEach { it.bitmap.recycle() }
                                            capturedFrames.clear()
                                        }
                                    } else {
                                        stitchingInProgress = true
                                        val framesToProcess = capturedFrames.toList()
                                        capturedFrames.clear()

                                        scope.launch(Dispatchers.IO) {
                                            try {
                                                val date = Date().time
                                                val firstRotation = framesToProcess.first().rotation

                                                val rawMats = mutableListOf<Mat>()
                                                try {
                                                    for ((idx, frame) in framesToProcess.withIndex()) {
                                                        val mat = Mat()
                                                        Utils.bitmapToMat(frame.bitmap, mat)
                                                        Log.d("Stitcher", "Frame $idx: mat=${mat.cols()}x${mat.rows()} ch=${mat.channels()}")
                                                        rawMats.add(mat)
                                                    }

                                                    Log.d("Stitcher", "Stitching ${rawMats.size} raw frames")
                                                    val stitchResult = stitchReceipts(rawMats)
                                                    Log.d("Stitcher", "Stitch result: method=${stitchResult.method} success=${stitchResult.success} size=${stitchResult.stitchedMat.cols()}x${stitchResult.stitchedMat.rows()}")
                                                    rawMats.forEach { it.release() }
                                                    rawMats.clear()

                                                    val cropRect = segmentReceipt(stitchResult.stitchedMat)
                                                    Log.d("Stitcher", "Post-stitch crop: ${cropRect.width}x${cropRect.height} at (${cropRect.x},${cropRect.y})")
                                                    val cropped = Mat(stitchResult.stitchedMat, cropRect).clone()
                                                    stitchResult.stitchedMat.release()
                                                    val finalResult = StitchResult(cropped, stitchResult.success, stitchResult.method)

                                                    val stitchedBitmap = finalResult.stitchedMat.toBitmap()
                                                    finalResult.stitchedMat.release()

                                                    val unwrappedFile = File(unwrappedImageFolder, "$date.jpg")
                                                    unwrappedFile.saveJPEG(bitmap = stitchedBitmap, rotation = firstRotation)

                                                    val originalFile = File(originalImageFolder, "$date.jpg")
                                                    originalFile.saveJPEG(
                                                        bitmap = framesToProcess.first().bitmap,
                                                        quality = 98,
                                                        rotation = firstRotation
                                                    )

                                                    val sp = context.getSharedPreferences("auto_scan", Context.MODE_PRIVATE)
                                                    val thresholds = ThresholdPrefs.loadFromPreferences(ThresholdPrefs.SharedPreferencesAdapter(sp))
                                                    val scores = AutoScanStateStore.buildScoresPayload(thresholds)
                                                    val extra = mapOf(
                                                        "resolution" to mapOf(
                                                            "width" to stitchedBitmap.width,
                                                            "height" to stitchedBitmap.height,
                                                            "rotation" to firstRotation
                                                        ),
                                                        "stitching" to mapOf(
                                                            "frameCount" to framesToProcess.size,
                                                            "method" to finalResult.method,
                                                            "success" to finalResult.success
                                                        )
                                                    )
                                                    val metaMap = buildQualityMetadata(thresholds, scores = scores, extra = extra)
                                                    val metaJson = toJson(metaMap)

                                                    stitchedBitmap.recycle()

                                                    onImageProcessed?.let { callback ->
                                                        callback(ScanOutput(
                                                            originalPath = originalFile.absolutePath,
                                                            dewarpedPath = unwrappedFile.absolutePath,
                                                            qualityJson = metaJson
                                                        ))
                                                    }
                                                } catch (e: Exception) {
                                                    rawMats.forEach { it.release() }

                                                    try {
                                                        val fallbackMat = Mat()
                                                        Utils.bitmapToMat(framesToProcess.first().bitmap, fallbackMat)

                                                        val fallbackBitmap = fallbackMat.toBitmap()
                                                        fallbackMat.release()

                                                        val unwrappedFile = File(unwrappedImageFolder, "$date.jpg")
                                                        unwrappedFile.saveJPEG(bitmap = fallbackBitmap, rotation = firstRotation)

                                                        val originalFile = File(originalImageFolder, "$date.jpg")
                                                        originalFile.saveJPEG(
                                                            bitmap = framesToProcess.first().bitmap,
                                                            quality = 98,
                                                            rotation = firstRotation
                                                        )

                                                        val sp = context.getSharedPreferences("auto_scan", Context.MODE_PRIVATE)
                                                        val thresholds = ThresholdPrefs.loadFromPreferences(ThresholdPrefs.SharedPreferencesAdapter(sp))
                                                        val scores = AutoScanStateStore.buildScoresPayload(thresholds)
                                                        val extra = mapOf(
                                                            "resolution" to mapOf(
                                                                "width" to fallbackBitmap.width,
                                                                "height" to fallbackBitmap.height,
                                                                "rotation" to firstRotation
                                                            ),
                                                            "stitching" to mapOf(
                                                                "frameCount" to framesToProcess.size,
                                                                "method" to "fallback_single",
                                                                "success" to false
                                                            )
                                                        )
                                                        val metaMap = buildQualityMetadata(thresholds, scores = scores, extra = extra)
                                                        val metaJson = toJson(metaMap)

                                                        fallbackBitmap.recycle()

                                                        onImageProcessed?.let { callback ->
                                                            callback(ScanOutput(
                                                                originalPath = originalFile.absolutePath,
                                                                dewarpedPath = unwrappedFile.absolutePath,
                                                                qualityJson = metaJson
                                                            ))
                                                        }
                                                    } catch (_: Exception) {
                                                    }
                                                }
                                            } finally {
                                                framesToProcess.forEach { it.bitmap.recycle() }
                                                stitchingInProgress = false
                                            }
                                        }
                                    }
                                }
                            } else null,
                            fileReceived = { pair ->
                                if (canCaptureMore) {
                                    capturedFrames.add(CapturedFrame(pair.first, pair.second))
                                } else {
                                    pair.first.recycle()
                                }
                            },
                            isVideoScanMode = isVideoScanMode,
                            onVideoScanModeChange = { isVideoScanMode = it },
                            videoScanActive = videoScanActive,
                            onVideoScanActiveChange = { videoScanActive = it },
                            videoFrameCount = videoFrameCount,
                            onVideoFrameCountChange = { videoFrameCount = it },
                            onVideoScanComplete = {
                                videoScanActive = false
                            },
                            onVideoStitchResult = { stitchedMat, stitchRotation ->
                                stitchingInProgress = true
                                scope.launch(Dispatchers.IO) {
                                    var matReleased = false
                                    try {
                                        val date = Date().time

                                        val cropRect = segmentReceipt(stitchedMat)
                                        val cropped = Mat(stitchedMat, cropRect).clone()
                                        stitchedMat.release()
                                        matReleased = true

                                        val croppedBitmap = cropped.toBitmap()
                                        cropped.release()

                                        val unwrappedFile = File(unwrappedImageFolder, "$date.jpg")
                                        unwrappedFile.saveJPEG(bitmap = croppedBitmap, rotation = 0)

                                        val originalFile = File(originalImageFolder, "$date.jpg")
                                        originalFile.saveJPEG(bitmap = croppedBitmap, quality = 98, rotation = 0)

                                        val sp = context.getSharedPreferences("auto_scan", Context.MODE_PRIVATE)
                                        val thresholds = ThresholdPrefs.loadFromPreferences(ThresholdPrefs.SharedPreferencesAdapter(sp))
                                        val scores = AutoScanStateStore.buildScoresPayload(thresholds)
                                        val extra = mapOf(
                                            "resolution" to mapOf(
                                                "width" to croppedBitmap.width,
                                                "height" to croppedBitmap.height,
                                                "rotation" to stitchRotation
                                            ),
                                            "stitching" to mapOf(
                                                "method" to "continuous",
                                                "success" to true
                                            )
                                        )
                                        val metaMap = buildQualityMetadata(thresholds, scores = scores, extra = extra)
                                        val metaJson = toJson(metaMap)

                                        croppedBitmap.recycle()

                                        onImageProcessed?.let { callback ->
                                            callback(ScanOutput(
                                                originalPath = originalFile.absolutePath,
                                                dewarpedPath = unwrappedFile.absolutePath,
                                                qualityJson = metaJson
                                            ))
                                        }
                                    } catch (e: Exception) {
                                        Log.d("Root", "Video stitch processing failed: ${e.message}")
                                        if (!matReleased) stitchedMat.release()
                                    } finally {
                                        stitchingInProgress = false
                                    }
                                }
                            }
                        )
                    }
                }
            }
            if (stitchingInProgress) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(Color.Black.copy(alpha = 0.6f)),
                    contentAlignment = Alignment.Center
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        CircularProgressIndicator(color = Color.White)
                        Spacer(modifier = Modifier.height(16.dp))
                        Text(
                            text = "Processing...",
                            color = Color.White,
                            style = MaterialTheme.typography.body1
                        )
                    }
                }
            }
        }
    }
}
