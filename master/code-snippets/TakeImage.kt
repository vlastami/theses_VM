package com.littletrickster.scanner

import android.content.Context
import android.graphics.Bitmap
import android.widget.Toast
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageProxy
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicText
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp
import androidx.compose.material.CircularProgressIndicator
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.FlashAuto
import androidx.compose.material.icons.filled.FlashOff
import androidx.compose.material.icons.filled.FlashOn
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.vector.rememberVectorPainter
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.core.content.edit
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.drop
import kotlinx.coroutines.flow.launchIn
import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext


@Composable
fun TakeImage(
    frameCount: Int = 0,
    maxFrames: Int = 3,
    canCaptureMore: Boolean = true,
    onDoneClick: (() -> Unit)? = null,
    fileReceived: (file: Pair<Bitmap, Int>) -> Unit = {},
    isVideoScanMode: Boolean = false,
    onVideoScanModeChange: (Boolean) -> Unit = {},
    videoScanActive: Boolean = false,
    onVideoScanActiveChange: (Boolean) -> Unit = {},
    onVideoScanComplete: () -> Unit = {},
    onVideoStitchResult: (org.opencv.core.Mat, Int) -> Unit = { _, _ -> },
    videoFrameCount: Int = 0,
    onVideoFrameCountChange: (Int) -> Unit = {}
) {
    val context = LocalContext.current
    val prefs = rememberScannerSharedPrefs()
    val th = remember(context) {
        val sp = context.getSharedPreferences("auto_scan", Context.MODE_PRIVATE)
        ThresholdPrefs.loadFromPreferences(ThresholdPrefs.SharedPreferencesAdapter(sp))
    }

    var currentFlashMode by remember {
        mutableStateOf(prefs.getInt("flash_mode", ImageCapture.FLASH_MODE_AUTO))
    }

    var capturing by remember { mutableStateOf(false) }


    val imageCaptureConfig = remember {
        ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
            .setResolutionSelector(defaultResolutionSelector())
            .build()
    }

    LaunchedEffect(null) {
        val flow = snapshotFlow { currentFlashMode }

        flow.drop(1).onEach {
            prefs
                .edit {
                    this.putInt("flash_mode", it)
                }
        }.launchIn(this)

        flow.onEach { imageCaptureConfig.flashMode = it }
            .launchIn(this)
    }

    val scope = rememberCoroutineScope()

    fun doCapture(isAutoCapture: Boolean) {
        if (capturing) return
        capturing = true
        scope.launch(Dispatchers.IO) {
            var bitmap: Bitmap? = null
            var finalBitmap: Bitmap? = null
            var passedToConsumer = false
            try {
                // Capture to a temporary file to avoid fragile YUV->RGB conversion
                val temp = context.tempFolder()
                val name = "capture_${System.currentTimeMillis()}.jpg"
                val file = imageCaptureConfig.takePicture(temp, name)

                // Decode bitmap
                bitmap = android.graphics.BitmapFactory.decodeFile(file.absolutePath)

                // Read rotation from EXIF
                val exif = androidx.exifinterface.media.ExifInterface(file.absolutePath)
                val orientation = exif.getAttributeInt(
                    androidx.exifinterface.media.ExifInterface.TAG_ORIENTATION,
                    androidx.exifinterface.media.ExifInterface.ORIENTATION_NORMAL
                )
                val rotation = when (orientation) {
                    androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_90 -> 90
                    androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_180 -> 180
                    androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_270 -> 270
                    else -> 0
                }

                // Normalize: apply rotation to pixels so UI can use rotation=0 (avoids any aspect mismatch)
                finalBitmap = if (rotation != 0) {
                    val m = android.graphics.Matrix()
                    m.postRotate(rotation.toFloat())
                    android.graphics.Bitmap.createBitmap(bitmap!!, 0, 0, bitmap.width, bitmap.height, m, true)
                } else bitmap

                // Post-capture blur check
                val isBlurry = isPostCaptureBlurry(finalBitmap!!, th.minVoLRelaxed)

                if (isBlurry && isAutoCapture) {
                    withContext(Dispatchers.Main) {
                        Toast.makeText(context, "Blurry photo rejected. Hold steady and try again.", Toast.LENGTH_SHORT).show()
                    }
                    // Don't set passedToConsumer = true, so it gets recycled in finally
                } else {
                    if (isBlurry) { // This implies !isAutoCapture
                        withContext(Dispatchers.Main) {
                            Toast.makeText(context, "This photo looks blurry. Consider trying again.", Toast.LENGTH_LONG).show()
                        }
                        fileReceived(finalBitmap to 1) // 1 = blurry
                    } else { // Not blurry
                        fileReceived(finalBitmap to 0) // 0 = sharp
                    }
                    passedToConsumer = true
                }

            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    val msg = "${e.javaClass.simpleName}: ${e.message}"
                    Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
                }
            } finally {
                if (!passedToConsumer) {
                    finalBitmap?.recycle()
                }
                if (finalBitmap !== bitmap) {
                    bitmap?.recycle()
                }
                capturing = false
            }
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
    ) {
        PointPreview(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f),
            imageCaptureConfig = imageCaptureConfig,
            onAutoCaptureRequest = { if (canCaptureMore) doCapture(isAutoCapture = true) },
            captureInProgress = capturing,
            frameCount = frameCount,
            maxFrames = maxFrames,
            canCaptureMore = canCaptureMore,
            isVideoScanMode = isVideoScanMode,
            videoScanActive = videoScanActive,
            onVideoScanComplete = onVideoScanComplete,
            onVideoStitchResult = onVideoStitchResult,
            onVideoFrameCountChange = onVideoFrameCountChange
        )

        BottomBar(
            modifier = Modifier.fillMaxWidth().navigationBarsPadding(),
            capturing = capturing,
            captureEnabled = canCaptureMore,
            captureClick = { if (canCaptureMore) doCapture(isAutoCapture = false) },
            currentFlashMode = currentFlashMode,
            modeClick = {
                val nextMode = when (currentFlashMode) {
                    ImageCapture.FLASH_MODE_OFF -> ImageCapture.FLASH_MODE_AUTO
                    ImageCapture.FLASH_MODE_AUTO -> ImageCapture.FLASH_MODE_ON
                    else -> ImageCapture.FLASH_MODE_OFF
                }
                currentFlashMode = nextMode
            },
            frameCount = frameCount,
            videoFrameCount = videoFrameCount,
            maxFrames = maxFrames,
            onDoneClick = onDoneClick,
            isVideoScanMode = isVideoScanMode,
            onScanModeToggle = { onVideoScanModeChange(!isVideoScanMode) },
            videoScanActive = videoScanActive,
            onScanStartStop = {
                if (videoScanActive) {
                    onVideoScanActiveChange(false)
                } else {
                    onVideoScanActiveChange(true)
                }
            }
        )
    }

}


@Preview
@Composable
fun BottomBar(
    modifier: Modifier = Modifier,
    capturing: Boolean = false,
    captureEnabled: Boolean = true,
    captureClick: () -> Unit = {},
    currentFlashMode: Int = ImageCapture.FLASH_MODE_AUTO,
    modeClick: () -> Unit = {},
    frameCount: Int = 0,
    videoFrameCount: Int = 0,
    maxFrames: Int = 3,
    onDoneClick: (() -> Unit)? = null,
    isVideoScanMode: Boolean = false,
    onScanModeToggle: () -> Unit = {},
    videoScanActive: Boolean = false,
    onScanStartStop: () -> Unit = {}
) {

    Box(
        modifier = Modifier
            .height(100.dp)
            .then(modifier)
    ) {

        Row(
            modifier = Modifier.fillMaxSize(),
            horizontalArrangement = Arrangement.SpaceAround,
            verticalAlignment = Alignment.CenterVertically
        ) {

            Box(modifier = Modifier.size(70.dp), contentAlignment = Alignment.Center) {
                if (isVideoScanMode) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        BasicText(
                            text = if (videoScanActive) "$videoFrameCount/$maxFrames" else "Scan",
                            style = TextStyle(
                                color = if (videoScanActive) Color(0xFFFF5252) else Color.White,
                                fontSize = 16.sp,
                                fontWeight = FontWeight.Bold
                            )
                        )
                        if (!videoScanActive) {
                            BasicText(
                                text = "mode",
                                style = TextStyle(color = Color.White.copy(alpha = 0.7f), fontSize = 10.sp)
                            )
                        }
                    }
                } else {
                    if (frameCount > 0) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            BasicText(
                                text = "$frameCount/$maxFrames",
                                style = TextStyle(color = Color.White, fontSize = 16.sp, fontWeight = FontWeight.Bold)
                            )
                            if (captureEnabled) {
                                BasicText(
                                    text = "frames",
                                    style = TextStyle(color = Color.White.copy(alpha = 0.7f), fontSize = 10.sp)
                                )
                            }
                        }
                    } else if (captureEnabled) {
                        BasicText(
                            text = "Up to\n$maxFrames shots",
                            style = TextStyle(
                                color = Color.White.copy(alpha = 0.7f), fontSize = 10.sp,
                                textAlign = androidx.compose.ui.text.style.TextAlign.Center
                            )
                        )
                    }
                }
            }

            if (isVideoScanMode) {
                if (!videoScanActive && videoFrameCount == 0) {
                    Box(
                        modifier = Modifier
                            .background(Color(0xFFFF5252), RoundedCornerShape(8.dp))
                            .clickable { onScanStartStop() }
                            .padding(horizontal = 20.dp, vertical = 10.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        BasicText(
                            text = "Start Scan",
                            style = TextStyle(color = Color.White, fontSize = 16.sp, fontWeight = FontWeight.Bold)
                        )
                    }
                } else if (videoScanActive) {
                    Box(
                        modifier = Modifier
                            .background(Color(0xFFFF5252), CircleShape)
                            .clickable { onScanStartStop() }
                            .size(60.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Box(
                            modifier = Modifier
                                .background(Color.White, RoundedCornerShape(4.dp))
                                .size(20.dp)
                        )
                    }
                } else if (videoFrameCount > 1) {
                    Box(
                        modifier = Modifier
                            .padding(horizontal = 16.dp, vertical = 8.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        BasicText(
                            text = "Processing...",
                            style = TextStyle(color = Color.White.copy(alpha = 0.7f), fontSize = 14.sp)
                        )
                    }
                } else {
                    Box(modifier = Modifier.size(50.dp))
                }

                if (!isVideoScanMode || (!videoScanActive && videoFrameCount == 0)) {
                    Box(modifier = Modifier.size(50.dp))
                }
            } else {
                if (onDoneClick != null && frameCount > 0) {
                    Box(
                        modifier = Modifier
                            .background(Color(0xFF4CAF50), RoundedCornerShape(8.dp))
                            .clickable { onDoneClick() }
                            .padding(horizontal = 16.dp, vertical = 8.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        BasicText(
                            text = "Done",
                            style = TextStyle(color = Color.White, fontSize = 16.sp, fontWeight = FontWeight.Bold)
                        )
                    }
                } else {
                    Box(modifier = Modifier.size(50.dp))
                }

                CaptureButton(enabled = !capturing && captureEnabled) {
                    captureClick()
                }
            }

            val flashAuto = rememberVectorPainter(Icons.Filled.FlashAuto)
            val flashOff = rememberVectorPainter(Icons.Filled.FlashOff)
            val flashOn = rememberVectorPainter(Icons.Filled.FlashOn)

            val currentPainter = when (currentFlashMode) {
                ImageCapture.FLASH_MODE_AUTO -> flashAuto
                ImageCapture.FLASH_MODE_OFF -> flashOff
                else -> flashOn
            }

            Image(painter = currentPainter, "flash",
                Modifier
                    .clickable { modeClick() }
                    .padding(8.dp),
                colorFilter = ColorFilter.tint(Color.White)
            )

            if (!videoScanActive && frameCount == 0 && videoFrameCount == 0) {
                Box(
                    modifier = Modifier
                        .background(
                            if (isVideoScanMode) Color(0xFF90CAF9) else Color(0xFF333333),
                            RoundedCornerShape(16.dp)
                        )
                        .clickable { onScanModeToggle() }
                        .padding(horizontal = 14.dp, vertical = 8.dp),
                    contentAlignment = Alignment.Center
                ) {
                    BasicText(
                        text = if (isVideoScanMode) "Photo" else "Scan",
                        style = TextStyle(
                            color = if (isVideoScanMode) Color.Black else Color.White,
                            fontSize = 14.sp,
                            fontWeight = FontWeight.Bold
                        )
                    )
                }
            } else {
                Box(modifier = Modifier.size(20.dp))
            }

        }

    }
}


val offsetAnim = tween<Offset>(durationMillis = 220, easing = LinearEasing)


fun defaultResolutionSelector() = ResolutionSelector.Builder().apply {
    setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
}.build()

fun highResolutionSelector() = ResolutionSelector.Builder().apply {
    setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
    setResolutionStrategy(
        ResolutionStrategy(
            android.util.Size(1280, 960),
            ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
        )
    )
}.build()


@Preview
@Composable
fun CaptureButtonPreview(
) {
    var enabled by remember { mutableStateOf(true) }
    CaptureButton(enabled) { enabled = !enabled }
}


@Composable
fun CaptureButton(
    enabled: Boolean = true,
    click: () -> Unit = {}
) {
    val interactionSource = remember { MutableInteractionSource() }

    val clicked by interactionSource.collectIsPressedAsState()

    val delta by animateDpAsState(
        targetValue = if (clicked) 20.dp else 0.dp,
        animationSpec = tween(durationMillis = 220, easing = LinearEasing)
    )
    val alpha by animateFloatAsState(
        targetValue = if (enabled) 1f else 0f,
        animationSpec = tween(durationMillis = 220, easing = LinearEasing)
    )



    Box(
        modifier = Modifier
            .clickable(
                enabled = enabled,
                onClick = click,
                interactionSource = interactionSource,
                indication = null
            )
            .padding(2.dp)
            .border(2.dp, Color.White, CircleShape) // inner border
            .padding(6.dp + delta / 2) // padding

    ) {

        if (!enabled) {
            CircularProgressIndicator(
                Modifier
                    .alpha(1f - alpha)
                    .size(50.dp - delta),
//                color = Color.Black,
                strokeCap = StrokeCap.Round
            )
        }

        Box(
            Modifier
                .alpha(alpha)
                .size(50.dp - delta)
                .background(Color.White, CircleShape)

        )


    }
}
