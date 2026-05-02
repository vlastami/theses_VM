package com.littletrickster.scanner

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.material.MaterialTheme
import androidx.compose.material.Surface
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.lifecycleScope
import com.littletrickster.scanner.ui.theme.ScannerTheme
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.launch
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsControllerCompat


class ScanActivity : ComponentActivity() {

    companion object {
        // Key for the returned Bitmap in the Intent, matching the one in ScannerActivity
        const val EXTRA_RESULT_BITMAP = "scanned_bitmap"
        // Key for the returned image file path (legacy single-path result)
        const val EXTRA_RESULT_IMAGE_PATH = "image_path"
        // New extras for v2 contract
        const val EXTRA_RESULT_ORIGINAL_PATH = "original_path"
        const val EXTRA_RESULT_DEWARPED_PATH = "dewarped_path"
        const val EXTRA_RESULT_QUALITY_JSON = "quality_json"
    }

    // Flow to receive the final processed output
    private val finalImageFlow = MutableSharedFlow<ScanOutput>(replay = 0)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        lifecycleScope.launch {
            finalImageFlow.collect { output ->
                sendResultAndFinish(output)
            }
        }

        setContent {
            val composeScope = rememberCoroutineScope()
            val context = LocalContext.current

            ScannerTheme {
                PermissionRequester {
                    ProvideIsPick {
                        Surface(color = MaterialTheme.colors.background) {
                            Root(
                                onImageProcessed = { output ->
                                    composeScope.launch {
                                        finalImageFlow.emit(output)
                                    }
                                }
                            )
                        }
                    }
                }
            }
        }
    }

    private fun sendResultAndFinish(output: ScanOutput) {
        val data = Intent()
        // Legacy single path: choose dewarped by default
        data.putExtra(EXTRA_RESULT_IMAGE_PATH, output.dewarpedPath)
        // V2 contract
        data.putExtra(EXTRA_RESULT_ORIGINAL_PATH, output.originalPath)
        data.putExtra(EXTRA_RESULT_DEWARPED_PATH, output.dewarpedPath)
        output.qualityJson?.let { data.putExtra(EXTRA_RESULT_QUALITY_JSON, it) }
        setResult(Activity.RESULT_OK, data)
        finish()
    }
}
