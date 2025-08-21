package com.samsung.videotraffic.ml

import android.content.Context
import android.util.Log
import com.samsung.videotraffic.model.ClassificationResult
import com.samsung.videotraffic.model.TrafficFeatures
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp

class VideoTrafficClassifier(private val context: Context) {

    private var interpreter: Interpreter? = null
    private var isModelLoaded = false

    companion object {
        private const val TAG = "VideoTrafficClassifier"
        private const val MODEL_FILE = "video_traffic_model.tflite"
        private const val INPUT_SIZE = TrafficFeatures.FEATURE_COUNT
        private const val OUTPUT_SIZE = 2 // Binary classification: video vs non-video
    }

    init {
        loadModel()
    }

    private fun loadModel() {
        try {
            // Try to load the TensorFlow Lite model
            val modelBuffer = loadModelFile()
            val options = Interpreter.Options()
            interpreter = Interpreter(modelBuffer, options)
            isModelLoaded = true
            Log.d(TAG, "TensorFlow Lite model loaded successfully")
        } catch (e: Exception) {
            Log.w(TAG, "Could not load TensorFlow Lite model: ${e.message}")
            Log.w(TAG, "Using fallback heuristic classifier")
            isModelLoaded = false
        }
    }

    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(MODEL_FILE)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun classify(features: TrafficFeatures): ClassificationResult {
        return if (isModelLoaded) {
            classifyWithTensorFlow(features)
        } else {
            classifyWithHeuristics(features)
        }
    }

    private fun classifyWithTensorFlow(features: TrafficFeatures): ClassificationResult {
        try {
            val interpreter = this.interpreter ?: return classifyWithHeuristics(features)

            // Prepare input
            val inputBuffer = ByteBuffer.allocateDirect(INPUT_SIZE * 4) // 4 bytes per float
            inputBuffer.order(ByteOrder.nativeOrder())
            
            val featureArray = TrafficFeatures.toFloatArray(features)
            val normalizedFeatures = normalizeFeatures(featureArray)
            
            for (value in normalizedFeatures) {
                inputBuffer.putFloat(value)
            }

            // Prepare output
            val outputBuffer = ByteBuffer.allocateDirect(OUTPUT_SIZE * 4)
            outputBuffer.order(ByteOrder.nativeOrder())

            // Run inference
            interpreter.run(inputBuffer, outputBuffer)

            // Parse output
            outputBuffer.rewind()
            val nonVideoProb = outputBuffer.float
            val videoProb = outputBuffer.float

            // Apply softmax to get proper probabilities
            val softmaxOutput = softmax(floatArrayOf(nonVideoProb, videoProb))
            
            val isVideo = softmaxOutput[1] > softmaxOutput[0]
            val confidence = if (isVideo) softmaxOutput[1] else softmaxOutput[0]

            return ClassificationResult(
                classification = if (isVideo) ClassificationResult.Classification.REEL
                else ClassificationResult.Classification.NON_REEL,
                confidence = confidence
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error in TensorFlow classification: ${e.message}")
            return classifyWithHeuristics(features)
        }
    }

    private fun classifyWithHeuristics(features: TrafficFeatures): ClassificationResult {
        try {
            // Heuristic-based classification as fallback
            var videoScore = 0f
            var totalScore = 0f

            // High bitrate suggests video
            if (features.bitrate > 1_000_000) { // 1 Mbps
                videoScore += 2f
            } else if (features.bitrate > 500_000) { // 500 kbps
                videoScore += 1f
            } else if (features.bitrate > 100_000) { // 100 kbps
                videoScore += 0.5f
            }
            totalScore += 2f

            // Large packet sizes suggest video
            if (features.packetSize > 1400) {
                videoScore += 1f
            } else if (features.packetSize > 500) {
                videoScore += 0.5f
            }
            totalScore += 1f

            // Consistent intervals suggest streaming
            if (features.packetSizeVariation < features.packetSize * 0.3f) {
                videoScore += 1f
            }
            totalScore += 1f

            // High data volume over time suggests video
            val avgBytesPerSecond = if (features.connectionDuration > 0) {
                features.dataVolume / (features.connectionDuration / 1000f)
            } else 0f
            
            if (avgBytesPerSecond > 100_000) { // 100 KB/s
                videoScore += 1f
            } else if (avgBytesPerSecond > 50_000) { // 50 KB/s
                videoScore += 0.5f
            }
            totalScore += 1f

            // Low burstiness suggests steady streaming
            if (features.burstiness < 2f) {
                videoScore += 1f
            }
            totalScore += 1f

            val confidence = if (totalScore > 0) videoScore / totalScore else 0.5f
            val isVideo = confidence > 0.5f

            Log.d(TAG, "Heuristic classification: bitrate=${features.bitrate}, packetSize=${features.packetSize}, confidence=$confidence")

            return ClassificationResult(
                classification = if (isVideo) ClassificationResult.Classification.REEL
                else ClassificationResult.Classification.NON_REEL,
                confidence = confidence
            )
            
        } catch (e: Exception) {
            Log.e(TAG, "Error in heuristic classification: ${e.message}")
            // Return unknown result as last resort
            return ClassificationResult(
                classification = ClassificationResult.Classification.UNKNOWN,
                confidence = 0.5f
            )
        }
    }

    private fun normalizeFeatures(features: FloatArray): FloatArray {
        // Simple min-max normalization
        // In a real implementation, you'd use the same normalization used during training
        return features.map { value ->
            when {
                value < 0 -> 0f
                value > 1_000_000 -> 1f // Cap at 1M for very large values
                else -> value / 1_000_000f
            }
        }.toFloatArray()
    }

    private fun softmax(input: FloatArray): FloatArray {
        val max = input.maxOrNull() ?: 0f
        val exp = input.map { exp(it - max) }
        val sum = exp.sum()
        return exp.map { it / sum }.toFloatArray()
    }

    fun release() {
        interpreter?.close()
        interpreter = null
        isModelLoaded = false
    }
}
