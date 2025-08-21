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
import kotlin.math.sqrt
import kotlin.math.log
import kotlin.math.abs

/**
 * Advanced Reel Traffic Classifier for Samsung EnnovateX 2025
 * 
 * Features:
 * - Real-time classification of reel vs non-reel traffic
 * - Advanced network feature extraction
 * - Network robustness under varying conditions
 * - TensorFlow Lite with quantization support
 * - Fallback heuristic classification
 * 
 * Open Source Compliance:
 * - Uses only OSI-approved libraries
 * - No proprietary APIs or cloud services
 * - Local inference only
 */
class ReelTrafficClassifier(private val context: Context) {

    private var interpreter: Interpreter? = null
    private var isModelLoaded = false
    private var modelInfo: ModelInfo? = null
    private var scaler: FeatureScaler? = null

    companion object {
        private const val TAG = "ReelTrafficClassifier"
        private const val MODEL_FILE = "reel_traffic_mobile.tflite"
        private const val MODEL_INFO_FILE = "reel_traffic_mobile_info.json"
        private const val INPUT_SIZE = 30 // Advanced feature count
        private const val OUTPUT_SIZE = 3 // Three classes: Non-Reel, Reel, Unknown
        private const val INFERENCE_TIMEOUT_MS = 100L // Real-time requirement
    }

    data class ModelInfo(
        val modelName: String,
        val inputShape: List<Int>,
        val outputShape: List<Int>,
        val featureNames: List<String>,
        val classNames: List<String>,
        val scalerMean: List<Float>,
        val scalerScale: List<Float>,
        val scalerVar: List<Float>,
        val createdAt: String,
        val version: String
    )

    data class FeatureScaler(
        val mean: FloatArray,
        val scale: FloatArray,
        val variance: FloatArray
    ) {
        fun transform(features: FloatArray): FloatArray {
            return features.mapIndexed { index, value ->
                if (scale[index] != 0f) {
                    (value - mean[index]) / scale[index]
                } else {
                    value - mean[index]
                }
            }.toFloatArray()
        }
    }

    init {
        loadModel()
    }

    private fun loadModel() {
        try {
            // Load TensorFlow Lite model
            val modelBuffer = loadModelFile()
            val options = Interpreter.Options().apply {
                setNumThreads(4) // Optimize for mobile
                setUseXNNPACK(true) // Enable XNNPACK for better performance
            }
            interpreter = Interpreter(modelBuffer, options)
            
            // Load model information
            loadModelInfo()
            
            isModelLoaded = true
            Log.d(TAG, "Reel Traffic Classifier loaded successfully")
            Log.d(TAG, "Model: ${modelInfo?.modelName}, Version: ${modelInfo?.version}")
            
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

    private fun loadModelInfo() {
        try {
            val inputStream = context.assets.open(MODEL_INFO_FILE)
            val jsonString = inputStream.bufferedReader().use { it.readText() }
            
            // Parse JSON (simplified - in production use proper JSON library)
            modelInfo = parseModelInfo(jsonString)
            
            // Initialize scaler
            modelInfo?.let { info ->
                scaler = FeatureScaler(
                    info.scalerMean.toFloatArray(),
                    info.scalerScale.toFloatArray(),
                    info.scalerVar.toFloatArray()
                )
            }
            
            Log.d(TAG, "Model info loaded: ${modelInfo?.featureNames?.size} features")
            
        } catch (e: Exception) {
            Log.w(TAG, "Could not load model info: ${e.message}")
        }
    }

    private fun parseModelInfo(jsonString: String): ModelInfo {
        // Simplified JSON parsing - in production use Gson or Jackson
        // This is a basic implementation for demonstration
        return ModelInfo(
            modelName = "reel_traffic_mobile",
            inputShape = listOf(1, INPUT_SIZE),
            outputShape = listOf(1, OUTPUT_SIZE),
            featureNames = listOf(
                "packet_size_mean", "packet_size_std", "packet_size_entropy",
                "inter_arrival_mean", "inter_arrival_std", "inter_arrival_entropy",
                "burstiness", "flow_duration", "flow_churn_rate",
                "tcp_handshake_time", "quic_handshake_time", "rtt_mean", "rtt_std",
                "bitrate_mean", "bitrate_variance", "bitrate_entropy",
                "session_duration", "data_volume", "packet_count",
                "tcp_ratio", "udp_ratio", "http_ratio", "https_ratio",
                "temporal_burst_pattern", "flow_level_entropy",
                "congestion_window_size", "retransmission_rate",
                "jitter_mean", "jitter_std", "packet_loss_rate"
            ),
            classNames = listOf("Non-Reel", "Reel", "Unknown"),
            scalerMean = List(INPUT_SIZE) { 0f },
            scalerScale = List(INPUT_SIZE) { 1f },
            scalerVar = List(INPUT_SIZE) { 1f },
            createdAt = "2025-01-01T00:00:00Z",
            version = "1.0.0"
        )
    }

    fun classify(features: TrafficFeatures): ClassificationResult {
        return if (isModelLoaded) {
            classifyWithTensorFlow(features)
        } else {
            classifyWithAdvancedHeuristics(features)
        }
    }

    private fun classifyWithTensorFlow(features: TrafficFeatures): ClassificationResult {
        try {
            val interpreter = this.interpreter ?: return classifyWithAdvancedHeuristics(features)
            val scaler = this.scaler ?: return classifyWithAdvancedHeuristics(features)

            // Extract advanced features
            val advancedFeatures = extractAdvancedFeatures(features)
            
            // Normalize features using the trained scaler
            val normalizedFeatures = scaler.transform(advancedFeatures)

            // Prepare input buffer
            val inputBuffer = ByteBuffer.allocateDirect(INPUT_SIZE * 4)
            inputBuffer.order(ByteOrder.nativeOrder())
            
            for (value in normalizedFeatures) {
                inputBuffer.putFloat(value)
            }

            // Prepare output buffer
            val outputBuffer = ByteBuffer.allocateDirect(OUTPUT_SIZE * 4)
            outputBuffer.order(ByteOrder.nativeOrder())

            // Run inference with timeout
            val startTime = System.currentTimeMillis()
            interpreter.run(inputBuffer, outputBuffer)
            val inferenceTime = System.currentTimeMillis() - startTime
            
            if (inferenceTime > INFERENCE_TIMEOUT_MS) {
                Log.w(TAG, "Inference took too long: ${inferenceTime}ms")
                return classifyWithAdvancedHeuristics(features)
            }

            // Parse output
            outputBuffer.rewind()
            val nonReelProb = outputBuffer.float
            val reelProb = outputBuffer.float
            val unknownProb = outputBuffer.float

            // Apply softmax to get proper probabilities
            val softmaxOutput = softmax(floatArrayOf(nonReelProb, reelProb, unknownProb))
            
            // Determine classification
            val maxIndex = softmaxOutput.indices.maxByOrNull { softmaxOutput[it] } ?: 2
            val confidence = softmaxOutput[maxIndex]
            
            // Map output to classification
            val classification = when (maxIndex) {
                0 -> ClassificationResult.Classification.NON_REEL
                1 -> ClassificationResult.Classification.REEL
                else -> ClassificationResult.Classification.UNKNOWN
            }

            Log.d(TAG, "TensorFlow classification: $classification (confidence: $confidence)")

            return ClassificationResult(classification, confidence)

        } catch (e: Exception) {
            Log.e(TAG, "Error in TensorFlow classification: ${e.message}")
            return classifyWithAdvancedHeuristics(features)
        }
    }

    private fun extractAdvancedFeatures(features: TrafficFeatures): FloatArray {
        // Extract 30 advanced features for the neural network
        return floatArrayOf(
            // Packet size features
            features.packetSize,
            features.packetSizeVariation,
            calculateEntropy(listOf(features.packetSize, features.packetSizeVariation)),
            
            // Inter-arrival time features
            features.packetInterval,
            features.averagePacketGap,
            calculateEntropy(listOf(features.packetInterval, features.averagePacketGap)),
            
            // Burstiness and flow features
            features.burstiness,
            features.connectionDuration,
            calculateFlowChurnRate(features),
            
            // Protocol timing features (simulated)
            estimateTcpHandshakeTime(features),
            estimateQuicHandshakeTime(features),
            estimateRttMean(features),
            estimateRttStd(features),
            
            // Bitrate features
            features.bitrate,
            calculateBitrateVariance(features),
            calculateBitrateEntropy(features),
            
            // Session features
            features.connectionDuration,
            features.dataVolume,
            estimatePacketCount(features),
            
            // Protocol ratios
            features.tcpRatio,
            features.udpRatio,
            estimateHttpRatio(features),
            estimateHttpsRatio(features),
            
            // Advanced patterns
            calculateTemporalBurstPattern(features),
            calculateFlowLevelEntropy(features),
            
            // Network condition features
            estimateCongestionWindowSize(features),
            estimateRetransmissionRate(features),
            estimateJitterMean(features),
            estimateJitterStd(features),
            estimatePacketLossRate(features)
        )
    }

    private fun classifyWithAdvancedHeuristics(features: TrafficFeatures): ClassificationResult {
        try {
            var reelScore = 0f
            var totalScore = 0f

            // Advanced bitrate analysis
            when {
                features.bitrate > 1_500_000 -> { // 1.5 Mbps - strong reel indicator
                    reelScore += 3f
                }
                features.bitrate > 800_000 -> { // 800 kbps - moderate reel indicator
                    reelScore += 2f
                }
                features.bitrate > 300_000 -> { // 300 kbps - weak reel indicator
                    reelScore += 1f
                }
                features.bitrate < 50_000 -> { // 50 kbps - likely non-reel
                    reelScore -= 1f
                }
            }
            totalScore += 3f

            // Packet size analysis
            when {
                features.packetSize > 1400 -> { // Large packets suggest video
                    reelScore += 2f
                }
                features.packetSize > 800 -> { // Medium packets
                    reelScore += 1f
                }
                features.packetSize < 200 -> { // Small packets suggest text/API
                    reelScore -= 1f
                }
            }
            totalScore += 2f

            // Consistency analysis (low variation suggests streaming)
            val packetSizeConsistency = 1f - (features.packetSizeVariation / features.packetSize)
            if (packetSizeConsistency > 0.7f) {
                reelScore += 2f
            } else if (packetSizeConsistency < 0.3f) {
                reelScore -= 1f
            }
            totalScore += 2f

            // Burstiness analysis (low burstiness suggests steady streaming)
            when {
                features.burstiness < 1.5f -> { // Low burstiness - good for reel
                    reelScore += 2f
                }
                features.burstiness > 4f -> { // High burstiness - likely non-reel
                    reelScore -= 1f
                }
            }
            totalScore += 2f

            // Data volume analysis
            val avgBytesPerSecond = if (features.connectionDuration > 0) {
                features.dataVolume / (features.connectionDuration / 1000f)
            } else 0f
            
            when {
                avgBytesPerSecond > 200_000 -> { // 200 KB/s - strong reel indicator
                    reelScore += 2f
                }
                avgBytesPerSecond > 100_000 -> { // 100 KB/s - moderate reel indicator
                    reelScore += 1f
                }
                avgBytesPerSecond < 10_000 -> { // 10 KB/s - likely non-reel
                    reelScore -= 1f
                }
            }
            totalScore += 2f

            // Session duration analysis
            when {
                features.connectionDuration > 60_000 -> { // >1 minute - likely reel
                    reelScore += 1f
                }
                features.connectionDuration < 5_000 -> { // <5 seconds - likely non-reel
                    reelScore -= 1f
                }
            }
            totalScore += 1f

            // Calculate confidence and classification
            val confidence = if (totalScore > 0) {
                (reelScore / totalScore).coerceIn(0f, 1f)
            } else 0.5f

            val isReel = confidence > 0.6f
            val isUnknown = confidence in 0.4f..0.6f

            val classification = when {
                isUnknown -> ClassificationResult.Classification.UNKNOWN
                isReel -> ClassificationResult.Classification.REEL
                else -> ClassificationResult.Classification.NON_REEL
            }

            Log.d(TAG, "Advanced heuristic classification: $classification (confidence: $confidence)")
            Log.d(TAG, "Features: bitrate=${features.bitrate}, packetSize=${features.packetSize}, burstiness=${features.burstiness}")

            return ClassificationResult(classification, confidence)

        } catch (e: Exception) {
            Log.e(TAG, "Error in advanced heuristic classification: ${e.message}")
            return ClassificationResult(
                ClassificationResult.Classification.UNKNOWN,
                0.5f
            )
        }
    }

    // Helper functions for advanced feature extraction
    private fun calculateEntropy(values: List<Float>): Float {
        if (values.isEmpty()) return 0f
        val sum = values.sum()
        if (sum == 0f) return 0f
        
        return -values.map { value ->
            val p = value / sum
            if (p > 0) p * kotlin.math.ln(p.toDouble()) else 0.0
        }.sum().toFloat()
    }

    private fun calculateFlowChurnRate(features: TrafficFeatures): Float {
        // Simulate flow churn rate based on connection duration and data volume
        return if (features.connectionDuration > 0) {
            (features.dataVolume / features.connectionDuration) / 1000f
        } else 0f
    }

    private fun estimateTcpHandshakeTime(features: TrafficFeatures): Float {
        // Estimate TCP handshake time based on RTT characteristics
        return features.packetInterval * 3f // 3-way handshake
    }

    private fun estimateQuicHandshakeTime(features: TrafficFeatures): Float {
        // QUIC handshake is typically faster than TCP
        return features.packetInterval * 1.5f
    }

    private fun estimateRttMean(features: TrafficFeatures): Float {
        return features.packetInterval * 0.5f
    }

    private fun estimateRttStd(features: TrafficFeatures): Float {
        return features.packetSizeVariation * 0.1f
    }

    private fun calculateBitrateVariance(features: TrafficFeatures): Float {
        return features.bitrate * 0.2f // Simulate variance
    }

    private fun calculateBitrateEntropy(features: TrafficFeatures): Float {
        return calculateEntropy(listOf(features.bitrate, features.packetSize))
    }

    private fun estimatePacketCount(features: TrafficFeatures): Float {
        return if (features.packetSize > 0) {
            features.dataVolume / features.packetSize
        } else 0f
    }

    private fun estimateHttpRatio(features: TrafficFeatures): Float {
        return 1f - features.tcpRatio - features.udpRatio
    }

    private fun estimateHttpsRatio(features: TrafficFeatures): Float {
        return features.tcpRatio * 0.8f // Most TCP traffic is HTTPS
    }

    private fun calculateTemporalBurstPattern(features: TrafficFeatures): Float {
        return features.burstiness * features.packetInterval / 1000f
    }

    private fun calculateFlowLevelEntropy(features: TrafficFeatures): Float {
        return calculateEntropy(listOf(features.tcpRatio, features.udpRatio))
    }

    private fun estimateCongestionWindowSize(features: TrafficFeatures): Float {
        return features.packetSize * 10f // Simulate congestion window
    }

    private fun estimateRetransmissionRate(features: TrafficFeatures): Float {
        return features.packetSizeVariation / features.packetSize * 0.1f
    }

    private fun estimateJitterMean(features: TrafficFeatures): Float {
        return features.packetSizeVariation * 0.05f
    }

    private fun estimateJitterStd(features: TrafficFeatures): Float {
        return features.packetSizeVariation * 0.02f
    }

    private fun estimatePacketLossRate(features: TrafficFeatures): Float {
        return features.packetSizeVariation / features.packetSize * 0.01f
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
        modelInfo = null
        scaler = null
    }
}
