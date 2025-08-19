package com.samsung.videotraffic.adapter

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.samsung.videotraffic.database.entity.MonitoringSession
import com.samsung.videotraffic.databinding.ItemSessionHistoryBinding
import java.text.SimpleDateFormat
import java.util.*

class SessionHistoryAdapter(
    private val onSessionClick: (String) -> Unit
) : ListAdapter<MonitoringSession, SessionHistoryAdapter.SessionViewHolder>(SessionDiffCallback()) {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): SessionViewHolder {
        val binding = ItemSessionHistoryBinding.inflate(
            LayoutInflater.from(parent.context), 
            parent, 
            false
        )
        return SessionViewHolder(binding)
    }

    override fun onBindViewHolder(holder: SessionViewHolder, position: Int) {
        holder.bind(getItem(position))
    }

    inner class SessionViewHolder(
        private val binding: ItemSessionHistoryBinding
    ) : RecyclerView.ViewHolder(binding.root) {

        init {
            binding.root.setOnClickListener {
                val position = adapterPosition
                if (position != RecyclerView.NO_POSITION) {
                    onSessionClick(getItem(position).id)
                }
            }
        }

        fun bind(session: MonitoringSession) {
            val dateFormat = SimpleDateFormat("MMM dd, yyyy 'at' HH:mm", Locale.getDefault())
            val duration = if (session.endTime != null) {
                val durationMs = session.endTime - session.startTime
                val minutes = durationMs / (1000 * 60)
                val seconds = (durationMs / 1000) % 60
                "${minutes}m ${seconds}s"
            } else {
                "Active"
            }

            binding.apply {
                textSessionDate.text = dateFormat.format(Date(session.startTime))
                textSessionDuration.text = duration
                textDataProcessed.text = formatBytes(session.totalBytesMonitored)
                textVideoDetections.text = "${session.videoDetections}"
                textNonVideoDetections.text = "${session.nonVideoDetections}"
                textUnknownDetections.text = "${session.unknownDetections}"
                
                // Calculate video percentage
                val totalDetections = session.videoDetections + session.nonVideoDetections + session.unknownDetections
                val videoPercentage = if (totalDetections > 0) {
                    (session.videoDetections * 100) / totalDetections
                } else {
                    0
                }
                textVideoPercentage.text = "${videoPercentage}%"
                
                // Set status indicator color
                val statusColor = when {
                    session.isActive -> android.graphics.Color.parseColor("#4CAF50") // Green
                    session.videoDetections > session.nonVideoDetections -> android.graphics.Color.parseColor("#FF9800") // Orange
                    else -> android.graphics.Color.parseColor("#2196F3") // Blue
                }
                statusIndicator.setBackgroundColor(statusColor)
            }
        }

        private fun formatBytes(bytes: Long): String {
            return when {
                bytes < 1024 -> "${bytes} B"
                bytes < 1024 * 1024 -> "${bytes / 1024} KB"
                bytes < 1024 * 1024 * 1024 -> "${bytes / (1024 * 1024)} MB"
                else -> "${bytes / (1024 * 1024 * 1024)} GB"
            }
        }
    }

    private class SessionDiffCallback : DiffUtil.ItemCallback<MonitoringSession>() {
        override fun areItemsTheSame(oldItem: MonitoringSession, newItem: MonitoringSession): Boolean {
            return oldItem.id == newItem.id
        }

        override fun areContentsTheSame(oldItem: MonitoringSession, newItem: MonitoringSession): Boolean {
            return oldItem == newItem
        }
    }
}
