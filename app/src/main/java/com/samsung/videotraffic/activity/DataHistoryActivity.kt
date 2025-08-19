package com.samsung.videotraffic.activity

import android.os.Bundle
import android.view.MenuItem
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.Observer
import androidx.recyclerview.widget.LinearLayoutManager
import com.samsung.videotraffic.R
import com.samsung.videotraffic.adapter.SessionHistoryAdapter
import com.samsung.videotraffic.databinding.ActivityDataHistoryBinding
import com.samsung.videotraffic.repository.TrafficDataRepository
import kotlinx.coroutines.*
import java.text.SimpleDateFormat
import java.util.*

class DataHistoryActivity : AppCompatActivity() {

    private lateinit var binding: ActivityDataHistoryBinding
    private lateinit var repository: TrafficDataRepository
    private lateinit var sessionAdapter: SessionHistoryAdapter
    private val scope = CoroutineScope(Dispatchers.Main + SupervisorJob())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityDataHistoryBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupActionBar()
        setupRecyclerView()
        setupRepository()
        loadHistoryData()
        setupRefreshListener()
    }

    private fun setupActionBar() {
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.title = "Traffic History"
    }

    private fun setupRecyclerView() {
        sessionAdapter = SessionHistoryAdapter { sessionId ->
            // Open session details
            // TODO: Implement session detail view
        }
        binding.recyclerViewHistory.apply {
            layoutManager = LinearLayoutManager(this@DataHistoryActivity)
            adapter = sessionAdapter
        }
    }

    private fun setupRepository() {
        repository = TrafficDataRepository.getInstance(this)
    }

    private fun loadHistoryData() {
        scope.launch {
            try {
                binding.progressBar.visibility = android.view.View.VISIBLE
                binding.textNoData.visibility = android.view.View.GONE
                
                val sessions = withContext(Dispatchers.IO) {
                    repository.getAllSessionsList()
                }
                
                if (sessions.isEmpty()) {
                    binding.textNoData.visibility = android.view.View.VISIBLE
                    binding.recyclerViewHistory.visibility = android.view.View.GONE
                } else {
                    binding.textNoData.visibility = android.view.View.GONE
                    binding.recyclerViewHistory.visibility = android.view.View.VISIBLE
                    sessionAdapter.submitList(sessions)
                }
                
                binding.progressBar.visibility = android.view.View.GONE
                
            } catch (e: Exception) {
                binding.progressBar.visibility = android.view.View.GONE
                binding.textNoData.visibility = android.view.View.VISIBLE
                binding.textNoData.text = "Error loading data: ${e.message}"
            }
        }
    }

    private fun setupRefreshListener() {
        binding.swipeRefreshLayout.setOnRefreshListener {
            loadHistoryData()
            binding.swipeRefreshLayout.isRefreshing = false
        }
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            android.R.id.home -> {
                finish()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        scope.cancel()
    }
}
