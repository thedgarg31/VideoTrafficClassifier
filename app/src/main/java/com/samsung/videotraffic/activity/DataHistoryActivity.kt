package com.samsung.videotraffic.activity

import android.os.Bundle
import android.view.MenuItem
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.samsung.videotraffic.R

class DataHistoryActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        android.util.Log.d("DataHistoryActivity", "onCreate started")
        
        try {
            // Use a simple layout instead of complex binding
            setContentView(R.layout.activity_data_history_simple)
            android.util.Log.d("DataHistoryActivity", "Simple layout set successfully")

            setupActionBar()
            showTestData()
            android.util.Log.d("DataHistoryActivity", "onCreate completed successfully")
        } catch (e: Exception) {
            android.util.Log.e("DataHistoryActivity", "Error in onCreate", e)
            // Show error to user
            android.widget.Toast.makeText(this, "Error loading history: ${e.message}", android.widget.Toast.LENGTH_LONG).show()
            finish()
        }
    }

    private fun setupActionBar() {
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.title = "Traffic History"
    }

    private fun showTestData() {
        val testText = findViewById<TextView>(R.id.testText)
        testText?.text = "✅ Data History Activity Loaded Successfully!\n\n" +
                "📊 This screen will show:\n" +
                "• Session history\n" +
                "• Traffic analysis results\n" +
                "• Battery usage data\n" +
                "• Network statistics\n\n" +
                "🔧 Database integration coming soon!"
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
}
