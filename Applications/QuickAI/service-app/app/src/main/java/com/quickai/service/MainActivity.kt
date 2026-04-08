package com.quickai.service

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

/**
 * Simple activity to start/stop the LLM service
 */
class MainActivity : AppCompatActivity() {

    private var isServiceRunning = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val layout = android.widget.LinearLayout(this).apply {
            orientation = android.widget.LinearLayout.VERTICAL
            setPadding(48, 48, 48, 48)
        }

        val statusText = TextView(this).apply {
            text = "Quick.AI Service - Stopped"
            textSize = 20f
        }
        layout.addView(statusText)

        val infoText = TextView(this).apply {
            text = "\nREST API: http://localhost:${LlmService.DEFAULT_PORT}/v1/\n" +
                   "\nEndpoints:\n" +
                   "  GET  /v1/health\n" +
                   "  GET  /v1/models\n" +
                   "  POST /v1/engine/load\n" +
                   "  POST /v1/engine/unload\n" +
                   "  POST /v1/generate\n" +
                   "  GET  /v1/metrics\n"
            textSize = 14f
            setPadding(0, 24, 0, 24)
        }
        layout.addView(infoText)

        val toggleButton = Button(this).apply {
            text = "Start Service"
            setOnClickListener {
                if (isServiceRunning) {
                    stopService(Intent(this@MainActivity, LlmService::class.java))
                    statusText.text = "Quick.AI Service - Stopped"
                    this.text = "Start Service"
                    isServiceRunning = false
                } else {
                    val intent = Intent(this@MainActivity, LlmService::class.java)
                    startForegroundService(intent)
                    statusText.text = "Quick.AI Service - Running (port ${LlmService.DEFAULT_PORT})"
                    this.text = "Stop Service"
                    isServiceRunning = true
                }
            }
        }
        layout.addView(toggleButton)

        setContentView(layout)
    }
}
