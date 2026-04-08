package com.quickai.service

import android.app.Application
import android.util.Log
import java.io.File

class QuickAIApplication : Application() {

    companion object {
        private const val TAG = "QuickAI"
    }

    override fun onCreate() {
        super.onCreate()

        // Ensure models directory exists
        val modelsDir = File(filesDir, "models")
        if (!modelsDir.exists()) {
            modelsDir.mkdirs()
            Log.i(TAG, "Created models directory: ${modelsDir.absolutePath}")
        }
    }
}
