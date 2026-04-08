package com.quickai.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.util.Log

/**
 * Foreground service that runs the LLM HTTP server.
 * The service stays alive to handle REST API requests from clients.
 */
class LlmService : Service() {

    companion object {
        private const val TAG = "LlmService"
        private const val CHANNEL_ID = "quick_ai_service"
        private const val NOTIFICATION_ID = 1
        const val DEFAULT_PORT = 8080
        const val EXTRA_PORT = "port"
    }

    private var httpServer: LlmHttpServer? = null
    private var port: Int = DEFAULT_PORT

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "LlmService created")
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        port = intent?.getIntExtra(EXTRA_PORT, DEFAULT_PORT) ?: DEFAULT_PORT

        // Set model base path to app's files directory
        val modelsDir = "${filesDir.absolutePath}/models/"
        NativeEngine.nativeSetModelBasePath(modelsDir)
        Log.i(TAG, "Model base path: $modelsDir")

        startForeground(NOTIFICATION_ID, buildNotification())

        if (httpServer == null) {
            httpServer = LlmHttpServer(port).also {
                it.start()
                Log.i(TAG, "HTTP server started on port $port")
            }
        }

        return START_STICKY
    }

    override fun onDestroy() {
        Log.i(TAG, "LlmService destroying")
        httpServer?.stop()
        httpServer = null
        NativeEngine.nativeUnloadModel()
        super.onDestroy()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Quick.AI LLM Service",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "On-device LLM inference service"
            }
            getSystemService(NotificationManager::class.java)
                .createNotificationChannel(channel)
        }
    }

    private fun buildNotification(): Notification {
        return Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("Quick.AI Service")
            .setContentText("LLM service running on port $port")
            .setSmallIcon(android.R.drawable.ic_menu_manage)
            .setOngoing(true)
            .build()
    }
}
