// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    QuickAIService.kt
 * @brief   QuickAI's long-lived foreground service.
 *
 * Hosts:
 *  - An embedded loopback HTTP server (NanoHTTPD) bound to 127.0.0.1:3453
 *    exposing a REST surface that is a 1:1 mapping of the handle-based
 *    causal_lm_api.h entry points (see Architecture.md §2.4).
 *  - A ModelRegistry that maps model_id → ModelWorker, giving per-model
 *    FIFO execution and cross-model parallelism (Architecture.md §2.6).
 *
 * Runs in a `:remote` process (see AndroidManifest.xml) so it can be
 * bound by LauncherApp and by arbitrary client apps without pulling them
 * into the same process.
 */
package com.example.QuickAI

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.util.Log
import com.example.QuickAI.service.DEFAULT_QUICKAI_PORT
import com.example.QuickAI.service.HttpServer
import com.example.QuickAI.service.ModelRegistry
import com.example.QuickAI.service.NativeCausalLm

class QuickAIService : Service() {

    private lateinit var registry: ModelRegistry
    private lateinit var httpServer: HttpServer

    /**
     * @brief This service does not expose a bound interface — clients
     * talk to it over HTTP only. Returning null here keeps the binding
     * API surface minimal; Android still respects
     * `android:exported="true"` for `startService`-style cross-app
     * launches, and the remote REST endpoint is the real API surface.
     */
    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "QuickAIService onCreate (pid=${android.os.Process.myPid()})")

        // Best-effort native lib load — if it fails (e.g. emulator without
        // prebuilt .so files) the service still comes up and REST calls
        // that need the native library get a clean MODEL_LOAD_FAILED.
        NativeCausalLm.ensureLoaded()

        registry = ModelRegistry()
        httpServer = HttpServer(
            hostname = "127.0.0.1",
            preferredPort = DEFAULT_QUICKAI_PORT,
            registry = registry
        )

        // Promote to foreground BEFORE starting work so Android doesn't
        // ANR us for a delayed startForeground call.
        startForegroundWithNotification(port = -1)

        val port = httpServer.start()
        if (port > 0) {
            // Refresh the notification with the real port number.
            startForegroundWithNotification(port)
        } else {
            Log.e(TAG, "Failed to start HTTP server")
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // START_STICKY: if the system kills us, recreate as soon as
        // possible. The onCreate() flow re-opens the port.
        return START_STICKY
    }

    override fun onDestroy() {
        Log.i(TAG, "QuickAIService onDestroy")
        try {
            httpServer.stop()
        } catch (t: Throwable) {
            Log.w(TAG, "httpServer.stop threw", t)
        }
        try {
            registry.shutdownAll()
        } catch (t: Throwable) {
            Log.w(TAG, "registry.shutdownAll threw", t)
        }
        super.onDestroy()
    }

    // --- notification plumbing ----------------------------------------

    private fun startForegroundWithNotification(port: Int) {
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "QuickAI service",
                NotificationManager.IMPORTANCE_LOW
            )
            nm.createNotificationChannel(channel)
        }

        val text = if (port > 0) {
            "Listening on 127.0.0.1:$port"
        } else {
            "Starting…"
        }

        @Suppress("DEPRECATION") // fine for an internal low-priority notification
        val notification: Notification = Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("QuickAI")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.stat_notify_sync)
            .setOngoing(true)
            .build()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    companion object {
        private const val TAG = "QuickAIService"
        private const val CHANNEL_ID = "quickai_service"
        private const val NOTIFICATION_ID = 0x51434149 // "QCAI"
    }
}
