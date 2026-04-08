// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    LauncherApp.kt
 * @brief   Minimal launcher UI whose only job is to start the
 *          QuickAIService foreground service so the REST endpoint is
 *          available system-wide. See Architecture.md §2.1.
 *
 * UI is built in code to keep the project self-contained and free of
 * resource XML boilerplate.
 */
package com.example.QuickAI

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.example.QuickAI.service.DEFAULT_QUICKAI_PORT

class LauncherApp : AppCompatActivity() {

    private lateinit var statusText: TextView

    private val requestNotificationPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { _ -> startQuickAiService() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val pad = (resources.displayMetrics.density * 24).toInt()
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad, pad, pad)
            gravity = Gravity.TOP
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT)
        }

        val title = TextView(this).apply {
            text = "QuickAI"
            textSize = 24f
        }
        root.addView(title)

        statusText = TextView(this).apply {
            text = "REST endpoint: http://127.0.0.1:$DEFAULT_QUICKAI_PORT\nService: starting…"
            setPadding(0, pad / 2, 0, pad / 2)
        }
        root.addView(statusText)

        val startBtn = Button(this).apply {
            text = "Start QuickAI service"
            layoutParams = LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT)
            setOnClickListener { ensurePermissionAndStartService() }
        }
        root.addView(startBtn)

        val stopBtn = Button(this).apply {
            text = "Stop QuickAI service"
            layoutParams = LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT)
            setOnClickListener {
                stopService(Intent(this@LauncherApp, QuickAIService::class.java))
                statusText.text = "Service: stopped"
            }
        }
        root.addView(stopBtn)

        setContentView(root)

        ensurePermissionAndStartService()
    }

    /**
     * @brief On Android 13+ we need POST_NOTIFICATIONS at runtime, because
     * the foreground service in QuickAIService posts an ongoing
     * notification to keep itself alive.
     */
    private fun ensurePermissionAndStartService() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            val granted = ContextCompat.checkSelfPermission(
                this, Manifest.permission.POST_NOTIFICATIONS
            ) == PackageManager.PERMISSION_GRANTED
            if (!granted) {
                requestNotificationPermission.launch(
                    Manifest.permission.POST_NOTIFICATIONS
                )
                return
            }
        }
        startQuickAiService()
    }

    private fun startQuickAiService() {
        val intent = Intent(this, QuickAIService::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(intent)
        } else {
            startService(intent)
        }
        statusText.text =
            "REST endpoint: http://127.0.0.1:$DEFAULT_QUICKAI_PORT\nService: start requested"
    }
}
