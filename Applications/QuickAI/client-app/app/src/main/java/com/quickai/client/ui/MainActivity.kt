package com.quickai.client.ui

import android.os.Bundle
import android.widget.FrameLayout
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.bottomnavigation.BottomNavigationView
import com.quickai.client.api.ServiceClient

/**
 * Main activity with tab navigation:
 * Chat | Models | Benchmark | API Test
 */
class MainActivity : AppCompatActivity() {

    lateinit var client: ServiceClient
    private lateinit var container: FrameLayout

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        client = ServiceClient("localhost", 8080)

        val rootLayout = android.widget.LinearLayout(this).apply {
            orientation = android.widget.LinearLayout.VERTICAL
            layoutParams = android.widget.LinearLayout.LayoutParams(
                android.widget.LinearLayout.LayoutParams.MATCH_PARENT,
                android.widget.LinearLayout.LayoutParams.MATCH_PARENT
            )
        }

        container = FrameLayout(this).apply {
            id = android.view.View.generateViewId()
            layoutParams = android.widget.LinearLayout.LayoutParams(
                android.widget.LinearLayout.LayoutParams.MATCH_PARENT,
                0, 1f
            )
        }
        rootLayout.addView(container)

        val navView = BottomNavigationView(this).apply {
            menu.add(0, 0, 0, "Chat").setIcon(android.R.drawable.ic_menu_edit)
            menu.add(0, 1, 1, "Models").setIcon(android.R.drawable.ic_menu_agenda)
            menu.add(0, 2, 2, "Benchmark").setIcon(android.R.drawable.ic_menu_recent_history)
            menu.add(0, 3, 3, "API Test").setIcon(android.R.drawable.ic_menu_manage)
            setOnItemSelectedListener { item ->
                when (item.itemId) {
                    0 -> showFragment(ChatFragment())
                    1 -> showFragment(ModelsFragment())
                    2 -> showFragment(BenchmarkFragment())
                    3 -> showFragment(ApiTestFragment())
                }
                true
            }
        }
        rootLayout.addView(navView)

        setContentView(rootLayout)

        // Show Chat tab by default
        showFragment(ChatFragment())
    }

    private fun showFragment(fragment: androidx.fragment.app.Fragment) {
        supportFragmentManager.beginTransaction()
            .replace(container.id, fragment)
            .commit()
    }
}
