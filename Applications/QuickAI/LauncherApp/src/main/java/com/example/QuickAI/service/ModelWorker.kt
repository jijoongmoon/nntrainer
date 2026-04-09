// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    ModelWorker.kt
 * @brief   One Thread + one FIFO queue dedicated to a single loaded model.
 *
 * All work touching a given model flows through its own ModelWorker, so
 * requests for the same model from different client apps are strictly
 * serialized in arrival order. Workers for different models run in
 * parallel — so two different models can execute concurrently on separate
 * native handles.
 *
 * See Architecture.md §2.6 for the overall concurrency story.
 */
package com.example.QuickAI.service

import android.util.Log
import com.example.quickdotai.BackendResult
import com.example.quickdotai.QuickDotAI
import com.example.quickdotai.StreamSink
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit

/**
 * @brief A pending unit of work targeting a specific model.
 *
 * The worker thread pulls jobs off the queue in arrival order and
 * processes them one at a time. Each job carries its own completion
 * callback so the HTTP server thread can block on it with a timeout.
 */
internal sealed class Job {
    /**
     * @brief Bootstrap job injected by ModelWorker.start() so that the
     * blocking model load executes on the worker thread, keeping every
     * backend call on a single thread.
     */
    class Load(val onResult: (BackendResult<Unit>) -> Unit) : Job()

    /** Inference (runModel). */
    class Run(
        val prompt: String,
        val onResult: (BackendResult<String>) -> Unit
    ) : Job()

    /**
     * @brief Streaming inference. Deltas + terminal events are delivered
     * through [sink] on the worker thread (or on the backend's own
     * streaming thread for LiteRT-LM, which is thread-safe against the
     * sink's backing queue). ModelWorker guarantees that exactly one of
     * [StreamSink.onDone] or [StreamSink.onError] is delivered, even on
     * exceptional paths (backend throw / shutdown).
     */
    class RunStream(
        val prompt: String,
        val sink: StreamSink
    ) : Job()

    /** Metrics fetch. */
    class Metrics(
        val onResult: (BackendResult<PerformanceMetrics>) -> Unit
    ) : Job()

    /** Sentinel used to shut the worker down cleanly. */
    object Shutdown : Job()
}

/**
 * @brief Worker for a single loaded model.
 *
 * Lifecycle:
 *  1. Constructed with a pre-created (but not yet loaded) [QuickDotAI].
 *  2. [start] kicks off the background thread and synchronously loads the
 *     model. Returns the load result.
 *  3. Clients call [submitRun] / [submitMetrics] which enqueue a job.
 *  4. [shutdown] posts a sentinel and waits for the worker thread to exit.
 *
 * The queue is bounded to [capacity] to give backpressure: if full, the
 * submit methods fail fast with [QuickAiError.QUEUE_FULL] so the HTTP
 * handler can return 503 without blocking the server thread indefinitely.
 */
class ModelWorker(
    val modelId: String,
    private val loadRequest: LoadModelRequest,
    private val backend: QuickDotAI,
    private val capacity: Int = DEFAULT_CAPACITY
) {
    private val queue = LinkedBlockingQueue<Job>(capacity)

    @Volatile
    private var running: Boolean = false

    private val thread: Thread = Thread({ runLoop() }, "QuickAI-Worker-$modelId")

    val backendKind: String get() = backend.kind
    val architecture: String? get() = backend.architecture

    /**
     * @brief Start the worker thread and synchronously perform the model
     * load. Returns the load outcome.
     */
    fun start(): BackendResult<Unit> {
        running = true
        thread.isDaemon = true
        thread.start()

        // Loading can be expensive (seconds), so do it on the worker
        // thread and block here for the result. Subsequent work then flows
        // through the queue without blocking on load.
        val latch = java.util.concurrent.CountDownLatch(1)
        var outcome: BackendResult<Unit> = BackendResult.Err(QuickAiError.UNKNOWN)
        // Enqueue a Load job so that the blocking model load runs on the
        // worker thread, keeping all backend calls single-threaded.
        queue.put(Job.Load { result ->
            outcome = result
            latch.countDown()
        })

        // Wait up to 5 minutes for load to finish.
        if (!latch.await(5, TimeUnit.MINUTES)) {
            return BackendResult.Err(
                QuickAiError.MODEL_LOAD_FAILED,
                "Timed out waiting for model load"
            )
        }
        return outcome
    }

    /**
     * @brief Enqueue a run request. Returns false if the queue is full.
     * The [onResult] callback is invoked from the worker thread.
     */
    fun submitRun(prompt: String, onResult: (BackendResult<String>) -> Unit): Boolean {
        return queue.offer(Job.Run(prompt, onResult))
    }

    /**
     * @brief Enqueue a streaming run request. Returns false if the queue
     * is full — in that case the caller should return 503 WITHOUT calling
     * any method on [sink], because the sink has not been handed off to
     * the worker yet.
     *
     * On successful enqueue, ModelWorker owns [sink] and will deliver
     * exactly one terminal event (done or error) before returning to the
     * queue loop.
     */
    fun submitRunStream(prompt: String, sink: StreamSink): Boolean {
        return queue.offer(Job.RunStream(prompt, sink))
    }

    /**
     * @brief Enqueue a metrics fetch. Returns false if the queue is full.
     */
    fun submitMetrics(onResult: (BackendResult<PerformanceMetrics>) -> Unit): Boolean {
        return queue.offer(Job.Metrics(onResult))
    }

    /**
     * @brief Post the shutdown sentinel and join the worker thread. Any
     * still-pending jobs get dropped with a NOT_INITIALIZED error so
     * callers don't hang forever.
     */
    fun shutdown() {
        running = false
        queue.put(Job.Shutdown)
        try {
            thread.join(10_000)
        } catch (ie: InterruptedException) {
            Thread.currentThread().interrupt()
        }
    }

    // --- internals -----------------------------------------------------

    private fun runLoop() {
        try {
            while (running) {
                val job = try {
                    queue.take()
                } catch (ie: InterruptedException) {
                    Thread.currentThread().interrupt()
                    break
                }

                when (job) {
                    is Job.Load -> {
                        val r = try {
                            backend.load(loadRequest)
                        } catch (t: Throwable) {
                            BackendResult.Err(
                                QuickAiError.MODEL_LOAD_FAILED,
                                t.message
                            )
                        }
                        job.onResult(r)
                        // If load failed, stop the loop so we don't process
                        // any queued jobs against an un-loaded backend.
                        if (r is BackendResult.Err) {
                            running = false
                        }
                    }
                    is Job.Run -> {
                        val r = try {
                            backend.run(job.prompt)
                        } catch (t: Throwable) {
                            BackendResult.Err(
                                QuickAiError.INFERENCE_FAILED,
                                t.message
                            )
                        }
                        job.onResult(r)
                    }
                    is Job.RunStream -> {
                        // Guarantee exactly one terminal event on the
                        // sink, even if the backend throws or forgets to
                        // close the stream. We wrap the sink in a thin
                        // filter that remembers whether onDone/onError
                        // has been seen and fills in a final onError in
                        // the finally block if not.
                        val sink = job.sink
                        var closed = false
                        val guard = object : StreamSink {
                            override fun onDelta(text: String) = sink.onDelta(text)
                            override fun onDone() {
                                closed = true
                                sink.onDone()
                            }
                            override fun onError(
                                error: QuickAiError,
                                message: String?
                            ) {
                                closed = true
                                sink.onError(error, message)
                            }
                        }
                        try {
                            backend.runStreaming(job.prompt, guard)
                        } catch (t: Throwable) {
                            Log.e(TAG, "runStreaming threw for $modelId", t)
                            if (!closed) {
                                guard.onError(
                                    QuickAiError.INFERENCE_FAILED,
                                    t.message
                                )
                            }
                        } finally {
                            // Ensure the downstream HTTP chunked writer
                            // always gets a terminal frame so it can
                            // close the socket.
                            if (!closed) {
                                guard.onError(
                                    QuickAiError.UNKNOWN,
                                    "stream terminated without completion"
                                )
                            }
                        }
                    }
                    is Job.Metrics -> {
                        val r = try {
                            backend.metrics()
                        } catch (t: Throwable) {
                            BackendResult.Err(QuickAiError.UNKNOWN, t.message)
                        }
                        job.onResult(r)
                    }
                    is Job.Shutdown -> {
                        break
                    }
                }
            }
        } finally {
            drainAndFail()
            try {
                backend.close()
            } catch (t: Throwable) {
                Log.w(TAG, "backend.close() threw for $modelId", t)
            }
        }
    }

    /**
     * @brief On shutdown, fail every still-queued job so callers waking up
     * on their latches see a clear error instead of blocking forever.
     */
    private fun drainAndFail() {
        while (true) {
            val job = queue.poll() ?: break
            when (job) {
                is Job.Load -> job.onResult(
                    BackendResult.Err(QuickAiError.NOT_INITIALIZED, "worker shutting down")
                )
                is Job.Run -> job.onResult(
                    BackendResult.Err(QuickAiError.NOT_INITIALIZED, "worker shutting down")
                )
                is Job.RunStream -> job.sink.onError(
                    QuickAiError.NOT_INITIALIZED,
                    "worker shutting down"
                )
                is Job.Metrics -> job.onResult(
                    BackendResult.Err(QuickAiError.NOT_INITIALIZED, "worker shutting down")
                )
                Job.Shutdown -> { /* ignore */ }
            }
        }
    }

    companion object {
        private const val TAG = "ModelWorker"
        const val DEFAULT_CAPACITY = 32
    }
}
