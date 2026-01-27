# ☄️ CausalLM Inference with NNTrainer

This application provides a standalone executable and an optional C API to run causal LLM models using NNTrainer.
It supports *inference* mode (text generation) on various devices, including Android.

## Features

- **Standalone Application (`nntr_causallm`)**: A command-line tool to load models and generate text.
- **C API (Optional)**: A lightweight C interface (`libcausallm.so`) for integrating LLM capabilities into other applications (e.g., Android JNI, iOS, or other C/C++ apps).
- **Supported Backends**: CPU (OpenMP), with GPU/NPU support planned.

## Supported models

- Llama
- Qwen3 (0.6B, 1.7B, 4B, 7B, 14B, 32B) [[link](https://huggingface.co/Qwen/Qwen3-4B)]
- Qwen3-MoE (30B-A3B) [[link](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)]
- GPT-OSS (MoE: 20B, 120B) [[link](https://huggingface.co/openai/gpt-oss-20b)]
- You can try your own model with custom layers!
- Feel free to contribute! 😊

For more details, please refer to the [Model Documentation](models/README.md).

## CausalLM API

The CausalLM application exposes a C API for easy integration with other applications (e.g., Android JNI).
The API allows loading models, running inference, and retrieving performance metrics.

For detailed documentation, please refer to [API Documentation](api/README.md).

## How to run

### 1. Prepare Model Files
- Download and copy the model files from huggingface to `res/{model}` directory.
- The folder should contain:
    - `config.json`
    - `generation_config.json`
    - `tokenizer.json`
    - `tokenizer_config.json`
    - `vocab.json`
    - `nntr_config.json`
    - nntrainer weight binfile (matches with the name in `nntr_config.json`)

### 2. PC Build & Test

Compile the application with transformer support enabled.

```bash
$ meson build -Denable-fp16=true -Dthread-backend=omp -Denable-transformer=true -Domp-num-threads=4
$ ninja -C build
```

Run the model:

```bash
$ export OMP_THREAD_LIMIT=16 && export OMP_WAIT_POLICY=active && export OMP_PROC_BIND=true && export OMP_PLACES=cores && export OMP_NUM_THREADS=4
$ ./build/Applications/CausalLM/nntr_causallm {your model config folder}
```

e.g.,
```bash
$ ./build/Applications/CausalLM/nntr_causallm /tmp/nntrainer/Applications/CausalLM/res/qwen3-4b/
```

### 3. Android Build & Test

The Android build process has been updated to simplify the compilation of the CausalLM library and executable, along with its dependencies (nntrainer, tokenizers-cpp).

#### Prerequisites
- Android NDK (e.g., r21d or later)
- CMake
- Rust (for tokenizers-cpp)
- ADB (Android Debug Bridge)

#### Build Scripts

The following scripts are provided in `Applications/CausalLM/` to handle the build process:

1.  **`build_android.sh`**:
    - This is the main build script.
    - It automates the entire process:
        - Builds `nntrainer` core library for Android.
        - Builds `tokenizers-cpp` static library (`libtokenizers_android_c.a`) if missing.
        - Downloads `json.hpp` dependency.
        - Compiles `libcausallm.so` (Shared Library for API) and `nntrainer_causallm` (Executable).
    - **Usage**: `./build_android.sh`

2.  **`build_tokenizer_android.sh`**:
    - Specifically builds the `tokenizers-cpp` library for Android (arm64-v8a by default).
    - It clones `mlc-ai/tokenizers-cpp`, configures it with CMake, and builds it using Rust toolchain.
    - Called automatically by `build_android.sh` if needed.

3.  **`install_android.sh`**:
    - Installs the built artifacts to a connected Android device.
    - Pushes libraries (`libcausallm.so`, `libnntrainer.so`, etc.) and the executable.
    - Creates a helper script `run_causallm.sh` on the device.
    - **Usage**: `./install_android.sh`

#### Build Instructions

1.  **Set NDK Path**:
    ```bash
    export ANDROID_NDK=/path/to/your/android-ndk
    ```

2.  **Run Build Script**:
    ```bash
    cd Applications/CausalLM
    ./build_android.sh
    ```
    This will generate the following in `jni/libs/arm64-v8a/`:
    - `libcausallm.so`: The CausalLM API shared library.
    - `nntrainer_causallm`: The command-line executable.
    - `libnntrainer.so` & `libccapi-nntrainer.so`: NNTrainer core libraries.

3.  **Install to Device**:
    Connect your Android device via ADB.
    ```bash
    ./install_android.sh
    ```

4.  **Run on Device**:
    You need to push your model files to the device first (e.g., to `/data/local/tmp/nntrainer/causallm/models/`).
    
    ```bash
    adb shell
    cd /data/local/tmp/nntrainer/causallm
    ./run_causallm.sh ./models/your-model-folder
    ```