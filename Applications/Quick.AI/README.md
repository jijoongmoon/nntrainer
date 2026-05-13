# Quick.AI⚡

Custom model extensions for [nntrainer](https://github.com/nntrainer/nntrainer) CausalLM application.

Build your own CausalLM models as **self-registering plugins** — no modification to nntrainer's source code required.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  nntrainer (submodule)                                       │
│  ├── main.cpp + Factory singleton                            │
│  │     ├── Qwen3ForCausalLM      (built-in)                  │
│  │     ├── GptOssForCausalLM     (built-in)                  │
│  │     └── ...                                               │
│  │                                                           │
│  quick-dot-ai (this repo)                                    │
│  ├── quick_dot_ai            (standalone executable)         │
│  └── libquick_dot_ai.so          (plugin for LD_PRELOAD)     │
│        __attribute__((constructor)) runs before main()       │
│        → Factory::Instance().registerModel(...)              │
└──────────────────────────────────────────────────────────────┘
```

Custom models use `__attribute__((constructor))` to register with the `Factory` singleton **before `main()` starts**. The standalone executable (`quick_dot_ai`) statically links the custom models via `link_whole`, so they are always available without `LD_PRELOAD`. A shared library is also built for plugin mode with the original `nntr_causallm`.

This means:

- nntrainer's `main.cpp` is used as-is — never copied or modified
- When nntrainer updates, nothing in this repo breaks
- Multiple custom models can be added independently

## Directory Structure

```
project-root/
├── nntrainer/                          # Shared nntrainer submodule (untouched)
├── meson.build                         # Unified root build (single project entry point)
├── meson.options                       # Build options (platform, enable-qnn, etc.)
├── build.sh                            # Unified build script (x86 + android)
├── install_android.sh                  # Unified android device installation
├── cross/
│   └── android-aarch64.cross.in        # NDK cross-compilation template
├── src/                                # CausalLM custom model extension
│   ├── models/
│   │   ├── meson.build                 # Lists model subdirectories
│   │   └── qnn-transformer/           # QNN transformer model (android only)
│   │       ├── qnn_transformer.cpp
│   │       └── meson.build
│   └── meson.build                     # src subdir build
├── qnn/                                # QNN context library (android only)
│   ├── qnn_context.cpp
│   ├── jni/                            # QNN SDK wrappers + RPC manager
│   └── meson.build
├── api/                                # C API for deploying models
│   ├── quick_dot_ai_api.h
│   ├── quick_dot_ai_api.cpp
│   ├── model_config.cpp
│   └── meson.build
└── api-app/                            # API test application
    ├── test_api.cpp
    └── meson.build
```

## How to Create a Custom Model

### 1. Define Your Model Class


```
Transformer          (base: embedding + decoder blocks + norm)
    ├── CausalLM     (adds LM head + generation logic)
```

Key virtual methods to override:
- `createAttention()` — Q/K/V projections, MHA configuration
- `createMlp()` — Feed-forward network
- `createTransformerDecoderBlock()` — Full decoder block
- `registerCustomLayers()` — Register custom nntrainer layers

### 2. Self-Register in the `.cpp` File

At the bottom of your `.cpp` file, add:

```cpp
__attribute__((constructor)) static void register_my_models() {
  causallm::Factory::Instance().registerModel(
    "MyModelForCausalLM",
    [](causallm::json cfg, causallm::json generation_cfg,
       causallm::json nntr_cfg) {
      return std::make_unique<causallm::MyModel>(
        cfg, generation_cfg, nntr_cfg);
    });
}
```

### 3. Configure Your Model

Create config files in `res/your_model/`:
- **config.json**: Set `"architectures": ["MyModelForCausalLM"]` (must match registered key)
- **generation_config.json**: Token IDs, sampling parameters
- **nntr_config.json**: NNTrainer settings (tensor types, sequence lengths, etc.)

### 4. Add to Build System

For a new model `models/my_model/`:

1. Create `models/my_model/meson.build`:
```meson
my_model_src = [meson.current_source_dir() / 'my_model.cpp']
my_model_inc = include_directories('.')
quick_dot_ai_src += my_model_src
quick_dot_ai_inc += my_model_inc
```

2. Add `subdir('my_model')` to `models/meson.build`

## Building

All builds are driven by the unified `build.sh` script at the project root.

### x86 / Linux

```bash
# Build all targets (src + api + api-test)
./build.sh

# Build only src (model library + executable)
./build.sh --target=src

# Clean rebuild
./build.sh --clean
```

Two ways to run:

```bash
# Standalone executable (recommended — custom models built in)
LD_LIBRARY_PATH=nntrainer/builddir_x86/nntrainer:nntrainer/builddir_x86/api/ccapi:builddir_x86/src:builddir_x86/api \
  builddir_x86/src/quick_dot_ai /path/to/model "Your prompt"

# Plugin mode (inject into existing nntr_causallm via LD_PRELOAD)
LD_PRELOAD=$(pwd)/builddir_x86/src/libquick_dot_ai.so nntr_causallm /path/to/model
```

### Android (arm64-v8a)

```bash
export ANDROID_NDK=/path/to/android-ndk

# Build all targets
./build.sh --platform=android

# Build with QNN support (android only)
./build.sh --platform=android --enable-qnn

# Build only src
./build.sh --platform=android --target=src

# Install to device
./install_android.sh

# Run on device
adb shell /data/local/tmp/Quick.AI/run.sh /path/to/model
```

### Build Options

| Option | Default | Description |
|---|---|---|
| `--platform=x86\|android` | `x86` | Target platform |
| `--target=src,api,api-test` | `all` | Comma-separated list of targets to build |
| `--enable-qnn` | off | Enable QNN integration (android only) |
| `--clean` | off | Clean rebuild from scratch |

Meson options (set via `-D` or in `meson.options`):

| Option | Default | Description |
|---|---|---|
| `platform` | `auto` | Target platform (`auto`, `x86`, `android`) |
| `enable-qnn` | `false` | Build QNN context lib + qnn-transformer model (android only) |
| `enable-fp16` | `true` | Enable FP16 support (effective on android/ARM only) |
| `enable-api` | `true` | Build `libquick_dot_ai_api.so` |
| `enable-api-test` | `true` | Build `quick_dot_ai_test` executable |

## Prerequisites

- C++17 compiler
- [Meson](https://mesonbuild.com/) >= 0.55.0
- [Ninja](https://ninja-build.org/)
- [Android NDK](https://developer.android.com/ndk) (for android builds)
- OpenBLAS (for x86 builds: `apt install libopenblas-dev`)
- nntrainer dependencies (see nntrainer documentation)
