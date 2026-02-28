Overview
--------

​This directory contains a specialized Python pipeline designed to export Hugging Face's Open Model such as **Qwen2.5 (1.5B)** Causal LM into [NNTrainer](https://github.com/nntrainer/nntrainer) compatible formats.

​Instead of dealing with dynamic Python control flows and parsing overhead at runtime, this script utilizes torch.fx to statically trace the model's computational graph and directly generates both an NNTrainer .ini configuration and **ready-to-compile C++ Builder Classes (****.h****, ****.cpp****)** for seamless Android NDK integration.


​Core Engineering Solutions
---------------------------

​Exporting complex LLMs from Hugging Face via torch.fx typically results in severe tracing failures (e.g., ValueError: code: co_varnames is too small) due to dynamic attention mask generation and KV cache branching (if/else) inside the Python code.

​This script solves these inherent compiler limitations through:

1.  ​**Static Monkey Patching:** We surgically override the forward functions of Qwen2Model and Qwen2ForCausalLM in memory before tracing. This strips away dynamic caching, mask generation, and dictionary packaging, leaving a pure, static mathematical pipeline.
2.  ​**Granular Graph Partitioning:** Instead of treating the massive Qwen2DecoderLayer as a single black box, the custom QwenTracer dives inside to preserve Attention and RMSNorm as custom NNTrainer layers, while automatically splitting the MLP block into fundamental C++ layers (fully_connected, activation, multiplication, addition).
3.  ​**Zero-Overhead C++ Generation:** To maximize initialization speed on edge devices (Android NDK), the script bypasses .ini file I/O at runtime by auto-generating a C++ class that hardcodes the NNTrainer createLayer and addLayer API calls.

​Requirements
-------------

​Ensure you have the following installed in your Python environment:

``` bash
pip install torch transformers
```

To enable automatic code formatting for the generated C++ files, you must install clang-format on your system:

-   ​**Ubuntu/Debian:** sudo apt-get install clang-format
-   ​**Windows (Chocolatey):** choco install llvm
-   ​**macOS:** brew install clang-format

​What the Script Does
---------------------

​When executed, the script performs the following steps sequentially:

1.  ​**Monkey Patches** the HF Qwen model to remove dynamic control flows.
2.  ​**Traces** the model using a custom torch.fx.Tracer to extract a clean static graph.
3.  ​**Maps** PyTorch fx.Node operations to NNTrainer layer specifications.
4.  ​**Generates** .ini, .h, and .cpp files representing the model architecture.
5.  ​**Formats** the C++ files using Google's C++ style guidelines via clang-format.

​Generated Output Files
-----------------------

​The script will create an output/ (or nntrainer_qwen_build/) directory containing:

-   ​qwen2_model.ini: The text-based topology configuration for NNTrainer. (Useful for debugging and verification).
-   ​qwen2_model.h: The C++ header file defining the hybrid_ai::Qwen2Model class.
-   ​qwen2_model.cpp: The implementation file containing the NNTrainer C++ API calls (ml::train::createLayer) mapped exactly to the traced PyTorch nodes.

​Usage
------

1.  ​Download the target Qwen model weights to your local directory (e.g., ./qwen2.5-1.5b-local).
2.  ​Run the exporter script:

``` bash
python export_qwen_to_nntrainer.py
```

1.  ​Check the console output to verify that clang-format was applied successfully.
2.  ​Copy the generated qwen2_model.h and qwen2_model.cpp into your Android NDK (jni/) project folder.

### ​Integration Example (Android JNI)

``` c++
#include "qwen2_model.h"

// Initialize the model inside your JNI environment

hybrid_ai::Qwen2Model* qwen_model = new hybrid_ai::Qwen2Model();

// Build the graph using the auto-generated layers

qwen_model->build();

// Load weights (Requires the .bin weight extraction step)

qwen_model->loadWeights("/data/local/tmp/qwen_weights.bin");

```

​TODO
------
Make it works for general open model.

