import os
import subprocess
import torch
import torch.nn as nn
import operator
from torch.fx import Tracer, GraphModule
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2ForCausalLM

# ==========================================
# 1. Monkey Patch (Remove Dynamic Logic)
# ==========================================
def custom_qwen2_model_forward(self, input_ids=None):

    inputs_embeds = self.embed_tokens(input_ids)
        
    hidden_states = inputs_embeds
    
    for decoder_layer in self.layers:
        layer_outputs = decoder_layer(
            hidden_states, attention_mask=None, position_ids=None,
            past_key_value=None, output_attentions=False, use_cache=False,
        )
        hidden_states = layer_outputs[0]
    hidden_states = self.norm(hidden_states)
    return tuple([hidden_states])

def custom_qwen2_causal_lm_forward(self, input_ids=None):
    outputs = self.model(input_ids=input_ids)
    logits = self.lm_head(outputs[0])
    return tuple([logits])

Qwen2Model.forward = custom_qwen2_model_forward
Qwen2ForCausalLM.forward = custom_qwen2_causal_lm_forward

# ==========================================
# 2. Custom Tracer (Attention, Norm)
# ==========================================
class QwenTracer(Tracer):
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        if super().is_leaf_module(m, module_qualified_name):
            return True
        class_name = type(m).__name__
        # Split Qwen2DecoderLayer and Remain Attention & RMSNorm
        if 'Attention' in class_name or 'RMSNorm' in class_name:
            return True
        return False

# =====================================================================
# 3. NNTrainer Mapping ( Brief, Not exact. This is just for the Test )
# ====================================================================
def export_to_granular_nntrainer_config(fx_graph: torch.fx.GraphModule):
    nntrainer_layers = []
    node_to_layer_name = {}

    for node in fx_graph.graph.nodes:
        
        print(node.name)
        layer_cfg = {"name": node.name}
        
        if node.op == 'placeholder':
            layer_cfg["type"] = "input"
            layer_cfg["input_shape"]="1:1:1:1024"
            nntrainer_layers.append(layer_cfg)
            node_to_layer_name[node.name] = node.name
            continue

        
        # Find Previous Node Name and set input_layers
        input_nodes = []
        for arg in node.args:
            if isinstance(arg, torch.fx.Node):
                # ig getitem, then get previous real layer name
                input_nodes.append(node_to_layer_name.get(arg.name, arg.name))
        
        if input_nodes:
            layer_cfg["input_layers"] = ", ".join(input_nodes)

        # Call Module (Linear, Attention, Norm, Activation)
        if node.op == 'call_module':
            target_mod = fx_graph.get_submodule(node.target)
            mod_class_name = type(target_mod).__name__
            
            if mod_class_name == 'Embedding':
                layer_cfg["type"] = "embedding"
                layer_cfg["in_dim"] = target_mod.num_embeddings
                layer_cfg["out_dim"] = target_mod.embedding_dim
                
            elif mod_class_name == 'Linear':
                layer_cfg["type"] = "fully_connected"
                layer_cfg["unit"] = target_mod.out_features
                layer_cfg["disable_bias"] = "true" if target_mod.bias is None else "false"
                
            elif 'RMSNorm' in mod_class_name:
                layer_cfg["type"] = "rms_norm"
                layer_cfg["epsilon"] = target_mod.variance_epsilon if hasattr(target_mod, 'variance_epsilon') else 1e-6
                
            elif 'Attention' in mod_class_name:
                # NNTrainer Custom Attention
                layer_cfg["type"] = "qwen_attention" 
                
            elif mod_class_name == 'SiLU':
                layer_cfg["type"] = "activation"
                layer_cfg["activation"] = "silu"

        # Python Internal Ops (+, *, getitem)
        elif node.op == 'call_function':
            if node.target == operator.add:
                layer_cfg["type"] = "addition"
            elif node.target == operator.mul:
                layer_cfg["type"] = "multiplication"
            elif node.target == operator.getitem:
                # Need to upgrade
                prev_node = node.args[0]
                node_to_layer_name[node.name] = node_to_layer_name.get(prev_node.name, prev_node.name)
                continue

        # Adding 
        if "type" in layer_cfg:
            nntrainer_layers.append(layer_cfg)
            node_to_layer_name[node.name] = node.name

    return nntrainer_layers

def generate_cpp_class(nntrainer_layers, class_name="ClassName"):
    # ==========================================
    # 1. Create Header (.h)
    # ==========================================
    header_lines = [
        f"// Auto-generated Header for {class_name}",
        f"#ifndef __{class_name.upper()}_H__",
        f"#ifdef __{class_name.upper()}_H__",
        "",
        "#ifdef _WIN32",
        "#define WIN_EXPORT __declspec(dllexport)",
        "#define WSTR std::wstring",
        "#define WCHAR_P wchar_t *",
        "#else",
        "#define WIN_EXPORT",
        "#define WSTR std::string",
        "#define WCHAR_P std::string &",
        "#endif",
        "",
        "#include <model.h>",
        "#include <layer.h>",
        "#include <memory>",
        "#include <string>",
        "#include <vector>",
        "",
        "namespace nntrainer {",
        "",
        f"WIN_EXPORT class {class_name} {{",
        "private:",
        "    std::unique_ptr<ml::train::Model> model;",
        "    bool is_compiled;",
        "",
        "public:",
        f"    {class_name}();",
        f"    ~{class_name}() = default;",
        "",
        "    // Construct Graph",
        "    int build();",
        "",
        "    // Loading",
        "    int load_weight(const std::string& bin_file_path);",
        "",
        "    // Inference",
        "    std::vector<float> run(const std::vector<float>& input_ids);",
        "};",
        "",
        "} // namespace nntrainer",
        "#endif"
    ]
    header_code = "\n".join(header_lines)

    # ==========================================
    # 2. Create Source (.cpp)
    # ==========================================
    cpp_lines = [
        f"// Auto-generated Source for {class_name}",

        f"#include \"{class_name.lower()}.h\"",
        "#include <app_context.h>",
        "#include <engine.h>",
        "#include <iostream>",
        "",
        "namespace nntrainer {",
        "",
        "using namespace ml::train;",
        "",
        f"{class_name}::{class_name}() : is_compiled(false) {{",
        "    model = createModel(ModelType::NEURAL_NET);",
        "}",
        "",
        f"int {class_name}::build() {{",
        "    if (!model) {",
        "        std::cerr << \"Model pointer is null\" << std::endl;",
        "        return -1;",
        "    }",
        "    int status;"
    ]

    # C++ Code Generation for layers
    for layer in nntrainer_layers:
        var_name = layer["name"].replace(".", "_")
        layer_type = layer["type"]
        
        props = [f'"name={var_name}"']
        for key, value in layer.items():
            if key not in ["name", "type"]:
                props.append(f'"{key}={value}"')
        
        props_str = ", ".join(props)
        
        cpp_lines.append(f"    // Layer: {var_name} ({layer_type})")
        cpp_lines.append(f"    auto {var_name} = createLayer(\"{layer_type}\", {{{props_str}}});")
        cpp_lines.append(f"    status = model->addLayer({var_name});")
        cpp_lines.append(f"    if (status != ML_ERROR_NONE) return status;\n")

    cpp_lines.extend([
        "    // Compile Graph",
        "    status = model->compile();",
        "    if (status != ML_ERROR_NONE) return status;",
        "",
        "    is_compiled = true;",
        "    // initialize Graph",
        "",
        "    status = model->initialize();",
        "    if (status != ML_ERROR_NONE) return status;",
        "",
        "    is_compiled = true;",        
        "    return 0;",
        "}",
        "",
        f"int {class_name}::load_weights(const std::string& bin_file_path) {{",
        "    if (!is_compiled) {",
        "        std::cerr << \"Model must be built before loading weights\" << std::endl;",
        "        return -1;",
        "    }",
        "    // Load ",
        "    // return model->load(bin_file_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);",
        "    return 0; ",
        "}",
        "",
        f"std::vector<float> {class_name}::run(const std::vector<float>& input_ids) {{",
        "    std::vector<float> output;",
        "    // NNTrainer Run (Inference) ",
        "    return output;",
        "}",
        "",
        "} // namespace nntrainer"
    ])
    
    source_code = "\n".join(cpp_lines)
    return header_code, source_code

def save_and_format_files(nntrainer_config, h_code, cpp_code, output_dir="./output"):
    # 1. output dir
    os.makedirs(output_dir, exist_ok=True)
    
    ini_path = os.path.join(output_dir, "qwen2_model.ini")
    h_path = os.path.join(output_dir, "qwen2_model.h")
    cpp_path = os.path.join(output_dir, "qwen2_model.cpp")

    # 2. Save .ini
    with open(ini_path, "w", encoding="utf-8") as f:
        f.write("=== NNTrainer Qwen2.5 Model Configuration ===\n\n")
        for layer in nntrainer_config:
            f.write(f"[{layer['name'].replace('.', '_')}]\n")
            for key, value in layer.items():
                if key != "name":
                    f.write(f"{key} = {value}\n")
            f.write("\n")
    print(f"Saved: {ini_path}")


    # 3. Save .h and .cpp 
    with open(h_path, "w", encoding="utf-8") as f:
        f.write(h_code)
    with open(cpp_path, "w", encoding="utf-8") as f:
        f.write(cpp_code)
    print(f"Saved: {h_path}, {cpp_path}")

    # 4. Run clang-format
    try:
        subprocess.run(["clang-format", "-i", "--style=Google", h_path], check=True)
        subprocess.run(["clang-format", "-i", "--style=Google", cpp_path], check=True)
        print("✨ Successfully formatted C++ files using clang-format!")
    except FileNotFoundError:
        print("⚠️ Warning: 'clang-format' command not found. Files are saved but not formatted.")
        print("   Please install it (e.g., 'apt install clang-format' or 'choco install llvm').")
    except subprocess.CalledProcessError as e:
        print(f" Error during clang-format: {e}")    



# ==========================================
# 4. Execution & INI
# ==========================================
config = AutoConfig.from_pretrained("./qwen2.5-1.5b-local")
model = AutoModelForCausalLM.from_config(config)

tracer = QwenTracer()
graph = tracer.trace(model)
traced_qwen = GraphModule(model, graph)

nntrainer_config = export_to_granular_nntrainer_config(traced_qwen)

# --- Generate  ---
h_code, cpp_code = generate_cpp_class(nntrainer_config, "Qwen2Model")

# Save
save_and_format_files(nntrainer_config, h_code, cpp_code, output_dir="./nntrainer_qwen_build")
