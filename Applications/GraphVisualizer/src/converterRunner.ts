import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { spawn } from 'child_process';
import { ConversionResult } from './types';

export class ConverterRunner {
    private context: vscode.ExtensionContext;

    constructor(context: vscode.ExtensionContext) {
        this.context = context;
    }

    /** Detect TorchFXConverter path */
    private getConverterPath(): string {
        const config = vscode.workspace.getConfiguration('nntrainerGraph');
        const configured = config.get<string>('converterPath');
        if (configured) {
            return configured;
        }

        // Auto-detect: look for TorchFXConverter relative to workspace
        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (workspaceFolders) {
            for (const folder of workspaceFolders) {
                const candidate = path.join(
                    folder.uri.fsPath,
                    'Applications',
                    'TorchFXConverter'
                );
                if (fs.existsSync(path.join(candidate, 'converter.py'))) {
                    return candidate;
                }
            }
        }

        // Fallback: relative to extension
        const extDir = this.context.extensionPath;
        const relative = path.join(extDir, '..', 'TorchFXConverter');
        if (fs.existsSync(path.join(relative, 'converter.py'))) {
            return relative;
        }

        throw new Error(
            'TorchFXConverter not found. Set nntrainerGraph.converterPath in settings.'
        );
    }

    /** Get Python interpreter path */
    private getPythonPath(): string {
        const config = vscode.workspace.getConfiguration('nntrainerGraph');
        return config.get<string>('pythonPath') || 'python3';
    }

    /** Run TorchFXConverter and return structured result */
    async runConversion(modelId: string): Promise<ConversionResult | null> {
        const converterPath = this.getConverterPath();
        const pythonPath = this.getPythonPath();
        const seqLen = vscode.workspace
            .getConfiguration('nntrainerGraph')
            .get<number>('defaultSeqLen') || 8;

        // Use temp directory for output
        const tmpDir = path.join(this.context.globalStorageUri.fsPath, 'conversions', Date.now().toString());
        fs.mkdirSync(tmpDir, { recursive: true });

        // We use a wrapper script that outputs the full conversion result as JSON
        // including the fx graph serialization
        const wrapperScript = path.join(converterPath, 'vscode_bridge.py');
        if (!fs.existsSync(wrapperScript)) {
            // Generate the bridge script if it doesn't exist yet
            await this.createBridgeScript(converterPath);
        }

        return vscode.window.withProgress(
            {
                location: vscode.ProgressLocation.Notification,
                title: 'Converting model...',
                cancellable: true,
            },
            async (progress, token) => {
                return new Promise<ConversionResult | null>((resolve) => {
                    const args = [
                        wrapperScript,
                        '--model', modelId,
                        '--output', tmpDir,
                        '--seq-len', seqLen.toString(),
                        '--format', 'all',
                    ];

                    progress.report({ message: `Running: ${pythonPath} ${args.join(' ')}` });

                    const proc = spawn(pythonPath, args, {
                        cwd: converterPath,
                        env: { ...process.env, PYTHONPATH: converterPath },
                    });

                    let stdout = '';
                    let stderr = '';

                    proc.stdout.on('data', (data: Buffer) => {
                        stdout += data.toString();
                        const lines = data.toString().split('\n');
                        for (const line of lines) {
                            if (line.startsWith('PROGRESS:')) {
                                progress.report({ message: line.substring(9).trim() });
                            }
                        }
                    });

                    proc.stderr.on('data', (data: Buffer) => {
                        stderr += data.toString();
                    });

                    token.onCancellationRequested(() => {
                        proc.kill('SIGTERM');
                        resolve(null);
                    });

                    proc.on('close', (code) => {
                        if (code !== 0) {
                            vscode.window.showErrorMessage(
                                `Conversion failed (exit ${code}): ${stderr.slice(-500)}`
                            );
                            resolve(null);
                            return;
                        }

                        // Read the result JSON
                        const resultPath = path.join(tmpDir, 'conversion_result.json');
                        if (!fs.existsSync(resultPath)) {
                            vscode.window.showErrorMessage(
                                'Conversion produced no result file'
                            );
                            resolve(null);
                            return;
                        }

                        try {
                            const raw = fs.readFileSync(resultPath, 'utf-8');
                            const result = JSON.parse(raw) as ConversionResult;
                            resolve(result);
                        } catch (e) {
                            vscode.window.showErrorMessage(
                                `Failed to parse conversion result: ${e}`
                            );
                            resolve(null);
                        }
                    });
                });
            }
        );
    }

    /** Load a previously saved conversion result */
    async loadFromJson(filePath: string): Promise<ConversionResult | null> {
        try {
            const raw = fs.readFileSync(filePath, 'utf-8');
            return JSON.parse(raw) as ConversionResult;
        } catch (e) {
            vscode.window.showErrorMessage(`Failed to load: ${e}`);
            return null;
        }
    }

    /** Create the Python bridge script that serializes full ConversionResult */
    private async createBridgeScript(converterPath: string): Promise<void> {
        const script = `#!/usr/bin/env python3
"""VS Code bridge for TorchFXConverter.
Runs conversion and outputs full result as JSON for the visualizer."""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from converter import load_model, prepare_input_kwargs, get_config
from decomposer import AdaptiveConverter
from emitter_json import JsonEmitter
from emitter_ini import emit_ini
from emitter_cpp import emit_cpp_source, emit_cpp_header
from nntrainer_layers import NNTrainerLayerDef


def serialize_fx_graph(graph):
    """Serialize torch.fx Graph to list of node dicts."""
    nodes = []
    for node in graph.nodes:
        entry = {
            "name": node.name,
            "op": node.op,
            "target": str(node.target),
            "args": [str(a) for a in node.args if hasattr(a, 'name')],
            "output_shape": node.meta.get("output_shape"),
            "module_type": node.meta.get("module_type"),
            "scope": node.meta.get("scope", ""),
            "meta": {}
        }
        # Serialize safe meta keys
        safe_keys = [
            "leaf_module", "has_weight", "has_bias",
            "in_features", "out_features", "num_embeddings", "embedding_dim",
            "normalized_shape", "eps", "hidden_size",
            "is_rmsnorm", "is_rnn_module"
        ]
        for key in safe_keys:
            if key in node.meta:
                val = node.meta[key]
                try:
                    json.dumps(val)
                    entry["meta"][key] = val
                except (TypeError, ValueError):
                    entry["meta"][key] = str(val)
        nodes.append(entry)
    return nodes


def serialize_layer(layer):
    """Serialize NNTrainerLayerDef to dict."""
    props = {}
    for k, v in layer.properties.items():
        try:
            json.dumps(v)
            props[k] = v
        except (TypeError, ValueError):
            props[k] = str(v)

    return {
        "name": layer.name,
        "layer_type": layer.layer_type,
        "properties": props,
        "input_layers": list(layer.input_layers),
        "fx_node_name": layer.fx_node_name,
        "hf_module_name": layer.hf_module_name,
        "hf_module_type": layer.hf_module_type,
        "has_weight": layer.has_weight,
        "has_bias": layer.has_bias,
        "weight_hf_key": layer.weight_hf_key,
        "bias_hf_key": layer.bias_hf_key,
        "transpose_weight": layer.transpose_weight,
        "shared_from": layer.shared_from,
    }


def build_node_mapping(layers, fx_graph):
    """Build mapping between fx nodes and nntrainer layers."""
    mappings = []
    fx_node_names = {node.name for node in fx_graph.nodes}
    mapped_fx = set()

    for layer in layers:
        if layer.fx_node_name and layer.fx_node_name in fx_node_names:
            mappings.append({
                "fxNodeName": layer.fx_node_name,
                "nntrainerLayerName": layer.name,
                "hfModuleName": layer.hf_module_name,
                "mappingType": "direct"
            })
            mapped_fx.add(layer.fx_node_name)

    # Find unmapped fx nodes (skipped or no-op nodes)
    for node in fx_graph.nodes:
        if node.name not in mapped_fx and node.op not in ("placeholder", "output"):
            mappings.append({
                "fxNodeName": node.name,
                "nntrainerLayerName": "",
                "hfModuleName": "",
                "mappingType": "skipped"
            })

    return mappings


def serialize_structure(structure):
    """Serialize ModelStructure to dict."""
    if structure is None:
        return None

    blocks = []
    for b in (structure.blocks or []):
        block = {
            "block_idx": b.block_idx,
            "pre_attn_norm": b.pre_attn_norm,
            "attn_residual": b.attn_residual,
            "pre_ffn_norm": b.pre_ffn_norm,
            "ffn_residual": b.ffn_residual,
            "norm_type": b.norm_type,
            "attention": None,
            "ffn": None,
        }
        if b.attention:
            a = b.attention
            block["attention"] = {
                "block_idx": a.block_idx,
                "q_proj": a.q_proj,
                "k_proj": a.k_proj,
                "v_proj": a.v_proj,
                "o_proj": a.o_proj,
                "attention_type": a.attention_type,
                "has_rope": a.has_rope,
                "num_heads": getattr(a, "num_heads", 0),
                "num_kv_heads": getattr(a, "num_kv_heads", 0),
                "head_dim": getattr(a, "head_dim", 0),
            }
        if b.ffn:
            f = b.ffn
            block["ffn"] = {
                "block_idx": f.block_idx,
                "ffn_type": f.ffn_type,
                "gate_proj": getattr(f, "gate_proj", ""),
                "up_proj": getattr(f, "up_proj", ""),
                "down_proj": getattr(f, "down_proj", ""),
                "intermediate_size": getattr(f, "intermediate_size", 0),
            }
        blocks.append(block)

    return {
        "model_type": structure.model_type,
        "arch_type": structure.arch_type,
        "vocab_size": structure.vocab_size,
        "hidden_size": structure.hidden_size,
        "num_layers": structure.num_layers,
        "num_heads": structure.num_heads,
        "num_kv_heads": structure.num_kv_heads,
        "head_dim": structure.head_dim,
        "intermediate_size": structure.intermediate_size,
        "norm_eps": structure.norm_eps,
        "tie_word_embeddings": structure.tie_word_embeddings,
        "rope_theta": structure.rope_theta,
        "embedding": structure.embedding,
        "final_norm": structure.final_norm,
        "lm_head": structure.lm_head,
        "blocks": blocks,
    }


def main():
    parser = argparse.ArgumentParser(description="VS Code bridge for TorchFXConverter")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--seq-len", type=int, default=8, help="Sequence length")
    parser.add_argument("--format", nargs="+", default=["all"], help="Output formats")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--model-name", default=None, help="Override model name")
    parser.add_argument("--plugin-config", default=None, help="Plugin config path")
    args = parser.parse_args()

    print("PROGRESS: Loading model...", flush=True)
    model, hf_config = load_model(args.model)

    print("PROGRESS: Preparing inputs...", flush=True)
    config = get_config(hf_config, seq_len=args.seq_len)
    input_kwargs = prepare_input_kwargs(model, config)

    print("PROGRESS: Running adaptive conversion...", flush=True)
    converter = AdaptiveConverter(model, config)
    result = converter.convert(input_kwargs)

    layers = result.layers
    structure = result.model_structure

    # Determine model name
    model_name = args.model_name or args.model.replace("/", "_").replace("-", "_")

    print("PROGRESS: Generating outputs...", flush=True)

    # Generate C++ and INI
    cpp_source = ""
    ini_config = ""
    try:
        cpp_source = emit_cpp_source(layers, structure, model_name=model_name)
    except Exception as e:
        print(f"Warning: C++ emission failed: {e}", file=sys.stderr)
    try:
        ini_config = emit_ini(layers, structure, batch_size=args.batch_size)
    except Exception as e:
        print(f"Warning: INI emission failed: {e}", file=sys.stderr)

    # Build weight map
    weight_map = []
    for layer in layers:
        if layer.has_weight and layer.weight_hf_key:
            weight_map.append({
                "layer_name": layer.name,
                "layer_type": layer.layer_type,
                "weight_key": layer.weight_hf_key,
                "transpose_weight": layer.transpose_weight,
            })

    print("PROGRESS: Serializing conversion result...", flush=True)

    output = {
        "nntrainerLayers": [serialize_layer(l) for l in layers],
        "fxGraph": serialize_fx_graph(result.graph),
        "modelStructure": serialize_structure(structure),
        "weightMap": weight_map,
        "nodeMapping": build_node_mapping(layers, result.graph),
        "unsupportedOps": [l.name for l in result.unsupported_ops],
        "unknownLayers": [l.name for l in result.unknown_layers],
        "decomposedModules": list(result.decomposed_module_types),
        "cppSource": cpp_source,
        "iniConfig": ini_config,
    }

    os.makedirs(args.output, exist_ok=True)
    result_path = os.path.join(args.output, "conversion_result.json")
    with open(result_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"PROGRESS: Done. Result saved to {result_path}", flush=True)


if __name__ == "__main__":
    main()
`;
        const scriptPath = path.join(converterPath, 'vscode_bridge.py');
        fs.writeFileSync(scriptPath, script, 'utf-8');
    }
}
