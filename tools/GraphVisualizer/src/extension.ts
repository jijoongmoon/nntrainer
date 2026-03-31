import * as vscode from 'vscode';
import { ConverterRunner } from './converterRunner';
import { GraphVisualizerPanel } from './graphVisualizerPanel';
import { ModelExplorerProvider } from './modelExplorerProvider';
import { NodePropertiesProvider } from './nodePropertiesProvider';

let converterRunner: ConverterRunner;
let modelExplorer: ModelExplorerProvider;
let nodeProperties: NodePropertiesProvider;

export function activate(context: vscode.ExtensionContext) {
    converterRunner = new ConverterRunner(context);
    modelExplorer = new ModelExplorerProvider();
    nodeProperties = new NodePropertiesProvider();

    // Tree views
    vscode.window.registerTreeDataProvider('nntrainerGraph.modelExplorer', modelExplorer);
    vscode.window.registerTreeDataProvider('nntrainerGraph.nodeProperties', nodeProperties);

    // Command: Convert Model
    context.subscriptions.push(
        vscode.commands.registerCommand('nntrainerGraph.convert', async () => {
            const modelId = await vscode.window.showInputBox({
                prompt: 'Enter HuggingFace model ID or local model path',
                placeHolder: 'e.g., Qwen/Qwen3-0.6B, /path/to/model',
            });
            if (!modelId) {
                return;
            }

            const result = await converterRunner.runConversion(modelId);
            if (result) {
                modelExplorer.setConversionResult(result);
                GraphVisualizerPanel.createOrShow(context.extensionUri, result);
                vscode.window.showInformationMessage(
                    `Conversion complete: ${result.nntrainerLayers.length} layers`
                );
            }
        })
    );

    // Command: Convert Local PyTorch Model (.py file)
    context.subscriptions.push(
        vscode.commands.registerCommand('nntrainerGraph.convertLocalModel', async () => {
            const fileUri = await vscode.window.showOpenDialog({
                canSelectFiles: true,
                filters: { 'Python Files': ['py'] },
                title: 'Select PyTorch model file (.py)',
            });
            if (!fileUri || fileUri.length === 0) {
                return;
            }

            const pyFilePath = fileUri[0].fsPath;

            // Ask for the model class name within the file
            const className = await vscode.window.showInputBox({
                prompt: 'Enter the nn.Module class name to convert',
                placeHolder: 'e.g., MyModel, TransformerBlock',
            });
            if (!className) {
                return;
            }

            // Ask for input shape info
            const inputDesc = await vscode.window.showInputBox({
                prompt: 'Describe model inputs as JSON (or leave empty for auto-detect)',
                placeHolder: '{"input_ids": [1, 8], "attention_mask": [1, 8]}',
                value: '',
            });

            const result = await converterRunner.runLocalConversion(
                pyFilePath, className, inputDesc || undefined
            );
            if (result) {
                modelExplorer.setConversionResult(result);
                GraphVisualizerPanel.createOrShow(context.extensionUri, result);

                // Also open the torch source code in a side editor
                const doc = await vscode.workspace.openTextDocument(pyFilePath);
                await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside, true);

                vscode.window.showInformationMessage(
                    `Local model converted: ${result.nntrainerLayers.length} layers from ${className}`
                );
            }
        })
    );

    // Command: Profile Model
    context.subscriptions.push(
        vscode.commands.registerCommand('nntrainerGraph.profile', async () => {
            const modelId = await vscode.window.showInputBox({
                prompt: 'Enter model to profile (HuggingFace ID or local path)',
                placeHolder: 'e.g., Qwen/Qwen3-0.6B, ./test_model',
            });
            if (!modelId) {
                return;
            }

            const profileData = await converterRunner.runProfile(modelId);
            if (profileData) {
                // Send profile data to the existing visualizer panel
                GraphVisualizerPanel.sendProfileData(profileData);
                vscode.window.showInformationMessage(
                    `Profiling complete: ${profileData.layers.length} layers, ` +
                    `${profileData.total_time_ms.toFixed(2)} ms total`
                );
            }
        })
    );

    // Command: Node selected (from webview)
    context.subscriptions.push(
        vscode.commands.registerCommand('nntrainerGraph.nodeSelected', (data: { layerName: string; source: string }) => {
            if (!data) { return; }
            const result = modelExplorer.getConversionResult();
            if (!result) { return; }

            // Find the layer/node info and show in properties panel
            const props: Array<{ key: string; value: string }> = [];
            if (data.source === 'nntrainer') {
                const layer = result.nntrainerLayers.find(l => l.name === data.layerName);
                if (layer) {
                    props.push({ key: 'Name', value: layer.name });
                    props.push({ key: 'Type', value: layer.layer_type });
                    props.push({ key: 'Inputs', value: (layer.input_layers || []).join(', ') || '(none)' });
                    if (layer.hf_module_name) { props.push({ key: 'HF Module', value: layer.hf_module_name }); }
                    if (layer.hf_module_type) { props.push({ key: 'HF Type', value: layer.hf_module_type }); }
                    for (const [k, v] of Object.entries(layer.properties || {})) {
                        props.push({ key: k, value: String(v) });
                    }
                }
            } else {
                const node = result.fxGraph.find(n => n.name === data.layerName);
                if (node) {
                    props.push({ key: 'Name', value: node.name });
                    props.push({ key: 'Op', value: node.op });
                    props.push({ key: 'Target', value: node.target });
                    props.push({ key: 'Args', value: (node.args || []).join(', ') || '(none)' });
                    if (node.module_type) { props.push({ key: 'Module Type', value: node.module_type }); }
                    if (node.output_shape) { props.push({ key: 'Output Shape', value: JSON.stringify(node.output_shape) }); }
                }
            }
            nodeProperties.setProperties(props);
        })
    );

    // Command: Open Visualizer (from existing JSON)
    context.subscriptions.push(
        vscode.commands.registerCommand('nntrainerGraph.openVisualizer', async () => {
            const fileUri = await vscode.window.showOpenDialog({
                canSelectFiles: true,
                filters: { 'Conversion Result': ['json'] },
                title: 'Select conversion result JSON',
            });
            if (!fileUri || fileUri.length === 0) {
                return;
            }

            const result = await converterRunner.loadFromJson(fileUri[0].fsPath);
            if (result) {
                modelExplorer.setConversionResult(result);
                GraphVisualizerPanel.createOrShow(context.extensionUri, result);
            }
        })
    );
}

export function deactivate() {}
