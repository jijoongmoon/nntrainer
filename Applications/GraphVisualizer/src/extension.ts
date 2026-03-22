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
