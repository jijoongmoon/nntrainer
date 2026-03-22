import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { ConversionResult } from './types';

export class GraphVisualizerPanel {
    public static currentPanel: GraphVisualizerPanel | undefined;
    private static readonly viewType = 'nntrainerGraphVisualizer';

    private readonly panel: vscode.WebviewPanel;
    private readonly extensionUri: vscode.Uri;
    private disposables: vscode.Disposable[] = [];

    public static createOrShow(extensionUri: vscode.Uri, result: ConversionResult) {
        const column = vscode.window.activeTextEditor
            ? vscode.window.activeTextEditor.viewColumn
            : undefined;

        if (GraphVisualizerPanel.currentPanel) {
            GraphVisualizerPanel.currentPanel.panel.reveal(column);
            GraphVisualizerPanel.currentPanel.update(result);
            return;
        }

        const panel = vscode.window.createWebviewPanel(
            GraphVisualizerPanel.viewType,
            'NNTrainer Graph Visualizer',
            column || vscode.ViewColumn.One,
            {
                enableScripts: true,
                retainContextWhenHidden: true,
                localResourceRoots: [
                    vscode.Uri.joinPath(extensionUri, 'webview'),
                    vscode.Uri.joinPath(extensionUri, 'resources'),
                ],
            }
        );

        GraphVisualizerPanel.currentPanel = new GraphVisualizerPanel(panel, extensionUri);
        GraphVisualizerPanel.currentPanel.update(result);
    }

    private constructor(panel: vscode.WebviewPanel, extensionUri: vscode.Uri) {
        this.panel = panel;
        this.extensionUri = extensionUri;

        this.panel.onDidDispose(() => this.dispose(), null, this.disposables);

        this.panel.webview.onDidReceiveMessage(
            (message) => this.handleMessage(message),
            null,
            this.disposables
        );
    }

    public update(result: ConversionResult) {
        this.panel.webview.html = this.getHtml(result);
    }

    private handleMessage(message: { command: string; data?: unknown }) {
        switch (message.command) {
            case 'nodeSelected': {
                const data = message.data as { layerName: string; source: string };
                vscode.commands.executeCommand(
                    'nntrainerGraph.nodeSelected',
                    data
                );
                break;
            }
            case 'openCppSource': {
                const data = message.data as { lineHint: string };
                // Future: open generated C++ and jump to line
                vscode.window.showInformationMessage(
                    `C++ source: ${data.lineHint}`
                );
                break;
            }
        }
    }

    private dispose() {
        GraphVisualizerPanel.currentPanel = undefined;
        this.panel.dispose();
        while (this.disposables.length) {
            const d = this.disposables.pop();
            if (d) {
                d.dispose();
            }
        }
    }

    private getHtml(result: ConversionResult): string {
        const nonce = getNonce();
        const data = JSON.stringify(result);

        // Load HTML template from external file
        const htmlPath = path.join(
            path.dirname(path.dirname(__filename)),
            'webview', 'graphView.html'
        );
        let template = fs.readFileSync(htmlPath, 'utf-8');
        template = template.replace(/{{NONCE}}/g, nonce);
        template = template.replace('{{DATA}}', data);
        return template;
    }
}

function getNonce(): string {
    let text = '';
    const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    for (let i = 0; i < 32; i++) {
        text += possible.charAt(Math.floor(Math.random() * possible.length));
    }
    return text;
}
