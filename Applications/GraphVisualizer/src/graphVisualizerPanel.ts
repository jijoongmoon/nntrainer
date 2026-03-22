import * as vscode from 'vscode';
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

        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="Content-Security-Policy"
          content="default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NNTrainer Graph Visualizer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: var(--vscode-font-family);
            color: var(--vscode-foreground);
            background: var(--vscode-editor-background);
            overflow: hidden;
        }
        .toolbar {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            background: var(--vscode-titleBar-activeBackground);
            border-bottom: 1px solid var(--vscode-panel-border);
        }
        .toolbar button {
            padding: 4px 12px;
            background: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 12px;
        }
        .toolbar button:hover { background: var(--vscode-button-hoverBackground); }
        .toolbar button.active {
            background: var(--vscode-focusBorder);
        }
        .toolbar .separator {
            width: 1px;
            height: 20px;
            background: var(--vscode-panel-border);
        }
        .toolbar .stats {
            margin-left: auto;
            font-size: 11px;
            color: var(--vscode-descriptionForeground);
        }
        .main-container {
            display: flex;
            height: calc(100vh - 40px);
        }
        .graph-pane {
            flex: 1;
            overflow: auto;
            position: relative;
            border-right: 1px solid var(--vscode-panel-border);
        }
        .graph-pane:last-child { border-right: none; }
        .pane-header {
            position: sticky;
            top: 0;
            z-index: 10;
            padding: 6px 12px;
            font-size: 12px;
            font-weight: 600;
            background: var(--vscode-sideBar-background);
            border-bottom: 1px solid var(--vscode-panel-border);
        }
        .pane-header.torch { color: #e06c75; }
        .pane-header.nntrainer { color: #61afef; }
        .graph-canvas {
            padding: 16px;
            min-height: 100%;
        }
        .node {
            display: inline-flex;
            flex-direction: column;
            margin: 4px 0;
            padding: 6px 10px;
            border: 1px solid var(--vscode-panel-border);
            border-radius: 4px;
            font-size: 11px;
            cursor: pointer;
            transition: all 0.15s;
            background: var(--vscode-editor-background);
        }
        .node:hover {
            border-color: var(--vscode-focusBorder);
            box-shadow: 0 0 4px var(--vscode-focusBorder);
        }
        .node.selected {
            border-color: var(--vscode-focusBorder);
            background: var(--vscode-list-activeSelectionBackground);
        }
        .node.mapped { border-left: 3px solid #98c379; }
        .node.unmapped { border-left: 3px solid #e06c75; }
        .node.skipped { border-left: 3px solid #d19a66; opacity: 0.7; }
        .node .name { font-weight: 600; }
        .node .type { color: var(--vscode-descriptionForeground); font-size: 10px; }
        .node .shape { color: #61afef; font-size: 10px; }

        /* Source code panel */
        .source-pane {
            flex: 1;
            overflow: auto;
            display: none;
            border-right: 1px solid var(--vscode-panel-border);
        }
        .source-pane.visible { display: block; }
        .source-pane .pane-header { position: sticky; top: 0; z-index: 10; padding: 6px 12px; font-size: 12px; font-weight: 600; background: var(--vscode-sideBar-background); border-bottom: 1px solid var(--vscode-panel-border); }
        .source-pane .pane-header.torch-src { color: #e5c07b; }
        .source-pane .pane-header.cpp-src { color: #61afef; }
        .source-code {
            padding: 8px 12px;
            font-family: var(--vscode-editor-fontFamily, 'Consolas, monospace');
            font-size: 12px;
            white-space: pre;
            line-height: 1.5;
            tab-size: 4;
            counter-reset: line;
        }
        .source-code .line {
            display: block;
            padding: 0 8px 0 48px;
            position: relative;
        }
        .source-code .line::before {
            content: counter(line);
            counter-increment: line;
            position: absolute;
            left: 0;
            width: 40px;
            text-align: right;
            color: var(--vscode-editorLineNumber-foreground);
            font-size: 11px;
        }
        .source-code .line:hover { background: var(--vscode-editor-hoverHighlightBackground); }
        .source-code .line.highlighted { background: rgba(255, 213, 79, 0.15); }

        /* Verification panel */
        .verification-panel {
            width: 280px;
            min-width: 280px;
            overflow-y: auto;
            background: var(--vscode-sideBar-background);
            border-left: 1px solid var(--vscode-panel-border);
            padding: 8px;
            font-size: 11px;
            display: none;
        }
        .verification-panel.visible { display: block; }
        .verification-panel h3 {
            font-size: 12px;
            margin: 8px 0 4px;
            color: var(--vscode-sideBarTitle-foreground);
        }
        .badge {
            display: inline-block;
            padding: 1px 6px;
            border-radius: 8px;
            font-size: 10px;
            font-weight: 600;
        }
        .badge.ok { background: #2d4a2d; color: #98c379; }
        .badge.warn { background: #4a3d2d; color: #d19a66; }
        .badge.error { background: #4a2d2d; color: #e06c75; }
        .issue-item {
            padding: 4px 8px;
            margin: 2px 0;
            border-radius: 3px;
            background: var(--vscode-editor-background);
        }
    </style>
</head>
<body>
    <div class="toolbar">
        <button id="btn-side-by-side" class="active">Side by Side</button>
        <button id="btn-nntrainer-only">NNTrainer Only</button>
        <button id="btn-torch-only">Torch FX Only</button>
        <div class="separator"></div>
        <button id="btn-verify">Verification</button>
        <div class="separator"></div>
        <button id="btn-torch-source">PyTorch Source</button>
        <button id="btn-cpp-source">C++ Output</button>
        <div class="stats" id="stats"></div>
    </div>
    <div class="main-container">
        <div class="source-pane" id="torch-source-pane">
            <div class="pane-header torch-src">PyTorch Source Code</div>
            <div class="source-code" id="torch-source-code"></div>
        </div>
        <div class="graph-pane" id="torch-pane">
            <div class="pane-header torch">PyTorch FX Graph</div>
            <div class="graph-canvas" id="torch-canvas"></div>
        </div>
        <div class="graph-pane" id="nntrainer-pane">
            <div class="pane-header nntrainer">NNTrainer Graph</div>
            <div class="graph-canvas" id="nntrainer-canvas"></div>
        </div>
        <div class="source-pane" id="cpp-source-pane">
            <div class="pane-header cpp-src">Generated C++ Code</div>
            <div class="source-code" id="cpp-source-code"></div>
        </div>
        <div class="verification-panel" id="verification-panel"></div>
    </div>

    <script nonce="${nonce}">
        const vscode = acquireVsCodeApi();
        const data = ${data};

        // State
        let selectedNode = null;
        let viewMode = 'side-by-side';
        let verificationVisible = false;
        let torchSourceVisible = false;
        let cppSourceVisible = false;

        // Build node lookup maps
        const fxNodeMap = new Map();
        data.fxGraph.forEach(n => fxNodeMap.set(n.name, n));

        const nnLayerMap = new Map();
        data.nntrainerLayers.forEach(l => nnLayerMap.set(l.name, l));

        const mappingByFx = new Map();
        const mappingByNn = new Map();
        data.nodeMapping.forEach(m => {
            mappingByFx.set(m.fxNodeName, m);
            if (m.nntrainerLayerName) {
                mappingByNn.set(m.nntrainerLayerName, m);
            }
        });

        // Stats
        const mapped = data.nodeMapping.filter(m => m.mappingType === 'direct').length;
        const skipped = data.nodeMapping.filter(m => m.mappingType === 'skipped').length;
        const unmapped = data.unsupportedOps.length + data.unknownLayers.length;
        document.getElementById('stats').textContent =
            'FX: ' + data.fxGraph.length + ' nodes | ' +
            'NNTrainer: ' + data.nntrainerLayers.length + ' layers | ' +
            'Mapped: ' + mapped + ' | Skipped: ' + skipped +
            (unmapped > 0 ? ' | Unmapped: ' + unmapped : '');

        // Render Torch FX graph
        function renderTorchGraph() {
            const canvas = document.getElementById('torch-canvas');
            canvas.innerHTML = '';
            data.fxGraph.forEach(node => {
                if (node.op === 'placeholder' || node.op === 'output') return;
                const mapping = mappingByFx.get(node.name);
                const cls = mapping
                    ? (mapping.mappingType === 'direct' ? 'mapped' :
                       mapping.mappingType === 'skipped' ? 'skipped' : 'unmapped')
                    : 'unmapped';
                const div = document.createElement('div');
                div.className = 'node ' + cls;
                div.dataset.fxName = node.name;
                div.innerHTML =
                    '<span class="name">' + node.name + '</span>' +
                    '<span class="type">' + node.op + ': ' + node.target + '</span>' +
                    (node.output_shape
                        ? '<span class="shape">' + JSON.stringify(node.output_shape) + '</span>'
                        : '') +
                    (node.module_type
                        ? '<span class="type">' + node.module_type + '</span>'
                        : '');
                div.addEventListener('click', () => selectFxNode(node.name));
                canvas.appendChild(div);
            });
        }

        // Render NNTrainer graph
        function renderNNTrainerGraph() {
            const canvas = document.getElementById('nntrainer-canvas');
            canvas.innerHTML = '';
            data.nntrainerLayers.forEach(layer => {
                const mapping = mappingByNn.get(layer.name);
                const cls = mapping ? 'mapped' : '';
                const div = document.createElement('div');
                div.className = 'node ' + cls;
                div.dataset.layerName = layer.name;
                const inputStr = layer.input_layers.length > 0
                    ? ' <- [' + layer.input_layers.join(', ') + ']'
                    : '';
                div.innerHTML =
                    '<span class="name">' + layer.name + '</span>' +
                    '<span class="type">' + layer.layer_type + inputStr + '</span>' +
                    Object.entries(layer.properties).map(([k, v]) =>
                        '<span class="shape">' + k + '=' + v + '</span>'
                    ).join('');
                div.addEventListener('click', () => selectNnLayer(layer.name));
                canvas.appendChild(div);
            });
        }

        function selectFxNode(name) {
            clearSelection();
            const el = document.querySelector('[data-fx-name="' + name + '"]');
            if (el) el.classList.add('selected');

            const mapping = mappingByFx.get(name);
            if (mapping && mapping.nntrainerLayerName) {
                const nnEl = document.querySelector('[data-layer-name="' + mapping.nntrainerLayerName + '"]');
                if (nnEl) {
                    nnEl.classList.add('selected');
                    nnEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            }
            vscode.postMessage({ command: 'nodeSelected', data: { layerName: name, source: 'fx' } });
        }

        function selectNnLayer(name) {
            clearSelection();
            const el = document.querySelector('[data-layer-name="' + name + '"]');
            if (el) el.classList.add('selected');

            const mapping = mappingByNn.get(name);
            if (mapping && mapping.fxNodeName) {
                const fxEl = document.querySelector('[data-fx-name="' + mapping.fxNodeName + '"]');
                if (fxEl) {
                    fxEl.classList.add('selected');
                    fxEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            }
            vscode.postMessage({ command: 'nodeSelected', data: { layerName: name, source: 'nntrainer' } });
        }

        function clearSelection() {
            document.querySelectorAll('.node.selected').forEach(el => el.classList.remove('selected'));
        }

        // View mode buttons
        document.getElementById('btn-side-by-side').addEventListener('click', () => setView('side-by-side'));
        document.getElementById('btn-nntrainer-only').addEventListener('click', () => setView('nntrainer'));
        document.getElementById('btn-torch-only').addEventListener('click', () => setView('torch'));
        document.getElementById('btn-verify').addEventListener('click', () => {
            verificationVisible = !verificationVisible;
            document.getElementById('verification-panel').classList.toggle('visible', verificationVisible);
            document.getElementById('btn-verify').classList.toggle('active', verificationVisible);
        });

        function setView(mode) {
            viewMode = mode;
            document.querySelectorAll('.toolbar button').forEach(b => b.classList.remove('active'));
            const torchPane = document.getElementById('torch-pane');
            const nnPane = document.getElementById('nntrainer-pane');
            switch(mode) {
                case 'side-by-side':
                    torchPane.style.display = 'block';
                    nnPane.style.display = 'block';
                    document.getElementById('btn-side-by-side').classList.add('active');
                    break;
                case 'nntrainer':
                    torchPane.style.display = 'none';
                    nnPane.style.display = 'block';
                    document.getElementById('btn-nntrainer-only').classList.add('active');
                    break;
                case 'torch':
                    torchPane.style.display = 'block';
                    nnPane.style.display = 'none';
                    document.getElementById('btn-torch-only').classList.add('active');
                    break;
            }
        }

        // Render verification panel
        function renderVerification() {
            const panel = document.getElementById('verification-panel');
            let html = '<h3>Conversion Verification</h3>';

            // Summary
            const total = data.fxGraph.filter(n => n.op !== 'placeholder' && n.op !== 'output').length;
            const mappedCount = data.nodeMapping.filter(m => m.mappingType === 'direct').length;
            const ratio = total > 0 ? Math.round(mappedCount / total * 100) : 0;
            const badge = ratio === 100 ? 'ok' : ratio > 80 ? 'warn' : 'error';
            html += '<div class="issue-item">Coverage: <span class="badge ' + badge + '">' + ratio + '%</span> (' + mappedCount + '/' + total + ')</div>';

            // Unsupported ops
            if (data.unsupportedOps.length > 0) {
                html += '<h3>Unsupported Ops <span class="badge error">' + data.unsupportedOps.length + '</span></h3>';
                data.unsupportedOps.forEach(op => {
                    html += '<div class="issue-item">' + op + '</div>';
                });
            }

            // Unknown layers
            if (data.unknownLayers.length > 0) {
                html += '<h3>Unknown Layers <span class="badge warn">' + data.unknownLayers.length + '</span></h3>';
                data.unknownLayers.forEach(l => {
                    html += '<div class="issue-item">' + l + '</div>';
                });
            }

            // Decomposed modules
            if (data.decomposedModules.length > 0) {
                html += '<h3>Decomposed Modules <span class="badge warn">' + data.decomposedModules.length + '</span></h3>';
                data.decomposedModules.forEach(m => {
                    html += '<div class="issue-item">' + m + '</div>';
                });
            }

            // Skipped nodes
            const skippedNodes = data.nodeMapping.filter(m => m.mappingType === 'skipped');
            if (skippedNodes.length > 0) {
                html += '<h3>Skipped FX Nodes <span class="badge warn">' + skippedNodes.length + '</span></h3>';
                skippedNodes.slice(0, 20).forEach(m => {
                    html += '<div class="issue-item">' + m.fxNodeName + '</div>';
                });
                if (skippedNodes.length > 20) {
                    html += '<div class="issue-item">... and ' + (skippedNodes.length - 20) + ' more</div>';
                }
            }

            panel.innerHTML = html;
        }

        // Source code rendering
        function escapeHtml(text) {
            return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        }

        function renderSourceCode(containerId, code) {
            const container = document.getElementById(containerId);
            if (!code) {
                container.innerHTML = '<div style="padding:16px;color:var(--vscode-descriptionForeground)">No source code available</div>';
                return;
            }
            const lines = code.split('\\n');
            container.innerHTML = lines.map((line, i) =>
                '<span class="line" data-line="' + (i+1) + '">' + escapeHtml(line) + '</span>'
            ).join('\\n');
        }

        // Source toggle buttons
        document.getElementById('btn-torch-source').addEventListener('click', () => {
            torchSourceVisible = !torchSourceVisible;
            document.getElementById('torch-source-pane').classList.toggle('visible', torchSourceVisible);
            document.getElementById('btn-torch-source').classList.toggle('active', torchSourceVisible);
        });
        document.getElementById('btn-cpp-source').addEventListener('click', () => {
            cppSourceVisible = !cppSourceVisible;
            document.getElementById('cpp-source-pane').classList.toggle('visible', cppSourceVisible);
            document.getElementById('btn-cpp-source').classList.toggle('active', cppSourceVisible);
        });

        // Init
        renderTorchGraph();
        renderNNTrainerGraph();
        renderVerification();
        renderSourceCode('torch-source-code', data.torchSourceCode || '');
        renderSourceCode('cpp-source-code', data.cppSource || '');
    </script>
</body>
</html>`;
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
