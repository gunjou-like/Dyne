import init, { TranspiledSolver } from './pkg/dyne_solver_cnn_v04.js';

async function main() {
    const status = document.getElementById("status");
    
    // 1. Wasmの初期化
    await init();
    
    // 2. 重みバイナリのダウンロード
    status.innerText = "Fetching weights...";
    const response = await fetch('./model_weights.bin');
    if (!response.ok) {
        status.innerText = "Error loading weights!";
        return;
    }
    const buffer = await response.arrayBuffer();
    
    // 3. ゼロコピー初期化
    // バイナリデータをFloat32Arrayビューとしてラップし、Wasmに渡す
    const weights = new Float32Array(buffer);
    
    status.innerText = `Initializing Solver (Weights: ${weights.byteLength} bytes)...`;
    const solver = new TranspiledSolver(weights);
    
    console.log(solver.get_config());
    status.innerText = "Running Inference...";

    // 4. 推論実行 (前回と同じラプラシアンフィルタのテスト)
    const size = 5;
    const input = new Float32Array(size * size).fill(0.0);
    input[12] = 10.0; // 中心 (Index 12)

    const output = solver.run(input);
    console.log("Output:", output);

    // 5. 結果描画
    drawGrid("canvas-out", output, size);
    status.innerText = "Success! Check console for details.";
}

function drawGrid(canvasId, data, size) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext("2d");
    const imgData = ctx.createImageData(size, size);

    // 簡易可視化 (赤=負, 緑=正)
    for (let i = 0; i < data.length; i++) {
        const val = data[i];
        let r=0, g=0, b=0;
        if (val > 0) { g = Math.min(255, val * 20); }
        else if (val < 0) { r = Math.min(255, Math.abs(val) * 20); }
        
        const idx = i * 4;
        imgData.data[idx] = r;
        imgData.data[idx+1] = g;
        imgData.data[idx+2] = b;
        imgData.data[idx+3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);
}

main();