import init, { TranspiledSolver } from './pkg/dyne_solver_cnn_quantized.js';

async function main() {
    const status = document.getElementById("status");
    
    await init();
    
    // 1. 重みダウンロード (18 bytes)
    status.innerText = "Fetching quantized weights (f16)...";
    const response = await fetch('./model_weights.bin');
    const buffer = await response.arrayBuffer();
    
    // 2. バイト列としてWasmに渡す
    // 以前: new Float32Array(buffer)
    // 今回: new Uint8Array(buffer) -> Wasm内で f16->f32 展開される
    const weightsBytes = new Uint8Array(buffer);
    
    status.innerText = `Initializing (Binary Size: ${weightsBytes.byteLength} bytes)...`;
    const solver = new TranspiledSolver(weightsBytes);
    
    console.log(solver.get_config());

    // 3. 推論実行 (以前と同じ入力)
    const size = 5;
    const input = new Float32Array(size * size).fill(0.0);
    input[12] = 10.0; // 中心

    const output = solver.run(input);
    console.log("Output:", output);

    // 4. 描画
    drawGrid("canvas-out", output, size);
    status.innerText = "Success! Dequantization verified.";
}

function drawGrid(canvasId, data, size) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext("2d");
    const imgData = ctx.createImageData(size, size);

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