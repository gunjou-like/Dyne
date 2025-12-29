// 生成されたパッケージ (Cargo.tomlのname指定に合わせる)
import init, { TranspiledSolver } from './pkg/dyne_solver_cnn.js';

async function main() {
    await init();
    const solver = new TranspiledSolver();
    console.log(solver.get_config());

    // --- 1. 入力データの作成 (5x5) ---
    // 中心(2,2)だけ値を大きくする（白い点）
    const size = 5;
    const input = new Float32Array(size * size).fill(0.0);
    input[2 * size + 2] = 10.0; // 中心に値10.0を設定

    // --- 2. 実行 (Conv2d) ---
    const output = solver.run(input);

    console.log("Input:", input);
    console.log("Output:", output);

    // --- 3. 描画 ---
    drawGrid("canvas-in", input, size);
    drawGrid("canvas-out", output, size);
}

function drawGrid(canvasId, data, size) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext("2d");
    const imgData = ctx.createImageData(size, size);

    // データの最大・最小を取得して正規化（可視化のため）
    let min = Infinity, max = -Infinity;
    for(let v of data) {
        if(v < min) min = v;
        if(v > max) max = v;
    }
    // ゼロ除算回避
    if (max === min) max = min + 1;

    for (let i = 0; i < data.length; i++) {
        const val = data[i];
        
        // 値に応じて色付け (グレースケール)
        // 負の値（エッジの周囲）も見えるように、絶対値やオフセットで調整しても良いが
        // シンプルに 0~255 にマッピングする
        // ラプラシアンフィルタの場合、中心が正、周囲が負になることが多い
        
        // 簡易可視化: 負の値は赤、正の値は白/緑、0は黒
        let r=0, g=0, b=0;
        if (val > 0) {
            const intensity = Math.min(255, (val / max) * 255);
            g = intensity; b = intensity; r = intensity; // 白っぽい
        } else if (val < 0) {
            const intensity = Math.min(255, (Math.abs(val) / Math.abs(min)) * 255);
            r = intensity; // 赤っぽい
        }

        const pixelIndex = i * 4;
        imgData.data[pixelIndex + 0] = r;     // R
        imgData.data[pixelIndex + 1] = g;     // G
        imgData.data[pixelIndex + 2] = b;     // B
        imgData.data[pixelIndex + 3] = 255;   // Alpha
    }

    ctx.putImageData(imgData, 0, 0);
}

main();