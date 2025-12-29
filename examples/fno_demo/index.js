import init, { FNOSolver } from './pkg/dyne_solver_fno.js';

async function main() {
    const status = document.getElementById("status");
    
    // 1. 初期化
    await init();
    
    // 2. 重みロード
    status.innerText = "Fetching weights...";
    const response = await fetch('./model_weights.bin');
    const buffer = await response.arrayBuffer();
    const weights = new Float32Array(buffer);
    
    status.innerText = "Initializing FNO Solver (RustFFT)...";
    const solver = new FNOSolver(weights);
    console.log(solver.get_config());

    // 3. 入力データ作成: 矩形波 (Step Function)
    // 前半は -1.0, 後半は 1.0
    const len = 32;
    const input = new Float32Array(len);
    for (let i = 0; i < len; i++) {
        input[i] = (i < len / 2) ? -1.0 : 1.0;
    }

    // 4. 推論実行 (FFT -> Filter -> IFFT)
    const output = solver.run(input);
    
    console.log("Input:", input);
    console.log("Output:", output);

    // 5. 描画
    drawChart(input, output);
    status.innerText = "Success! High frequencies removed.";
}

function drawChart(input, output) {
    const canvas = document.getElementById("chart");
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    const pad = 20;

    // キャンバスをクリア
    ctx.clearRect(0, 0, w, h);
    
    // 背景を少し明るくしてグラフを見やすく
    ctx.fillStyle = "#222";
    ctx.fillRect(0, 0, w, h);

    // X軸線
    ctx.strokeStyle = "#555";
    ctx.beginPath();
    ctx.moveTo(pad, h/2); ctx.lineTo(w-pad, h/2);
    ctx.stroke();

    // プロット関数 (legendIndexを追加して位置をずらす)
    function plot(data, color, label, legendIndex) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        const step = (w - pad*2) / (data.length - 1);
        
        for (let i = 0; i < data.length; i++) {
            const x = pad + i * step;
            // Y軸スケーリング: -1.5 ~ 1.5
            const y = h/2 - (data[i] / 1.5) * (h/2 - pad);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
        
        // 凡例描画
        ctx.fillStyle = color;
        ctx.font = "16px sans-serif"; // フォントサイズ指定
        ctx.textAlign = "left";

        // const legendX = w - 180; // 以前の右上配置
        const legendX = pad + 10;  // 左端から少し離す
        const legendY = 30 + (legendIndex * 25);

        
        // 色付きの四角形
        ctx.fillRect(legendX, legendY - 12, 10, 10);
        // テキスト
        ctx.fillText(label, legendX + 15, legendY);
    }

    plot(input, "#0af", "Input (Square)", 0);      // 青
    plot(output, "#f05", "Output (Filtered)", 1);  // 赤
}

main();