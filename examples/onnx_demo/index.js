// 生成されたクレート名 (onnx2dyne.pyの --name 引数)
import init, { TranspiledSolver } from './pkg/dyne_solver_generated.js';

async function main() {
    await init();
    
    const solver = new TranspiledSolver();
    console.log(solver.get_config());

    // テスト入力
    const inputData = [1.0, 2.0, 3.0, 4.0, 5.0];
    const outputData = [];

    // ▼▼▼ 修正箇所: ループで1つずつ推論する ▼▼▼
    for (let i = 0; i < inputData.length; i++) {
        // Linear(1, 1) なので、長さ1の配列を渡す
        const singleInput = new Float32Array([inputData[i]]);
        
        // 実行
        const result = solver.run(singleInput);
        
        // 結果を保存 (resultは長さ1の配列なので[0]を取り出す)
        outputData.push(result[0]);
    }

    console.log("Input:", inputData);
    console.log("Output:", outputData);

    const outputDiv = document.getElementById("output");
    outputDiv.innerText = "Output: [" + outputData.join(", ") + "]";

    // 配列の型を合わせるために Float32Array に変換して描画関数へ
    drawChart(new Float32Array(inputData), new Float32Array(outputData));
}

function drawChart(input, output) {
    const canvas = document.getElementById("chart");
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    const padding = 50;

    // 背景
    ctx.fillStyle = "#222";
    ctx.fillRect(0,0,W,H);

    // 軸
    ctx.strokeStyle = "#888";
    ctx.beginPath();
    ctx.moveTo(padding, H-padding);
    ctx.lineTo(W-padding, H-padding); // X軸
    ctx.moveTo(padding, H-padding);
    ctx.lineTo(padding, padding); // Y軸
    ctx.stroke();

    // データプロット
    const maxVal = 10.0; 
    const stepX = (W - padding*2) / (input.length - 1);

    // Input (青)
    ctx.strokeStyle = "#00ccff";
    ctx.beginPath();
    for(let i=0; i<input.length; i++) {
        const x = padding + i * stepX;
        const y = (H-padding) - (input[i] / maxVal) * (H-padding*2);
        if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }
    ctx.stroke();
    ctx.fillStyle = "#00ccff";
    ctx.fillText("Input (x)", W-100, 50);

    // Output (赤)
    ctx.strokeStyle = "#ff4444";
    ctx.beginPath();
    for(let i=0; i<output.length; i++) {
        const x = padding + i * stepX;
        const y = (H-padding) - (output[i] / maxVal) * (H-padding*2);
        if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }
    ctx.stroke();
    ctx.fillStyle = "#ff4444";
    ctx.fillText("Output (2x)", W-100, 70);
}

main();