import init, { SplitWaveSolver, PartitionConfig } from './pkg/dyne_solver_split_wave.js';

async function main() {
    await init();
    
    // 領域サイズ設定
    const N = 50; // 各ノードのグリッド数
    
    // --- 1. インスタンス生成 ---
    // 左側担当 (Global 0~50)
    const solverLeft = new SplitWaveSolver(N);
    // 右側担当 (Global 50~100)
    const solverRight = new SplitWaveSolver(N);
    
    // パーティション設定 (Rust側に自分の位置を教える)
    // PartitionConfig(global_offset, local_size, global_size)
    solverLeft.set_partition(new PartitionConfig(0, N, N*2));
    solverRight.set_partition(new PartitionConfig(N, N, N*2));

    // --- 2. 初期条件設定 (波を起こす) ---
    // 左側の真ん中あたりにガウス波束を置く
    const initialLeft = new Float32Array(N);
    for(let i=0; i<N; i++) {
        // xは 0.0 ~ 0.5 の範囲
        const x = i / (N * 2.0); 
        // x=0.25 (左側の中央) にピーク
        initialLeft[i] = Math.exp(-1000 * (x - 0.25)**2);
    }
    solverLeft.set_state(initialLeft);

    // --- 3. メインループ (Sync & Step) ---
    const ctxLeft = document.getElementById("canvas-left").getContext("2d");
    const ctxRight = document.getElementById("canvas-right").getContext("2d");
    
    function loop() {
        // --- A. Boundary Sync (境界同期) ---
        // 1. 各自の端の値を取得 (Get Ghost)
        const leftOut = solverLeft.get_right_out(); // 左ノードの右端
        const rightOut = solverRight.get_left_out(); // 右ノードの左端
        
        // 2. 相手に渡す (Set Ghost)
        // 左ノードの右隣は、右ノードの左端 (rightOut) ではなく「自分の右にあるセル」
        // つまり「右ノードの左端(rightOut)が、左ノードにとっての右側の境界値」になる
        solverLeft.set_right_in(rightOut); 
        solverRight.set_left_in(leftOut);
        
        // ※左ノードの左端と、右ノードの右端は固定壁(0)のまま

        // --- B. Step (計算) ---
        const uLeft = solverLeft.run();
        const uRight = solverRight.run();
        
        // --- C. Draw ---
        drawWave(ctxLeft, uLeft, "Left");
        drawWave(ctxRight, uRight, "Right");
        
        requestAnimationFrame(loop);
    }
    
    loop();
    document.getElementById("status").innerText = "Running: Wave moves from Left to Right";
}

function drawWave(ctx, data, label) {
    const w = ctx.canvas.width;
    const h = ctx.canvas.height;
    ctx.fillStyle = "#222";
    ctx.fillRect(0, 0, w, h);
    
    ctx.strokeStyle = label === "Left" ? "#0f0" : "#f0f"; // 左=緑, 右=ピンク
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    const step = w / data.length;
    for(let i=0; i<data.length; i++) {
        const x = i * step;
        const y = h/2 - data[i] * 50; // 振幅スケール
        if(i===0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();
}

main();