import init, { HeatSolver } from './pkg/dyne_solver_heat.js';

// ★重要: dyne.toml の width と同じ値にしてください
const PART_WIDTH = 52; 
const OVERLAP = 2;

let runtime1, runtime2;
let state1, state2;
let ctx1, ctx2;
let simulationInterval = null;

async function main() {
    try {
        console.log("1. Initializing WASM...");
        await init();

        // ▼▼▼ 削除: JSON読み込みブロックは完全に不要です ▼▼▼
        // console.log("2. Fetching Weights JSON...");
        // const jsonText = await fetch(...);
        // ▲▲▲ 削除ここまで ▲▲▲

        console.log("2. Creating Runtimes (Level 1)...");
        // 数値計算ソルバなので引数なしで初期化
        runtime1 = new HeatSolver();
        runtime2 = new HeatSolver();
        console.log("   -> Runtimes created.");

        // 初期状態 (配列確保)
        state1 = new Float32Array(PART_WIDTH).fill(0);
        state2 = new Float32Array(PART_WIDTH).fill(0);

        // 初期条件: 左側の領域(runtime1)に熱源を置く
        // 初期条件: 左側の領域(runtime1)の「右端」に熱源を置く
        for (let i = 0; i < PART_WIDTH; i++) {
            // 右端付近 (40〜50) を 1.0 にする
            // これにより、スタート直後に canvas2 へ熱が流れ込みます
            if (i > PART_WIDTH - 15) {
                state1[i] = 1.0;
            } else {
                state1[i] = 0.0;
            }
        }

        // キャンバス取得 (HTMLに canvas1, canvas2 がある前提)
        ctx1 = document.getElementById("canvas1").getContext("2d");
        ctx2 = document.getElementById("canvas2").getContext("2d");
        
        draw();
        
        console.log("✅ Ready! Click 'Start Simulation' button to begin.");

    } catch (e) {
        console.error("❌ ERROR:", e);
        alert(e);
    }
}

window.startSimulation = function() {
    if (simulationInterval) {
        console.log("⚠️ Simulation already running");
        return;
    }
    
    if (!runtime1 || !runtime2) {
        console.error("❌ Runtimes not initialized yet");
        alert("Please wait for initialization to complete");
        return;
    }
    
    console.log("🚀 Simulation Started");
    simulationInterval = setInterval(() => {
        // Run (1ステップ計算)
        const next1 = runtime1.run(state1);
        const next2 = runtime2.run(state2);

        // 結果をJS側の配列に反映
        state1 = new Float32Array(next1);
        state2 = new Float32Array(next2);

        // --- 境界同期 (簡易実装) ---
        // 左(runtime1) の右端の熱を、右(runtime2) の左端に伝える
        // これにより、熱が canvas1 から canvas2 へ "染み出して" いきます
        state2[0] = state1[PART_WIDTH - 2]; 
        state2[1] = state1[PART_WIDTH - 1];
        
        // (本来は逆方向 state2 -> state1 も必要ですが、まずはこれでOK)
        state1[PART_WIDTH - 1] = state2[1];
        draw();
    }, 50); // 計算速度調整
};

window.stopSimulation = function() {
    if (simulationInterval) {
        clearInterval(simulationInterval);
        simulationInterval = null;
        console.log("⏸️ Simulation Stopped");
    }
};

function draw() {
    visualize(ctx1, state1, "#ff4500"); // オレンジ (熱)
    visualize(ctx2, state2, "#ff4500");
}

function visualize(ctx, data, color) {
    if (!data) return;
    const W = ctx.canvas.width;
    const H = ctx.canvas.height;
    
    // 背景クリア
    ctx.fillStyle = "#222";
    ctx.fillRect(0, 0, W, H);
    
    // グラフ描画
    ctx.fillStyle = color;
    const w = W / data.length;
    for(let i=0; i<data.length; i++) {
        // 値(0.0~1.0) を 高さ(0~H) に変換して表示
        const val = data[i];
        const h = val * (H * 0.8); 
        ctx.fillRect(i * w, H - h, w, h);
    }
}

main();