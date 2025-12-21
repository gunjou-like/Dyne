import init, { DyneRuntime } from './pkg/raw_pinn_engine.js';

const PART_WIDTH = 52; 
const OVERLAP = 2;

let runtime1, runtime2;
let state1, state2;
let ctx1, ctx2;
let simulationInterval = null; // シミュレーションループのID

async function main() {
    try {
        console.log("1. Initializing WASM...");
        await init();

        console.log("2. Fetching Weights JSON...");
        // ONNXではなくJSONテキストをロード
        const jsonText = await fetch("../wave_weights.json?" + Date.now()).then(r => {
            if (!r.ok) throw new Error("Failed to load weights json");
            return r.text();
        });

        console.log("3. Creating Runtimes (Level 1)...");
        // 同じ重みを使って2つのインスタンスを作成
        runtime1 = new DyneRuntime(jsonText);
        runtime2 = new DyneRuntime(jsonText);
        console.log("   -> Runtimes created.");

        // 初期状態 (ガウスパルス)
        state1 = new Float32Array(PART_WIDTH).fill(0);
        state2 = new Float32Array(PART_WIDTH).fill(0);

        // 左側の真ん中あたりに山を作る
        for(let i=0; i<PART_WIDTH; i++) {
            state1[i] = Math.exp(-Math.pow(i - 25, 2) / 10);
        }

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
        // Run (入力配列の長さに基づいて計算してくれるので、Ghost Cell含めて渡してOK)
        const next1 = runtime1.run(state1);
        const next2 = runtime2.run(state2);

        state1 = new Float32Array(next1);
        state2 = new Float32Array(next2);

        // 境界同期 (Sync)
        // 左の有効領域の右端 -> 右のGhost(左端)
        state2[0] = state1[PART_WIDTH - 2 - OVERLAP]; 
        state2[1] = state1[PART_WIDTH - 1 - OVERLAP];
        
        // (逆方向の波もあれば逆も必要だが、今は省略)

        draw();
    }, 50);
};

window.stopSimulation = function() {
    if (simulationInterval) {
        clearInterval(simulationInterval);
        simulationInterval = null;
        console.log("⏸️ Simulation Stopped");
    }
};

function draw() {
    visualize(ctx1, state1, "#0ff");
    visualize(ctx2, state2, "#f0f");
}

function visualize(ctx, data, color) {
    if (!data) return;
    const W = ctx.canvas.width;
    const H = ctx.canvas.height;
    ctx.fillStyle = "#222";
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = color;
    const w = W / data.length;
    for(let i=0; i<data.length; i++) {
        // 値が小さいかもしれないので適当に増幅して表示
        const h = data[i] * 50 + H/2; 
        ctx.fillRect(i * w, H - h, w, h);
    }
}

main();