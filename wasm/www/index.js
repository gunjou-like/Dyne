import init, { DyneRuntime } from "../pkg/dyne_runtime.js";

// 設定: グリッドサイズなど
const TOTAL_WIDTH = 100;
const PART_WIDTH = 52; // のりしろ込み
const OVERLAP = 2;

let runtime1, runtime2;
let state1, state2; // 波の状態配列 (Float32Array)
let ctx1, ctx2;

async function main() {
    // 1. WASM初期化
    await init();

    // 2. モデルロード (Splitterで作ったファイルをfetch)
    const model1Bytes = await fetch("../dist/part_0.onnx").then(r => r.arrayBuffer());
    const model2Bytes = await fetch("../dist/part_1.onnx").then(r => r.arrayBuffer());

    console.log("3. Creating Runtimes...");
    
    // 変更点: 第2引数に PART_WIDTH (52) を渡す
    runtime1 = new DyneRuntime(new Uint8Array(model1Bytes), PART_WIDTH);
    runtime2 = new DyneRuntime(new Uint8Array(model2Bytes), PART_WIDTH);
    
    console.log("   -> Runtimes created.");

    // 4. 初期状態の作成 (左端にガウスパルス)
    state1 = new Float32Array(PART_WIDTH).fill(0);
    state2 = new Float32Array(PART_WIDTH).fill(0);
    
    // ガウスパルスを左側の真ん中に置く
    for(let i=0; i<PART_WIDTH; i++) {
        state1[i] = Math.exp(-Math.pow(i - 20, 2) / 10);
    }

    // Canvas準備
    ctx1 = document.getElementById("canvas1").getContext("2d");
    ctx2 = document.getElementById("canvas2").getContext("2d");
    
    draw();
}

// シミュレーションループ
window.startSimulation = async function() {
    setInterval(() => {
        // --- Step 1: 推論 (並列実行のつもり) ---
        const next1 = runtime1.run(state1);
        const next2 = runtime2.run(state2);

        // --- Step 2: 境界同期 (Boundary Sync) ---
        // 左モデルの右端(のりしろ除く) を 右モデルの左端(のりしろ)へ
        // 右モデルの左端(のりしろ除く) を 左モデルの右端(のりしろ)へ
        // ※ 簡易実装として、単純にデータをコピーします
        
        // MVPでは「左から右へ波が流れる」だけでいいので、
        // 左モデルの有効領域の右端 -> 右モデルの入力の左端(のりしろ)
        // Part 1 [0...50...52] -> valid range 0~50
        // Part 2 [0...52] -> ghost cells at 0~2
        
        // 更新
        state1 = next1;
        state2 = next2;
        
        // Sync: Part 1の x=48,49 を Part 2の x=0,1 (Ghost) にコピー
        state2[0] = state1[PART_WIDTH - 2 - OVERLAP]; 
        state2[1] = state1[PART_WIDTH - 1 - OVERLAP];

        draw();
    }, 50); // 20fps
};

function draw() {
    // 可視化ロジック (省略: 単純に棒グラフなどを描く)
    visualize(ctx1, state1, "cyan");
    visualize(ctx2, state2, "orange");
}

function visualize(ctx, data, color) {
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, 400, 200);
    ctx.fillStyle = color;
    const w = 400 / data.length;
    for(let i=0; i<data.length; i++) {
        const h = data[i] * 100;
        ctx.fillRect(i * w, 100 - h, w, h);
    }
}

main();