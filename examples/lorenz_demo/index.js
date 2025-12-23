// Cargo.toml の name = "dyne-solver-lorenz" なので、ファイル名は以下になります
import init, { LorenzSolver } from './pkg/dyne_solver_lorenz.js';

let solver;
let animationId = null;
let ctx;

// 描画スケール調整（ローレンツアトラクタを画面に収めるため）
const SCALE = 15; 
const OFFSET_X = 400; // キャンバス中央X
const OFFSET_Y = 550; // キャンバス下部Y (Z軸は上向きなので)

async function main() {
    await init();
    solver = new LorenzSolver();
    
    const canvas = document.getElementById("sim-canvas");
    ctx = canvas.getContext("2d");

    // 最初に画面を黒く塗りつぶす
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    console.log("✅ Ready. Press Start.");
}

window.startSimulation = function() {
    if (animationId) return;

    // 前回の点を記憶するための変数
    let prevX = null;
    let prevZ = null;

    function loop() {
        // 1. 計算: 入力は不要なので空配列を渡す
        // 戻り値は [x, y, z] の3要素配列
        const state = solver.run([]); 

        const x = state[0];
        const y = state[1];
        const z = state[2];

        // 2. 座標変換 (物理座標 -> キャンバス座標)
        const drawX = x * SCALE + OFFSET_X;
        const drawZ = -z * SCALE + OFFSET_Y; // Z軸を上に向けるため反転

        // 3. 描画 (線を繋ぐ)
        if (prevX !== null) {
            ctx.beginPath();
            ctx.strokeStyle = `hsl(${z * 2}, 100%, 50%)`; // Zの値で色を変える(虹色)
            ctx.lineWidth = 2;
            ctx.moveTo(prevX, prevZ);
            ctx.lineTo(drawX, drawZ);
            ctx.stroke();
        }

        prevX = drawX;
        prevZ = drawZ;

        // 4. 少しだけ画面を暗くして「残像」を残す (フェード効果)
        // 毎回真っ黒にクリアせず、半透明の黒を重ねることで軌跡を残す
        ctx.fillStyle = "rgba(0, 0, 0, 0.02)";
        ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);

        animationId = requestAnimationFrame(loop);
    }

    loop();
};

main();