// シミュレーション定数
const GRID_SIZE: usize = 128; // グリッド数 (メモリ節約のため控えめに)
const C: f32 = 1.0;           // 波の伝播速度
const DT: f32 = 0.1;          // 時間刻み
const DX: f32 = 0.1;          // 空間刻み

// クーラン数 (安定条件に関わる係数)
// r = (c * dt / dx)^2
const R_SQ: f32 = (C * DT / DX) * (C * DT / DX);

// 静的バッファ (ヒープを使わない！)
// 3つの配列で時間を管理します
static mut U_PREV: [f32; GRID_SIZE] = [0.0; GRID_SIZE]; // n-1
static mut U_CURR: [f32; GRID_SIZE] = [0.0; GRID_SIZE]; // n
static mut U_NEXT: [f32; GRID_SIZE] = [0.0; GRID_SIZE]; // n+1

// 初期化関数: ガウスパルス（波の発生源）を作る
#[no_mangle]
pub extern "C" fn init() -> *mut f32 {
    unsafe {
        // 中央にパルスを配置
        let center = (GRID_SIZE / 2) as f32;
        let width = 5.0;

        for i in 0..GRID_SIZE {
            let x = i as f32;
            // ガウス関数: exp( - (x - center)^2 / width )
            let val = (-((x - center) * (x - center)) / (2.0 * width * width)).exp();
            
            U_PREV[i] = val;
            U_CURR[i] = val; // 初速度0とするため過去=現在にする
        }
        
        U_CURR.as_mut_ptr()
    }
}

// ステップ関数: 1フレーム進める
#[no_mangle]
pub extern "C" fn step() -> *mut f32 {
    unsafe {
        // 1. 物理演算: 波動方程式の離散化
        // u_next[i] = 2*u_curr[i] - u_prev[i] + r^2 * (u_curr[i+1] - 2*u_curr[i] + u_curr[i-1])
        
        // 端（0とMAX）は固定端として計算しない（0.0のまま）
        for i in 1..(GRID_SIZE - 1) {
            let laplacian = U_CURR[i+1] - 2.0 * U_CURR[i] + U_CURR[i-1];
            U_NEXT[i] = 2.0 * U_CURR[i] - U_PREV[i] + R_SQ * laplacian;
            
            // 減衰項（少しずつ波を小さくする: シミュレーション安定化のため）
            U_NEXT[i] *= 0.999; 
        }

        // 2. バッファのローテーション (コピーによる更新)
        // ポインタすり替えの方が速いが、static mutではコピーの方が安全で確実
        for i in 0..GRID_SIZE {
            U_PREV[i] = U_CURR[i];
            U_CURR[i] = U_NEXT[i];
        }

        // 現在の状態（描画用データ）のポインタを返す
        U_CURR.as_mut_ptr()
    }
}

// デバッグ用: 配列サイズを返す
#[no_mangle]
pub extern "C" fn get_size() -> usize {
    GRID_SIZE
}