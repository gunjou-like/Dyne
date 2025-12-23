use wasm_bindgen::prelude::*;
use dyne_core::DyneEngine;

mod constants;
use constants::*;

#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

// 数値計算ソルバ本体
// AIモデルではないので、重みデータは持ちません
#[wasm_bindgen]
pub struct HeatSolver {
    alpha: f32, // 熱拡散率
}

impl DyneEngine for HeatSolver {
    fn step(&mut self, input: &[f32]) -> Vec<f32> {
        let len = input.len();
        
        // バリデーション
        if len != WIDTH {
             web_sys::console::error_1(&format!(
                "Dyne Error: Input width ({}) matches config ({})",
                len, WIDTH
            ).into());
            panic!("Dimension mismatch!");
        }

        let mut next_grid = input.to_vec();
        
        // 1次元 差分法 (Finite Difference Method)
        // 両端(0 と len-1)は固定端境界条件として更新しない
        for i in 1..len - 1 {
            let u = input[i];
            let u_left = input[i - 1];
            let u_right = input[i + 1];

            // 拡散項 (Laplacian)
            let laplacian = u_right - 2.0 * u + u_left;
            
            // 時間発展
            // alpha * dt はパラメータとして注入しても良いが、ここでは簡易化のため固定または定数利用
            // 安定条件のため係数は小さめに
            let diffusion_rate = self.alpha * DT; 
            
            next_grid[i] = u + diffusion_rate * laplacian;
        }

        next_grid
    }

    // ▼▼▼ 追加 ▼▼▼
    fn category(&self) -> dyne_core::ModelCategory {
        dyne_core::ModelCategory::PDE
    }

    fn get_boundary(&self) -> Vec<f32> {
        // 今は簡易的に空でも良いが、本来は両端の値を返す
        vec![] 
    }
    // ▲▲▲ 追加ここまで ▲▲▲

    fn get_config(&self) -> String {
        format!("Heat Solver (FDM), alpha={}, width={}", self.alpha, WIDTH)
    }
}

// WASM公開用
#[wasm_bindgen]
impl HeatSolver {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            alpha: 1.0, // 拡散係数 (必要ならdyne.tomlから注入も可能)
        }
    }

    pub fn run(&mut self, input: &[f32]) -> Vec<f32> {
        self.step(input)
    }
}