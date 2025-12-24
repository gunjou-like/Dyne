
use wasm_bindgen::prelude::*;
use dyne_core::{DyneEngine, ModelCategory};

// --- 重みデータの埋め込み ---
const W_0: [f32; 1] = [2.0000];
const B_0: [f32; 1] = [0.0000];

#[wasm_bindgen]
pub struct TranspiledSolver {}

impl DyneEngine for TranspiledSolver {
    fn step(&mut self, input: &[f32]) -> Vec<f32> {
        // 入力次元チェック (簡易)
        // let x = input; 
        
        // --- 推論ロジック (自動生成) ---
        let mut x = input.to_vec();
        x = dense(&x, &W_0, &B_0, 1, 1);
        
        x.to_vec()
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::PDE // 関数近似
    }

    fn get_boundary(&self) -> Vec<f32> { vec![] }

    fn get_config(&self) -> String {
        "Transpiled ONNX Model (MLP)".to_string()
    }
}

#[wasm_bindgen]
impl TranspiledSolver {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {}
    }
    
    pub fn run(&mut self, input: &[f32]) -> Vec<f32> {
        self.step(input)
    }

    // ▼▼▼ 追加箇所: JS向けにプロキシするメソッド ▼▼▼
    pub fn get_config(&self) -> String {
        <Self as DyneEngine>::get_config(self)
    }
}

// --- ヘルパー関数: 行列ベクトル積 (y = Wx + b) ---
fn dense(input: &[f32], weights: &[f32], bias: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut output = vec![0.0; out_dim];
    for i in 0..out_dim {
        let mut sum = 0.0;
        for j in 0..in_dim {
            sum += input[j] * weights[i * in_dim + j];
        }
        output[i] = sum + bias[i];
    }
    output
}

fn relu(input: &mut [f32]) {
    for x in input.iter_mut() {
        *x = x.max(0.0);
    }
}
