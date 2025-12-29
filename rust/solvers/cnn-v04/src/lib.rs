
use wasm_bindgen::prelude::*;
use dyne_core::{DyneEngine, ModelCategory};

#[wasm_bindgen]
pub struct TranspiledSolver {
    // 重みデータ全体を保持するフラットなベクタ
    weights: Vec<f32>,
}

impl DyneEngine for TranspiledSolver {
    fn step(&mut self, input: &[f32]) -> Vec<f32> {
        let mut x = input.to_vec();
        
        // --- 推論ロジック ---
// Conv Layer 0
        x = conv2d(&x, &self.weights, 
                   0, 9, 0, 0,
                   1, 1, 5, 5, 
                   3, 3, 1, 1);
        
        x
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::PDE
    }

    fn get_boundary(&self) -> Vec<f32> { vec![] }

    fn get_config(&self) -> String {
        "Transpiled ONNX Model (v0.4: Binary Weights)".to_string()
    }
}

#[wasm_bindgen]
impl TranspiledSolver {
    // コンストラクタ: JSからFloat32Arrayを受け取る
    #[wasm_bindgen(constructor)]
    pub fn new(weights_ptr: &[f32]) -> Self {
        Self {
            weights: weights_ptr.to_vec(),
        }
    }
    
    pub fn run(&mut self, input: &[f32]) -> Vec<f32> {
        self.step(input)
    }

    pub fn get_config(&self) -> String {
        <Self as DyneEngine>::get_config(self)
    }
}

// --- Helper Functions ---

fn relu(input: &mut [f32]) {
    for x in input.iter_mut() {
        *x = x.max(0.0);
    }
}

fn conv2d(
    input: &[f32], 
    all_weights: &[f32], // 全重みデータ
    w_start: usize, w_end: usize, // Weightの範囲
    b_start: usize, b_end: usize, // Biasの範囲
    in_c: usize, out_c: usize, 
    h: usize, w: usize,
    k_h: usize, k_w: usize,
    pad: usize, stride: usize
) -> Vec<f32> {
    // スライス切り出し
    let weight = &all_weights[w_start..w_end];
    let bias = if b_end > b_start { &all_weights[b_start..b_end] } else { &[] };

    let out_h = (h + 2 * pad - k_h) / stride + 1;
    let out_w = (w + 2 * pad - k_w) / stride + 1;
    let mut output = vec![0.0; out_c * out_h * out_w];

    for oc in 0..out_c {
        let bias_val = if bias.len() > 0 { bias[oc] } else { 0.0 };
        
        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut sum = bias_val;
                
                let h_start = (oh * stride) as isize - pad as isize;
                let w_start = (ow * stride) as isize - pad as isize;

                for ic in 0..in_c {
                    for kh in 0..k_h {
                        for kw in 0..k_w {
                            let cur_h = h_start + kh as isize;
                            let cur_w = w_start + kw as isize;

                            if cur_h >= 0 && cur_h < h as isize && cur_w >= 0 && cur_w < w as isize {
                                let input_idx = ic * (h * w) + (cur_h as usize) * w + (cur_w as usize);
                                // Weight index within the slice
                                let weight_idx = oc * (in_c * k_h * k_w) + ic * (k_h * k_w) + kh * k_w + kw;
                                sum += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
                
                let out_idx = oc * (out_h * out_w) + oh * out_w + ow;
                output[out_idx] = sum;
            }
        }
    }
    output
}

fn dense(
    input: &[f32], 
    all_weights: &[f32],
    w_start: usize, w_end: usize,
    b_start: usize, b_end: usize,
    out_dim: usize, in_dim: usize
) -> Vec<f32> {
    let weights = &all_weights[w_start..w_end];
    let bias = if b_end > b_start { &all_weights[b_start..b_end] } else { &[] };

    let mut output = vec![0.0; out_dim];
    for i in 0..out_dim {
        let mut sum = 0.0;
        for j in 0..in_dim {
            sum += input[j] * weights[i * in_dim + j];
        }
        output[i] = sum + if bias.len() > 0 { bias[i] } else { 0.0 };
    }
    output
}
