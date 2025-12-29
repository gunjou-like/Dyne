import onnx
import numpy as np
import os
import argparse
from onnx import numpy_helper

# --- FNO Template (Rust) ---
RUST_FNO_TEMPLATE = """
use wasm_bindgen::prelude::*;
use dyne_core::{DyneEngine, ModelCategory};
use rustfft::{FftPlanner, num_complex::Complex};
use std::sync::Arc;

#[wasm_bindgen]
pub struct FNOSolver {
    weights: Vec<f32>,
    fft: Arc<dyn rustfft::Fft<f32>>,
    ifft: Arc<dyn rustfft::Fft<f32>>,
    modes: usize,
    width: usize,
    length: usize,
}

impl DyneEngine for FNOSolver {
    fn step(&mut self, input: &[f32]) -> Vec<f32> {
        let len = self.length;
        // 1. Lift (Copy input to all channels for demo)
        let mut x_in = vec![0.0; len * self.width];
        for i in 0..len {
            let val = if i < input.len() { input[i] } else { 0.0 };
            for c in 0..self.width {
                x_in[i * self.width + c] = val; 
            }
        }

        // 2. FFT (per channel)
        let mut x_spectral = vec![Complex::new(0.0, 0.0); self.width * len];
        for c in 0..self.width {
            let mut buffer = vec![Complex::new(0.0, 0.0); len];
            for i in 0..len {
                buffer[i] = Complex::new(x_in[i * self.width + c], 0.0);
            }
            self.fft.process(&mut buffer);
            for i in 0..len {
                x_spectral[c * len + i] = buffer[i];
            }
        }

        // 3. Spectral Filter (Low-pass demo)
        for c in 0..self.width {
            for i in 0..len {
                // Keep first `modes` and last `modes` (symmetric)
                // Filter out high freq
                if i > self.modes && i < (len - self.modes) {
                     x_spectral[c * len + i] = Complex::new(0.0, 0.0);
                }
            }
        }

        // 4. IFFT & Project (Output 1st channel)
        let mut output = vec![0.0; len];
        let norm = 1.0 / (len as f32);
        
        // Use 1st channel as output
        let mut buffer = vec![Complex::new(0.0, 0.0); len];
        for i in 0..len {
            buffer[i] = x_spectral[0 * len + i]; // channel 0
        }
        self.ifft.process(&mut buffer);
            
        for i in 0..len {
            output[i] = buffer[i].re * norm;
        }

        output
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::PDE
    }
    
    fn get_boundary(&self) -> Vec<f32> { vec![] }
    fn get_config(&self) -> String {
        format!("FNO Solver (RustFFT) modes={} width={}", self.modes, self.width)
    }
}

#[wasm_bindgen]
impl FNOSolver {
    #[wasm_bindgen(constructor)]
    pub fn new(weights_ptr: &[f32]) -> Self {
        let length = 32; 
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(length);
        let ifft = planner.plan_fft_inverse(length);

        Self {
            weights: weights_ptr.to_vec(),
            fft,
            ifft,
            modes: 4,
            width: 16,
            length,
        }
    }
    
    pub fn run(&mut self, input: &[f32]) -> Vec<f32> {
        self.step(input)
    }
    
    pub fn get_config(&self) -> String {
        <Self as DyneEngine>::get_config(self)
    }
}
"""

TOML_TEMPLATE = """
[package]
name = "{crate_name}"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
wasm-bindgen = "0.2"
console_error_panic_hook = "0.1"
dyne-core = {{ path = "../../dyne-core" }}
rustfft = "6.1"
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_file")
    parser.add_argument("output_dir")
    parser.add_argument("--name", default="dyne-solver-fno")
    args = parser.parse_args()
    
    model = onnx.load(args.onnx_file)
    weights = []
    # 修正版ラッパーを通したため、重みテンソルの名前や構造は変わっていますが
    # ここでは単純に「含まれる全パラメータ」をフラットにして保存します。
    # Rust側では今のところ「重みを使わずFFT動作確認」だけを行うため、これで足ります。
    for tensor in model.graph.initializer:
        arr = numpy_helper.to_array(tensor).astype(np.float32)
        weights.extend(arr.flatten())

    os.makedirs(args.output_dir, exist_ok=True)
    src_dir = os.path.join(args.output_dir, "src")
    os.makedirs(src_dir, exist_ok=True)

    with open(os.path.join(src_dir, "lib.rs"), "w", encoding="utf-8") as f:
        f.write(RUST_FNO_TEMPLATE)

    with open(os.path.join(args.output_dir, "Cargo.toml"), "w", encoding="utf-8") as f:
        f.write(TOML_TEMPLATE.format(crate_name=args.name))

    bin_path = os.path.join(args.output_dir, "model_weights.bin")
    np.array(weights, dtype=np.float32).tofile(bin_path)

    print(f"✅ Generated FNO Solver at: {args.output_dir}")