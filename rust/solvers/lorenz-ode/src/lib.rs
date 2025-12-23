use wasm_bindgen::prelude::*;
use dyne_core::{DyneEngine, ModelCategory};

#[wasm_bindgen]
pub struct LorenzSolver {
    // 状態変数 (x, y, z)
    state: Vec<f32>,
    // パラメータ
    sigma: f32,
    rho: f32,
    beta: f32,
    dt: f32,
}

impl DyneEngine for LorenzSolver {
    fn step(&mut self, _input: &[f32]) -> Vec<f32> {
        // ODEの場合、外部からの空間入力(_input)は無視して
        // 内部状態(self.state)を更新します。
        
        let x = self.state[0];
        let y = self.state[1];
        let z = self.state[2];

        // ルンゲ・クッタ法 (RK4) ではなく簡易オイラー法で実装 (軽量化のため)
        // dx/dt = sigma * (y - x)
        // dy/dt = x * (rho - z) - y
        // dz/dt = x * y - beta * z

        let dx = self.sigma * (y - x);
        let dy = x * (self.rho - z) - y;
        let dz = x * y - self.beta * z;

        self.state[0] += dx * self.dt;
        self.state[1] += dy * self.dt;
        self.state[2] += dz * self.dt;

        self.state.clone()
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::ODE
    }

    fn get_boundary(&self) -> Vec<f32> {
        // 結合ODE（Coupled ODEs）のために状態すべてを返す
        self.state.clone()
    }

    fn get_config(&self) -> String {
        format!("Lorenz ODE (sigma={}, rho={}, beta={})", self.sigma, self.rho, self.beta)
    }
}

// WASM公開用
#[wasm_bindgen]
impl LorenzSolver {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            state: vec![1.0, 1.0, 1.0], // 初期値
            sigma: 10.0,
            rho: 28.0,
            beta: 8.0 / 3.0,
            dt: 0.01,
        }
    }

    pub fn run(&mut self, _input: &[f32]) -> Vec<f32> {
        self.step(_input)
    }
}