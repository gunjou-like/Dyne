use wasm_bindgen::prelude::*;
use dyne_core::{DyneEngine, ModelCategory, PartitionConfig};

// 1次元波動方程式 (分割対応)
// u_tt = c^2 * u_xx
#[wasm_bindgen]
pub struct SplitWaveSolver {
    u: Vec<f32>,       // 現在のステップ
    u_prev: Vec<f32>,  // 1つ前のステップ
    nx: usize,         // ローカルグリッド数
    dx: f32,
    dt: f32,
    c: f32,
    
    // パーティション情報
    offset: usize,     // Global offset
    
    // Ghost Cells (隣からの受信データ)
    ghost_left: f32,
    ghost_right: f32,
}

impl DyneEngine for SplitWaveSolver {
    fn step(&mut self, _input: &[f32]) -> Vec<f32> {
        let mut u_next = vec![0.0; self.nx];
        let r = (self.c * self.dt / self.dx).powi(2);

        for i in 0..self.nx {
            // ラプラシアン (u_xx) の計算に必要な近傍点
            let u_curr = self.u[i];
            
            // 左隣の値: 左端(i=0)ならGhost Cell、それ以外は配列内
            let u_left = if i == 0 {
                self.ghost_left 
            } else {
                self.u[i - 1]
            };

            // 右隣の値: 右端(i=nx-1)ならGhost Cell、それ以外は配列内
            let u_right = if i == self.nx - 1 {
                self.ghost_right
            } else {
                self.u[i + 1]
            };

            // 波動方程式の差分法
            // u_next = 2u - u_prev + r * (u_right - 2u + u_left)
            u_next[i] = 2.0 * u_curr - self.u_prev[i] + r * (u_right - 2.0 * u_curr + u_left);
        }

        // 時間更新
        self.u_prev = self.u.clone();
        self.u = u_next.clone();
        
        // Ghost Cellをリセット (通信遅れ対策として古い値を保持する戦略もあるが、今回はクリア)
        // self.ghost_left = 0.0; // 境界条件として0固定にする場合以外は保持したほうがよいかも

        self.u.clone()
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::PDE
    }

    fn get_boundary(&self) -> Vec<f32> {
        self.u.clone()
    }

    fn get_config(&self) -> String {
        format!("Split Wave Solver (Offset: {}, Size: {})", self.offset, self.nx)
    }

    // ▼▼▼ Protocol Implementation ▼▼▼

    fn set_partition(&mut self, config: PartitionConfig) {
        self.offset = config.global_offset;
        // 本来はresizeなども必要だが、デモ用に固定サイズまたは再初期化を想定
    }

    fn get_left_ghost(&self) -> Vec<f32> {
        // 自分の左端 (i=0) を隣に送る
        if self.u.len() > 0 { vec![self.u[0]] } else { vec![0.0] }
    }

    fn get_right_ghost(&self) -> Vec<f32> {
        // 自分の右端 (i=nx-1) を隣に送る
        if self.u.len() > 0 { vec![self.u[self.nx - 1]] } else { vec![0.0] }
    }

    fn set_left_ghost(&mut self, data: &[f32]) {
        if data.len() > 0 {
            self.ghost_left = data[0]; // 左の壁 (または隣の右端)
        }
    }

    fn set_right_ghost(&mut self, data: &[f32]) {
        if data.len() > 0 {
            self.ghost_right = data[0]; // 右の壁 (または隣の左端)
        }
    }
}

#[wasm_bindgen]
impl SplitWaveSolver {
    #[wasm_bindgen(constructor)]
    pub fn new(nx: usize) -> Self {
        let dx = 1.0 / (nx as f32 * 2.0); // 2つ繋げると全体で1.0になる想定
        let dt = 0.01;
        let c = 1.0;
        
        let mut u = vec![0.0; nx];
        
        // 初期条件: ガウス波束を中心に配置...したいが、
        // 分割されているため「自分がどこか」によって初期値を変える必要がある。
        // ここではJS側から set_u できるようにするか、単純にゼロスタートして境界から波を入れる。
        
        Self {
            u: u.clone(),
            u_prev: u,
            nx,
            dx,
            dt,
            c,
            offset: 0,
            ghost_left: 0.0,  // デフォルト固定端
            ghost_right: 0.0, // デフォルト固定端
        }
    }

    // JSから初期状態や入力波を注入するためのヘルパー
    pub fn set_state(&mut self, data: &[f32]) {
        self.u = data.to_vec();
        self.u_prev = data.to_vec();
    }
    
    // JS向けプロキシ
    pub fn run(&mut self) -> Vec<f32> {
        self.step(&[])
    }

        // ▼▼▼ このメソッドを追加してください ▼▼▼
    pub fn set_partition(&mut self, config: PartitionConfig) {
        <Self as DyneEngine>::set_partition(self, config);
    }
    // ▲▲▲▲▲▲
    
    // Boundary Sync Methods exposed to JS
    pub fn get_left_out(&self) -> Vec<f32> { <Self as DyneEngine>::get_left_ghost(self) }
    pub fn get_right_out(&self) -> Vec<f32> { <Self as DyneEngine>::get_right_ghost(self) }
    pub fn set_left_in(&mut self, d: &[f32]) { <Self as DyneEngine>::set_left_ghost(self, d); }
    pub fn set_right_in(&mut self, d: &[f32]) { <Self as DyneEngine>::set_right_ghost(self, d); }
}

