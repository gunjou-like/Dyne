use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ModelCategory {
    PDE, // 偏微分方程式 (場)
    ODE, // 常微分方程式 (状態)
}

// ▼▼▼ 新規追加: 領域分割の設定 ▼▼▼
#[wasm_bindgen]
#[derive(Clone, Copy, Debug)]
pub struct PartitionConfig {
    pub global_offset: usize, // 全体の中での開始位置 (例: 0 や 50)
    pub local_size: usize,    // このノードが担当するサイズ (例: 50)
    pub global_size: usize,   // 全体のサイズ (例: 100)
}

#[wasm_bindgen]
impl PartitionConfig {
    #[wasm_bindgen(constructor)]
    pub fn new(global_offset: usize, local_size: usize, global_size: usize) -> Self {
        Self { global_offset, local_size, global_size }
    }
}

// 注意: Trait定義自体はJSに公開されないため #[wasm_bindgen] は不要ですが、
// これを実装する構造体を通して機能が使われます。
pub trait DyneEngine {
    fn step(&mut self, input: &[f32]) -> Vec<f32>;
    
    fn category(&self) -> ModelCategory;
    
    // 既存の可視化用 (Global View用)
    fn get_boundary(&self) -> Vec<f32>; 
    fn get_config(&self) -> String;

    // ▼▼▼ 新規追加: 境界同期プロトコル (デフォルト実装あり) ▼▼▼
    
    // パーティション設定の注入
    fn set_partition(&mut self, _config: PartitionConfig) {
        // 必要に応じてオーバーライドする
    }

    // 左端のゴーストセルを取得 (送信)
    fn get_left_ghost(&self) -> Vec<f32> { vec![] }
    
    // 右端のゴーストセルを取得 (送信)
    fn get_right_ghost(&self) -> Vec<f32> { vec![] }
    
    // 左端からデータを受信 (隣の右端データを受け取る)
    fn set_left_ghost(&mut self, _data: &[f32]) {}
    
    // 右端からデータを受信 (隣の左端データを受け取る)
    fn set_right_ghost(&mut self, _data: &[f32]) {}
}