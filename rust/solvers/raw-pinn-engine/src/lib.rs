use wasm_bindgen::prelude::*;
use serde::Deserialize;
mod constants; // 自動生成されるファイルを読み込む
#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

// JSONデータ構造定義
#[derive(Deserialize)]
struct ModelWeights {
    conv1_w: Vec<f32>,
    conv1_b: Vec<f32>,
    conv2_w: Vec<f32>,
    conv2_b: Vec<f32>,
    head_w: Vec<f32>,
    head_b: Vec<f32>,
    hidden_dim: usize,
}

// 1D畳み込み層 (自作実装)
struct Conv1d {
    weight: Vec<f32>,
    bias: Vec<f32>,
    in_ch: usize,
    out_ch: usize,
    k: usize,
}

impl Conv1d {
    fn new(w: Vec<f32>, b: Vec<f32>, in_ch: usize, out_ch: usize, k: usize) -> Self {
        Self { weight: w, bias: b, in_ch, out_ch, k }
    }

    fn forward(&self, input: &[f32], length: usize, use_tanh: bool) -> Vec<f32> {
        // 出力バッファ確保
        let mut output = vec![0.0; self.out_ch * length];
        let pad = 1; // Padding=1固定

        for out_c in 0..self.out_ch {
            for i in 0..length {
                let mut sum = self.bias[out_c];
                
                for in_c in 0..self.in_ch {
                    for k_offset in 0..self.k {
                        // インデックス計算 (i - pad + k)
                        let input_idx_signed = (i as isize) - (pad as isize) + (k_offset as isize);
                        
                        // 範囲内チェック
                        if input_idx_signed >= 0 && input_idx_signed < length as isize {
                            let val = input[in_c * length + (input_idx_signed as usize)];
                            
                            // 重みインデックス (flattenされている)
                            let w_idx = out_c * (self.in_ch * self.k) + in_c * self.k + k_offset;
                            sum += val * self.weight[w_idx];
                        }
                    }
                }
                
                // 活性化関数
                let idx = out_c * length + i;
                if use_tanh {
                    output[idx] = sum.tanh();
                } else {
                    output[idx] = sum;
                }
            }
        }
        output
    }
}

// JS公開用クラス
#[wasm_bindgen]
pub struct DyneRuntime {
    layer1: Conv1d,
    layer2: Conv1d,
    head: Conv1d,
}

#[wasm_bindgen]
impl DyneRuntime {
    // コンストラクタ: JSON文字列を受け取る
    #[wasm_bindgen(constructor)]
    pub fn new(json_str: &str) -> Result<DyneRuntime, JsValue> {
        let w: ModelWeights = serde_json::from_str(json_str)
            .map_err(|e| e.to_string())?;

        let h = w.hidden_dim;
        let k = 3;

        Ok(DyneRuntime {
            layer1: Conv1d::new(w.conv1_w, w.conv1_b, 1, h, k),
            layer2: Conv1d::new(w.conv2_w, w.conv2_b, h, h, k),
            head:   Conv1d::new(w.head_w,  w.head_b,  h, 1, k),
        })
    }

    // 実行: 入力配列(1次元)を受け取り、次のステップを返す
    pub fn run(&self, input_wave: &[f32]) -> Vec<f32> {
        let len = input_wave.len();

        // ▼▼▼ 追加: 設定ファイルと実際のデータの整合性チェック ▼▼▼
        if len != constants::WIDTH {
            // ブラウザのコンソールにエラーを出してパニックさせる
            web_sys::console::error_1(&format!(
                "Dyne Error: Input width ({}) matches dyne.toml config ({})",
                len, constants::WIDTH
            ).into());
            panic!("Dimension mismatch! Configured: {}, Got: {}", constants::WIDTH, len);
        }
        // ▲▲▲ 追加終わり ▲▲▲
        
        let x1 = self.layer1.forward(input_wave, len, true);
        let x2 = self.layer2.forward(&x1, len, true);
        let out = self.head.forward(&x2, len, false);
        
        out
    }
}