use std::io::Cursor;
use tract_onnx::prelude::*;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct DyneRuntime {
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
}

#[wasm_bindgen]
impl DyneRuntime {
    #[wasm_bindgen(constructor)]
    pub fn new(model_bytes: &[u8], input_len: usize) -> Result<DyneRuntime, JsValue> {
        web_sys::console::log_1(&"Loading ONNX model...".into());
        
        let mut model = tract_onnx::onnx()
            .model_for_read(&mut Cursor::new(model_bytes))
            .map_err(|e| format!("Failed to load ONNX: {}", e))?;

        web_sys::console::log_1(&format!("Model loaded. Nodes: {}", model.nodes.len()).into());

        // Conv1D固定サイズ用に3次元入力 [Batch, Channel, Length] = [1, 1, input_len]
        model.set_input_fact(0, f32::fact([1, 1, input_len]).into())
             .map_err(|e| format!("Failed to set input shape: {}", e))?;

        web_sys::console::log_1(&"Input shape set".into());
        
        // 型情報を伝播させる
        model.analyse(false)
             .map_err(|e| format!("Failed to analyse model: {}", e))?;
        
        web_sys::console::log_1(&"Model analysed".into());

        // 型推論を実行する前に、モデルを解析
        model.analyse(false)
             .map_err(|e| format!("Failed to analyse model: {}", e))?;

        web_sys::console::log_1(&"Model analysed".into());

        // 型推論を試みる
        let typed_model = model
            .into_typed()
            .map_err(|e| format!("Failed type inference: {}", e))?;

        web_sys::console::log_1(&"Type inference successful".into());

        // 実行可能プランに変換（最適化をスキップ）
        let plan = typed_model
            .into_runnable()
            .map_err(|e| format!("Failed to create runnable: {}", e))?;

        web_sys::console::log_1(&"Runnable plan created".into());

        Ok(DyneRuntime { model: plan })
    }

    pub fn run(&self, input_wave: &[f32]) -> Result<Vec<f32>, JsValue> {
        let len = input_wave.len();
        
        // Conv1D用に3次元テンソル [1, 1, len] として作成
        let tensor = tract_ndarray::Array3::from_shape_vec((1, 1, len), input_wave.to_vec())
            .map_err(|e| format!("Failed to create tensor: {}", e))?;
        
        let input_tensor = tensor.into_tensor();

        let result = self.model.run(tvec!(input_tensor.into()))
            .map_err(|e| format!("Inference failed: {}", e))?;

        let output_tensor = result[0].to_array_view::<f32>()
            .map_err(|e| format!("Failed to get output: {}", e))?;
        
        Ok(output_tensor.iter().cloned().collect())
    }
}