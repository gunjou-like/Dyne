import onnx
import numpy as np
import os
import argparse

# テンプレート: Rustコードの骨組み
# 注意: Rustの {} は Pythonのformat用と区別するため {{}} と二重にする必要があります
RUST_TEMPLATE = """
use wasm_bindgen::prelude::*;
use dyne_core::{{DyneEngine, ModelCategory}};

// --- 重みデータの埋め込み ---
{constants_code}

#[wasm_bindgen]
pub struct TranspiledSolver {{}}

impl DyneEngine for TranspiledSolver {{
    fn step(&mut self, input: &[f32]) -> Vec<f32> {{
        // 入力次元チェック (簡易)
        // let x = input; 
        
        // --- 推論ロジック (自動生成) ---
{inference_code}
        
        x.to_vec()
    }}

    fn category(&self) -> ModelCategory {{
        ModelCategory::PDE // 関数近似
    }}

    fn get_boundary(&self) -> Vec<f32> {{ vec![] }}

    fn get_config(&self) -> String {{
        "Transpiled ONNX Model (MLP)".to_string()
    }}
}}

#[wasm_bindgen]
impl TranspiledSolver {{
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {{
        Self {{}}
    }}
    
    pub fn run(&mut self, input: &[f32]) -> Vec<f32> {{
        self.step(input)
    }}

    // ▼▼▼ 追加箇所: JS向けにプロキシするメソッド ▼▼▼
    pub fn get_config(&self) -> String {{
        <Self as DyneEngine>::get_config(self)
    }}
}}

// --- ヘルパー関数: 行列ベクトル積 (y = Wx + b) ---
fn dense(input: &[f32], weights: &[f32], bias: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {{
    let mut output = vec![0.0; out_dim];
    for i in 0..out_dim {{
        let mut sum = 0.0;
        for j in 0..in_dim {{
            sum += input[j] * weights[i * in_dim + j];
        }}
        output[i] = sum + bias[i];
    }}
    output
}}

fn relu(input: &mut [f32]) {{
    for x in input.iter_mut() {{
        *x = x.max(0.0);
    }}
}}
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
web-sys = {{ version = "0.3", features = ["console"] }}
"""

def numpy_to_rust_array(arr, name):
    """numpy配列をRustの const 配列文字列に変換"""
    flat = arr.flatten()
    data_str = ", ".join(f"{x:.4f}" for x in flat)
    return f"const {name}: [f32; {len(flat)}] = [{data_str}];"

def compile_onnx(onnx_path, output_dir, crate_name):
    model = onnx.load(onnx_path)
    graph = model.graph
    
    # 1. 重み(Initializer)の抽出
    weights = {}
    for tensor in graph.initializer:
        weights[tensor.name] = onnx.numpy_helper.to_array(tensor)
    
    # 2. コード生成用バッファ
    constants_code = []
    inference_code = []
    inference_code.append("        let mut x = input.to_vec();") # 入力を可変ベクタにコピー
    
    # 3. ノード走査
    layer_count = 0
    
    for node in graph.node:
        if node.op_type == "Gemm":
            # input(0)=X, input(1)=W, input(2)=Bias(Optional)
            w_name = node.input[1]
            w_arr = weights[w_name]
            out_dim, in_dim = w_arr.shape
            
            # Biasの有無をチェック
            if len(node.input) > 2:
                b_name = node.input[2]
                b_arr = weights[b_name]
            else:
                print(f"  -> Layer {layer_count}: No bias found (creating zeros)")
                b_arr = np.zeros((out_dim,), dtype=np.float32)

            # 定数定義を追加
            w_const = f"W_{layer_count}"
            b_const = f"B_{layer_count}"
            constants_code.append(numpy_to_rust_array(w_arr, w_const))
            constants_code.append(numpy_to_rust_array(b_arr, b_const))
            
            # 推論コード追加
            code = f"        x = dense(&x, &{w_const}, &{b_const}, {out_dim}, {in_dim});"
            inference_code.append(code)
            
            layer_count += 1
            
        elif node.op_type == "Relu":
            inference_code.append("        relu(&mut x);")
            
    # 4. ファイル書き出し
    os.makedirs(output_dir, exist_ok=True)
    src_dir = os.path.join(output_dir, "src")
    os.makedirs(src_dir, exist_ok=True)
    
    # lib.rs
    rust_code = RUST_TEMPLATE.format(
        constants_code="\n".join(constants_code),
        inference_code="\n".join(inference_code)
    )
    with open(os.path.join(src_dir, "lib.rs"), "w", encoding="utf-8") as f:
        f.write(rust_code)
        
    # Cargo.toml
    toml_code = TOML_TEMPLATE.format(crate_name=crate_name)
    with open(os.path.join(output_dir, "Cargo.toml"), "w", encoding="utf-8") as f:
        f.write(toml_code)
        
    print(f"✅ Transpiled to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_file", help="Input .onnx file")
    parser.add_argument("output_dir", help="Output rust crate directory")
    parser.add_argument("--name", help="Crate name", default="dyne-solver-generated")
    args = parser.parse_args()
    
    compile_onnx(args.onnx_file, args.output_dir, args.name)