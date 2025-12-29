import onnx
import numpy as np
import os
import argparse
from onnx import numpy_helper

# --- Rust テンプレート (v0.4 Weight Separation版) ---
RUST_TEMPLATE = """
use wasm_bindgen::prelude::*;
use dyne_core::{{DyneEngine, ModelCategory}};

#[wasm_bindgen]
pub struct TranspiledSolver {{
    // 重みデータ全体を保持するフラットなベクタ
    weights: Vec<f32>,
}}

impl DyneEngine for TranspiledSolver {{
    fn step(&mut self, input: &[f32]) -> Vec<f32> {{
        let mut x = input.to_vec();
        
        // --- 推論ロジック ---
{inference_code}
        
        x
    }}

    fn category(&self) -> ModelCategory {{
        ModelCategory::PDE
    }}

    fn get_boundary(&self) -> Vec<f32> {{ vec![] }}

    fn get_config(&self) -> String {{
        "Transpiled ONNX Model (v0.4: Binary Weights)".to_string()
    }}
}}

#[wasm_bindgen]
impl TranspiledSolver {{
    // コンストラクタ: JSからFloat32Arrayを受け取る
    #[wasm_bindgen(constructor)]
    pub fn new(weights_ptr: &[f32]) -> Self {{
        Self {{
            weights: weights_ptr.to_vec(),
        }}
    }}
    
    pub fn run(&mut self, input: &[f32]) -> Vec<f32> {{
        self.step(input)
    }}

    pub fn get_config(&self) -> String {{
        <Self as DyneEngine>::get_config(self)
    }}
}}

// --- Helper Functions ---

fn relu(input: &mut [f32]) {{
    for x in input.iter_mut() {{
        *x = x.max(0.0);
    }}
}}

fn conv2d(
    input: &[f32], 
    all_weights: &[f32], // 全重みデータ
    w_start: usize, w_end: usize, // Weightの範囲
    b_start: usize, b_end: usize, // Biasの範囲
    in_c: usize, out_c: usize, 
    h: usize, w: usize,
    k_h: usize, k_w: usize,
    pad: usize, stride: usize
) -> Vec<f32> {{
    // スライス切り出し
    let weight = &all_weights[w_start..w_end];
    let bias = if b_end > b_start {{ &all_weights[b_start..b_end] }} else {{ &[] }};

    let out_h = (h + 2 * pad - k_h) / stride + 1;
    let out_w = (w + 2 * pad - k_w) / stride + 1;
    let mut output = vec![0.0; out_c * out_h * out_w];

    for oc in 0..out_c {{
        let bias_val = if bias.len() > 0 {{ bias[oc] }} else {{ 0.0 }};
        
        for oh in 0..out_h {{
            for ow in 0..out_w {{
                let mut sum = bias_val;
                
                let h_start = (oh * stride) as isize - pad as isize;
                let w_start = (ow * stride) as isize - pad as isize;

                for ic in 0..in_c {{
                    for kh in 0..k_h {{
                        for kw in 0..k_w {{
                            let cur_h = h_start + kh as isize;
                            let cur_w = w_start + kw as isize;

                            if cur_h >= 0 && cur_h < h as isize && cur_w >= 0 && cur_w < w as isize {{
                                let input_idx = ic * (h * w) + (cur_h as usize) * w + (cur_w as usize);
                                // Weight index within the slice
                                let weight_idx = oc * (in_c * k_h * k_w) + ic * (k_h * k_w) + kh * k_w + kw;
                                sum += input[input_idx] * weight[weight_idx];
                            }}
                        }}
                    }}
                }}
                
                let out_idx = oc * (out_h * out_w) + oh * out_w + ow;
                output[out_idx] = sum;
            }}
        }}
    }}
    output
}}

fn dense(
    input: &[f32], 
    all_weights: &[f32],
    w_start: usize, w_end: usize,
    b_start: usize, b_end: usize,
    out_dim: usize, in_dim: usize
) -> Vec<f32> {{
    let weights = &all_weights[w_start..w_end];
    let bias = if b_end > b_start {{ &all_weights[b_start..b_end] }} else {{ &[] }};

    let mut output = vec![0.0; out_dim];
    for i in 0..out_dim {{
        let mut sum = 0.0;
        for j in 0..in_dim {{
            sum += input[j] * weights[i * in_dim + j];
        }}
        output[i] = sum + if bias.len() > 0 {{ bias[i] }} else {{ 0.0 }};
    }}
    output
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

def get_attr(node, attr_name, default=None):
    for attr in node.attribute:
        if attr.name == attr_name:
            if attr.type == onnx.AttributeProto.INTS:
                return list(attr.ints)
            elif attr.type == onnx.AttributeProto.INT:
                return attr.i
    return default

def compile_onnx_v04(onnx_path, output_dir, crate_name, input_h=5, input_w=5):
    model = onnx.load(onnx_path)
    graph = model.graph
    
    # 1. 重みデータの収集と結合
    weights_map = {} # name -> numpy array
    for tensor in graph.initializer:
        weights_map[tensor.name] = numpy_helper.to_array(tensor).astype(np.float32)
    
    # バイナリバッファ
    binary_blob = []
    # オフセット管理: name -> (start_idx, end_idx)
    offset_map = {} 
    current_offset = 0

    def append_weight(name, array):
        nonlocal current_offset
        flat = array.flatten()
        length = len(flat)
        binary_blob.extend(flat)
        start = current_offset
        end = current_offset + length
        offset_map[name] = (start, end)
        current_offset = end
        return start, end

    # 2. 推論コード生成と重み登録
    inference_code = []
    current_h = input_h
    current_w = input_w
    current_c = 1 
    
    layer_count = 0
    
    for node in graph.node:
        if node.op_type == "Conv":
            w_name = node.input[1]
            w_arr = weights_map[w_name]
            
            # バイナリに追加 (未登録なら)
            if w_name not in offset_map:
                append_weight(w_name, w_arr)
            w_start, w_end = offset_map[w_name]

            b_start, b_end = 0, 0
            if len(node.input) > 2:
                b_name = node.input[2]
                if b_name not in offset_map:
                    append_weight(b_name, weights_map[b_name])
                b_start, b_end = offset_map[b_name]

            out_c, in_c, k_h, k_w = w_arr.shape
            pads = get_attr(node, "pads", [0, 0, 0, 0])
            pad_val = pads[0]
            strides = get_attr(node, "strides", [1, 1])
            stride_val = strides[0]
            
            # 生成コード: オフセット値を直接埋め込む
            code = f"""
        // Conv Layer {layer_count}
        x = conv2d(&x, &self.weights, 
                   {w_start}, {w_end}, {b_start}, {b_end},
                   {in_c}, {out_c}, {current_h}, {current_w}, 
                   {k_h}, {k_w}, {pad_val}, {stride_val});
            """
            inference_code.append(code.strip())
            
            current_h = (current_h + 2 * pad_val - k_h) // stride_val + 1
            current_w = (current_w + 2 * pad_val - k_w) // stride_val + 1
            current_c = out_c
            layer_count += 1
            
        elif node.op_type == "Gemm":
            w_name = node.input[1]
            w_arr = weights_map[w_name]
            if w_name not in offset_map:
                append_weight(w_name, w_arr)
            w_start, w_end = offset_map[w_name]
            
            b_start, b_end = 0, 0
            if len(node.input) > 2:
                b_name = node.input[2]
                if b_name not in offset_map:
                    append_weight(b_name, weights_map[b_name])
                b_start, b_end = offset_map[b_name]

            out_dim, in_dim = w_arr.shape
            
            # Flatten if needed (簡易判定: 前段がConvならFlatten扱い)
            if current_h > 1 or current_w > 1:
                # 本来はReshapeノードなどをパースすべきだが、簡易的に対応
                pass 

            code = f"""
        // Dense Layer {layer_count}
        x = dense(&x, &self.weights,
                  {w_start}, {w_end}, {b_start}, {b_end},
                  {out_dim}, {in_dim});
            """
            inference_code.append(code.strip())
            layer_count += 1

        elif node.op_type == "Relu":
            inference_code.append("        relu(&mut x);")

    # 3. ファイル書き出し
    os.makedirs(output_dir, exist_ok=True)
    src_dir = os.path.join(output_dir, "src")
    os.makedirs(src_dir, exist_ok=True)
    
    # Rust Source
    rust_code = RUST_TEMPLATE.format(
        inference_code="\n".join(inference_code),
    )
    with open(os.path.join(src_dir, "lib.rs"), "w", encoding="utf-8") as f:
        f.write(rust_code)
        
    # Cargo.toml
    with open(os.path.join(output_dir, "Cargo.toml"), "w", encoding="utf-8") as f:
        f.write(TOML_TEMPLATE.format(crate_name=crate_name))
    
    # Binary Blob (.bin)
    bin_path = os.path.join(output_dir, "model_weights.bin")
    # float32配列をバイト列に変換して保存
    np_blob = np.array(binary_blob, dtype=np.float32)
    np_blob.tofile(bin_path)
        
    print(f"✅ Transpiled to: {output_dir}")
    print(f"📦 Weights saved to: {bin_path} ({len(np_blob)*4} bytes)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_file")
    parser.add_argument("output_dir")
    parser.add_argument("--name", default="dyne-solver-generated")
    parser.add_argument("--input_size", type=int, default=5)
    args = parser.parse_args()
    
    compile_onnx_v04(args.onnx_file, args.output_dir, args.name, input_h=args.input_size, input_w=args.input_size)