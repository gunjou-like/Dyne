import onnx
import numpy as np
import os
import argparse
from onnx import numpy_helper

# --- Rust テンプレート (Conv2d 対応版) ---
RUST_TEMPLATE = """
use wasm_bindgen::prelude::*;
use dyne_core::{{DyneEngine, ModelCategory}};

// --- 重みデータの埋め込み ---
{constants_code}

#[wasm_bindgen]
pub struct TranspiledSolver {{}}

impl DyneEngine for TranspiledSolver {{
    fn step(&mut self, input: &[f32]) -> Vec<f32> {{
        // 入力次元: [Channel=1, Height={input_h}, Width={input_w}] を想定
        let mut x = input.to_vec();
        
        // --- 推論ロジック (自動生成) ---
{inference_code}
        
        x
    }}

    fn category(&self) -> ModelCategory {{
        ModelCategory::PDE
    }}

    fn get_boundary(&self) -> Vec<f32> {{ vec![] }}

    fn get_config(&self) -> String {{
        "Transpiled ONNX Model (CNN)".to_string()
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

// Naive Conv2d Implementation (NCHW format)
// Input: [C_in, H, W] -> Output: [C_out, H_out, W_out]
fn conv2d(
    input: &[f32], 
    weight: &[f32], 
    bias: &[f32],
    in_c: usize, out_c: usize, 
    h: usize, w: usize,
    k_h: usize, k_w: usize,
    pad: usize, stride: usize
) -> Vec<f32> {{
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

// Helper for Linear (Gemm) if mixed
fn dense(input: &[f32], weights: &[f32], bias: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {{
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

def numpy_to_rust_array(arr, name):
    flat = arr.flatten()
    data_str = ", ".join(f"{x:.4f}" for x in flat)
    return f"const {name}: [f32; {len(flat)}] = [{data_str}];"

def get_attr(node, attr_name, default=None):
    for attr in node.attribute:
        if attr.name == attr_name:
            if attr.type == onnx.AttributeProto.INTS:
                return list(attr.ints)
            elif attr.type == onnx.AttributeProto.INT:
                return attr.i
    return default

def compile_onnx(onnx_path, output_dir, crate_name, input_h=5, input_w=5):
    model = onnx.load(onnx_path)
    graph = model.graph
    
    weights = {}
    for tensor in graph.initializer:
        weights[tensor.name] = numpy_helper.to_array(tensor)
    
    constants_code = []
    inference_code = []
    
    # 状態追跡変数 (簡易的なShape Inference)
    current_h = input_h
    current_w = input_w
    current_c = 1 # 入力チャンネル (初期値)
    
    layer_count = 0
    
    for node in graph.node:
        if node.op_type == "Conv":
            # Input: [X, W, B]
            w_name = node.input[1]
            w_arr = weights[w_name]
            # Conv Weight: [Out_C, In_C, K_H, K_W]
            out_c, in_c, k_h, k_w = w_arr.shape
            
            b_name = None
            b_arr = np.array([], dtype=np.float32)
            if len(node.input) > 2:
                b_name = node.input[2]
                b_arr = weights[b_name]
                
            # Attributes
            pads = get_attr(node, "pads", [0, 0, 0, 0]) # [y_begin, x_begin, y_end, x_end]
            pad_val = pads[0] # assume symmetric padding for now
            strides = get_attr(node, "strides", [1, 1])
            stride_val = strides[0]
            
            # Generate Rust Constants
            w_const = f"W_{layer_count}"
            b_const = f"B_{layer_count}"
            constants_code.append(numpy_to_rust_array(w_arr, w_const))
            constants_code.append(numpy_to_rust_array(b_arr, b_const))
            
            # Generate Call
            code = f"""
        // Conv Layer {layer_count}: In[{current_c}x{current_h}x{current_w}] -> Out[{out_c}x?x?]
        x = conv2d(&x, &{w_const}, &{b_const}, 
                   {current_c}, {out_c}, {current_h}, {current_w}, 
                   {k_h}, {k_w}, {pad_val}, {stride_val});
            """
            inference_code.append(code.strip())
            
            # Update Shape State
            current_h = (current_h + 2 * pad_val - k_h) // stride_val + 1
            current_w = (current_w + 2 * pad_val - k_w) // stride_val + 1
            current_c = out_c
            layer_count += 1
            
        elif node.op_type == "Relu":
            inference_code.append("        relu(&mut x);")
            
        elif node.op_type == "Gemm":
            # (省略: 前回のコードと同様だが、CNN直後のFlattenなどを考慮する必要あり)
            # 今回はCNNのみのデモなので簡易対応
            pass

    # ファイル書き出し
    os.makedirs(output_dir, exist_ok=True)
    src_dir = os.path.join(output_dir, "src")
    os.makedirs(src_dir, exist_ok=True)
    
    rust_code = RUST_TEMPLATE.format(
        constants_code="\n".join(constants_code),
        inference_code="\n".join(inference_code),
        input_h=input_h,
        input_w=input_w
    )
    
    with open(os.path.join(src_dir, "lib.rs"), "w", encoding="utf-8") as f:
        f.write(rust_code)
        
    with open(os.path.join(output_dir, "Cargo.toml"), "w", encoding="utf-8") as f:
        f.write(TOML_TEMPLATE.format(crate_name=crate_name))
        
    print(f"✅ Transpiled to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_file")
    parser.add_argument("output_dir")
    parser.add_argument("--name", default="dyne-solver-generated")
    parser.add_argument("--input_size", type=int, default=5, help="Input height/width (square)")
    args = parser.parse_args()
    
    compile_onnx(args.onnx_file, args.output_dir, args.name, input_h=args.input_size, input_w=args.input_size)