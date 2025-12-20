/// 最適化の有無をテストするスクリプト
/// 
/// 実行方法:
///     cargo test --release -- --nocapture
/// 
/// テスト内容:
/// 1. 最適化なしでモデルを読み込み
/// 2. 最適化ありでモデルを読み込み
/// 3. tract のバージョンと Conv1D サポート状況を確認

use std::fs;
use std::io::Cursor;
use tract_onnx::prelude::*;

#[test]
fn test_model_loading_without_optimization() {
    println!("\n=== Test 1: Loading model WITHOUT optimization ===");
    
    let model_path = "../dist/part_0.onnx";
    
    match fs::read(model_path) {
        Ok(model_bytes) => {
            println!("✓ Model file loaded: {} bytes", model_bytes.len());
            
            // モデルを読み込み
            match tract_onnx::onnx().model_for_read(&mut Cursor::new(&model_bytes)) {
                Ok(mut model) => {
                    println!("✓ ONNX model parsed successfully");
                    println!("  Nodes: {}", model.nodes.len());
                    println!("  Inputs: {:?}", model.inputs);
                    println!("  Outputs: {:?}", model.outputs);
                    
                    // ノードの種類を表示
                    println!("\n  Node types:");
                    for (i, node) in model.nodes.iter().enumerate() {
                        println!("    [{}] {} - {}", i, node.name, node.op);
                    }
                    
                    // 入力シェイプを設定
                    let input_len = 52;
                    match model.set_input_fact(0, f32::fact([1, 1, input_len]).into()) {
                        Ok(_) => println!("\n✓ Input shape set to [1, 1, {}]", input_len),
                        Err(e) => println!("\n✗ Failed to set input shape: {}", e),
                    }
                    
                    // 型推論（最適化なし）
                    match model.into_typed() {
                        Ok(typed_model) => {
                            println!("✓ Type inference successful (without optimization)");
                            
                            // 最適化なしで実行可能プランを作成
                            match typed_model.into_runnable() {
                                Ok(plan) => {
                                    println!("✓ Runnable plan created WITHOUT optimization");
                                    
                                    // 実際に推論を試す
                                    let input_data = vec![0.0f32; input_len];
                                    let tensor = tract_ndarray::Array3::from_shape_vec(
                                        (1, 1, input_len),
                                        input_data
                                    ).expect("Failed to create tensor");
                                    
                                    match plan.run(tvec!(tensor.into_tensor().into())) {
                                        Ok(result) => {
                                            println!("✓ Inference successful! Output shape: {:?}", result[0].shape());
                                        }
                                        Err(e) => {
                                            println!("✗ Inference failed: {}", e);
                                        }
                                    }
                                }
                                Err(e) => {
                                    println!("✗ Failed to create runnable plan: {}", e);
                                }
                            }
                        }
                        Err(e) => {
                            println!("✗ Type inference failed: {}", e);
                            eprintln!("Detailed error: {:?}", e);
                        }
                    }
                }
                Err(e) => {
                    println!("✗ Failed to parse ONNX model: {}", e);
                    panic!("Model parsing failed");
                }
            }
        }
        Err(e) => {
            println!("✗ Failed to read model file: {}", e);
            println!("  Note: Run this test from the wasm directory");
            // ファイルが見つからない場合はテストをスキップ
        }
    }
}

#[test]
fn test_model_loading_with_optimization() {
    println!("\n=== Test 2: Loading model WITH optimization ===");
    
    let model_path = "../dist/part_0.onnx";
    
    match fs::read(model_path) {
        Ok(model_bytes) => {
            println!("✓ Model file loaded: {} bytes", model_bytes.len());
            
            // モデルを読み込み
            match tract_onnx::onnx().model_for_read(&mut Cursor::new(&model_bytes)) {
                Ok(mut model) => {
                    println!("✓ ONNX model parsed successfully");
                    
                    // 入力シェイプを設定
                    let input_len = 52;
                    model.set_input_fact(0, f32::fact([1, 1, input_len]).into())
                        .expect("Failed to set input shape");
                    
                    println!("✓ Input shape set to [1, 1, {}]", input_len);
                    
                    // 型推論
                    match model.into_typed() {
                        Ok(typed_model) => {
                            println!("✓ Type inference successful");
                            
                            // 最適化を試みる
                            match typed_model.into_optimized() {
                                Ok(optimized) => {
                                    println!("✓ Optimization successful!");
                                    
                                    // 実行可能プランを作成
                                    match optimized.into_runnable() {
                                        Ok(_plan) => {
                                            println!("✓ Runnable plan created WITH optimization");
                                        }
                                        Err(e) => {
                                            println!("✗ Failed to create runnable plan: {}", e);
                                        }
                                    }
                                }
                                Err(e) => {
                                    println!("✗ Optimization failed: {}", e);
                                    println!("  This is the error we're investigating!");
                                }
                            }
                        }
                        Err(e) => {
                            println!("✗ Type inference failed: {}", e);
                        }
                    }
                }
                Err(e) => {
                    println!("✗ Failed to parse ONNX model: {}", e);
                }
            }
        }
        Err(e) => {
            println!("✗ Failed to read model file: {}", e);
            println!("  Note: Run this test from the wasm directory");
        }
    }
}

#[test]
fn test_tract_version_and_conv_support() {
    println!("\n=== Test 3: Tract version and Conv1D support ===");
    
    println!("Tract Core version: {}", env!("CARGO_PKG_VERSION"));
    println!("Tract ONNX version: 0.21.7 (from Cargo.toml)");
    
    // Conv1D のサポート状況を確認
    // Tract 0.21.7 では Conv はサポートされているはず
    println!("\nConv operator support:");
    println!("  Conv (1D/2D/3D): Supported in tract-onnx 0.21.x");
    println!("  Note: Conv1D is represented as Conv with specific dimension");
    
    // モデルを読み込んでConvノードを確認
    let model_path = "../dist/part_0.onnx";
    if let Ok(model_bytes) = fs::read(model_path) {
        if let Ok(model) = tract_onnx::onnx().model_for_read(&mut Cursor::new(&model_bytes)) {
            println!("\nSearching for Conv nodes in model:");
            let mut found_conv = false;
            for node in model.nodes.iter() {
                let op_name = format!("{}", node.op);
                if op_name.contains("Conv") {
                    println!("  ✓ Found: {} ({})", node.name, op_name);
                    found_conv = true;
                }
            }
            if !found_conv {
                println!("  No Conv nodes found in model");
            }
        }
    }
}

#[test]
fn test_simple_inference_without_optimization() {
    println!("\n=== Test 4: Simple inference test (no optimization) ===");
    
    let model_path = "../dist/part_0.onnx";
    
    if let Ok(model_bytes) = fs::read(model_path) {
        if let Ok(mut model) = tract_onnx::onnx().model_for_read(&mut Cursor::new(&model_bytes)) {
            let input_len = 52;
            model.set_input_fact(0, f32::fact([1, 1, input_len]).into())
                .expect("Failed to set input shape");
            
            if let Ok(typed_model) = model.into_typed() {
                // 最適化なしで実行
                if let Ok(plan) = typed_model.into_runnable() {
                    println!("✓ Model ready for inference (no optimization)");
                    
                    // ダミー入力で推論テスト
                    let input_data = vec![0.0f32; input_len];
                    let tensor = tract_ndarray::Array3::from_shape_vec(
                        (1, 1, input_len), 
                        input_data
                    ).expect("Failed to create tensor");
                    
                    let input_tensor = tensor.into_tensor();
                    
                    match plan.run(tvec!(input_tensor.into())) {
                        Ok(result) => {
                            println!("✓ Inference successful!");
                            println!("  Output shape: {:?}", result[0].shape());
                        }
                        Err(e) => {
                            println!("✗ Inference failed: {}", e);
                        }
                    }
                }
            }
        }
    }
}
