"""
ONNXモデルの構造とConv1Dパラメータを確認するテストスクリプト

使い方:
    python tests/test_onnx_model.py
"""

import onnx
import numpy as np
from pathlib import Path


def analyze_onnx_model(model_path: Path):
    """ONNXモデルの構造を解析して表示"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_path}")
    print(f"{'='*60}\n")
    
    try:
        model = onnx.load(str(model_path))
        
        # 基本情報
        print(f"IR Version: {model.ir_version}")
        print(f"Producer: {model.producer_name} {model.producer_version}")
        print(f"OpSet Version: {model.opset_import[0].version if model.opset_import else 'N/A'}")
        
        # 入力情報
        print(f"\n--- Inputs ---")
        for input_tensor in model.graph.input:
            shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param 
                    for dim in input_tensor.type.tensor_type.shape.dim]
            print(f"  {input_tensor.name}: {shape} ({input_tensor.type.tensor_type.elem_type})")
        
        # 出力情報
        print(f"\n--- Outputs ---")
        for output_tensor in model.graph.output:
            shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param 
                    for dim in output_tensor.type.tensor_type.shape.dim]
            print(f"  {output_tensor.name}: {shape} ({output_tensor.type.tensor_type.elem_type})")
        
        # ノード情報（特にConv関連）
        print(f"\n--- Nodes ({len(model.graph.node)} total) ---")
        conv_nodes = []
        for i, node in enumerate(model.graph.node):
            print(f"  [{i}] {node.op_type}: {node.name or 'unnamed'}")
            print(f"      Inputs: {list(node.input)}")
            print(f"      Outputs: {list(node.output)}")
            
            if 'Conv' in node.op_type:
                conv_nodes.append(node)
                print(f"      *** Conv Node Detected ***")
                # 属性を詳細表示
                for attr in node.attribute:
                    if attr.ints:
                        print(f"          {attr.name}: {list(attr.ints)}")
                    elif attr.i:
                        print(f"          {attr.name}: {attr.i}")
                    elif attr.s:
                        print(f"          {attr.name}: {attr.s.decode('utf-8')}")
                    elif attr.f:
                        print(f"          {attr.name}: {attr.f}")
        
        # Initializer（重みパラメータ）
        print(f"\n--- Initializers ({len(model.graph.initializer)} total) ---")
        for init in model.graph.initializer:
            shape = list(init.dims)
            print(f"  {init.name}: {shape} ({init.data_type})")
        
        # Conv ノードの詳細解析
        if conv_nodes:
            print(f"\n--- Conv Node Details ---")
            for conv in conv_nodes:
                print(f"\n  Node: {conv.name or 'unnamed'} ({conv.op_type})")
                
                # 属性の詳細
                attrs = {attr.name: attr for attr in conv.attribute}
                
                if 'kernel_shape' in attrs:
                    print(f"    kernel_shape: {list(attrs['kernel_shape'].ints)}")
                if 'strides' in attrs:
                    print(f"    strides: {list(attrs['strides'].ints)}")
                if 'pads' in attrs:
                    print(f"    pads: {list(attrs['pads'].ints)}")
                if 'dilations' in attrs:
                    print(f"    dilations: {list(attrs['dilations'].ints)}")
                if 'group' in attrs:
                    print(f"    group: {attrs['group'].i}")
                if 'auto_pad' in attrs:
                    print(f"    auto_pad: {attrs['auto_pad'].s.decode('utf-8')}")
                
                # 入力テンソルの形状を推定
                print(f"    Inputs: {list(conv.input)}")
                print(f"    Outputs: {list(conv.output)}")
        
        # モデルの検証
        print(f"\n--- Validation ---")
        try:
            onnx.checker.check_model(model)
            print("  ✓ Model is valid")
        except Exception as e:
            print(f"  ✗ Validation failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メイン処理"""
    # テスト対象のモデル
    models_to_test = [
        Path("wasm/dist/part_0.onnx"),
        Path("wasm/dist/part_1.onnx"),
        Path("examples/wave_pinn/wave_pinn.onnx"),
    ]
    
    results = {}
    for model_path in models_to_test:
        if model_path.exists():
            results[str(model_path)] = analyze_onnx_model(model_path)
        else:
            print(f"\n⚠️  File not found: {model_path}")
            results[str(model_path)] = False
    
    # サマリー
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for path, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {path}")


if __name__ == "__main__":
    main()
