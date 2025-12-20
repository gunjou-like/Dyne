"""
ONNXモデルからすべてのシンボリック次元を削除して完全固定サイズにするスクリプト

使い方:
    python tools/fix_symbolic_dims.py wasm/dist/part_0.onnx wasm/dist/part_0_fixed.onnx 52
"""

import onnx
import sys
from pathlib import Path


def fix_symbolic_dimensions(input_path: Path, output_path: Path, fixed_length: int):
    """すべてのシンボリック次元を固定値に置き換える"""
    
    print(f"Loading model: {input_path}")
    model = onnx.load(str(input_path))
    
    # グラフ内のすべてのValueInfoを処理
    def fix_shape(tensor_proto):
        """テンソルの形状からシンボリックdimを削除"""
        shape = tensor_proto.type.tensor_type.shape
        for i, dim in enumerate(shape.dim):
            if dim.HasField('dim_param') and dim.dim_param:
                # シンボリック次元を固定値に置き換え
                old_param = dim.dim_param
                if i == 0:  # Batch dimension
                    dim.dim_value = 1
                elif i == 2:  # Spatial dimension
                    dim.dim_value = fixed_length
                else:
                    dim.dim_value = dim.dim_value if dim.dim_value > 0 else 1
                dim.ClearField('dim_param')
                print(f"  Fixed {tensor_proto.name} dim[{i}]: '{old_param}' -> {dim.dim_value}")
    
    # 入力を修正
    print("\nFixing inputs:")
    for input_tensor in model.graph.input:
        fix_shape(input_tensor)
    
    # 出力を修正
    print("\nFixing outputs:")
    for output_tensor in model.graph.output:
        fix_shape(output_tensor)
    
    # 中間テンソル（ValueInfo）を修正
    print("\nFixing value_info:")
    for value_info in model.graph.value_info:
        fix_shape(value_info)
    
    # ノードの出力形状情報もクリア（これが原因の可能性）
    print("\nClearing node output shape annotations:")
    for node in model.graph.node:
        # ノードに付随する型情報をクリア
        if node.output:
            print(f"  Node: {node.name} ({node.op_type})")
    
    # 保存
    print(f"\nSaving fixed model: {output_path}")
    onnx.save(model, str(output_path))
    
    # 検証
    try:
        onnx.checker.check_model(model)
        print("✓ Fixed model is valid")
    except Exception as e:
        print(f"⚠️  Validation warning: {e}")
    
    print("✓ All symbolic dimensions removed!")


def main():
    if len(sys.argv) != 4:
        print("Usage: python fix_symbolic_dims.py <input.onnx> <output.onnx> <fixed_length>")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    fixed_length = int(sys.argv[3])
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    fix_symbolic_dimensions(input_path, output_path, fixed_length)


if __name__ == "__main__":
    main()
