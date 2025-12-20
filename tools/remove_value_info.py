"""
ONNX モデルから value_info (中間テンソル情報) を完全削除するスクリプト

tractがシンボリック次元を作成する原因となるvalue_infoをクリアします

使い方:
    python tools/remove_value_info.py wasm/dist/part_0.onnx wasm/dist/part_0_clean.onnx
"""

import onnx
import sys
from pathlib import Path


def remove_value_info(input_path: Path, output_path: Path):
    """value_infoをクリアして推論エンジンに形状を再計算させる"""
    
    print(f"Loading model: {input_path}")
    model = onnx.load(str(input_path))
    
    #  value_info をクリア
    print(f"Removing {len(model.graph.value_info)} value_info entries...")
    model.graph.ClearField('value_info')
    
    # 出力の形状情報もクリア（tractがここからシンボリックを作成する）
    print(f"\nClearing output shape info...")
    for output in model.graph.output:
        print(f"  Output: {output.name}")
        # 型情報は残すが、形状情報をクリア
        if output.HasField('type') and output.type.HasField('tensor_type'):
            output.type.tensor_type.ClearField('shape')
    
    print("\nSaving cleaned model...")
    onnx.save(model, str(output_path))
    
    # 検証（形状情報がないので警告が出る可能性がある）
    try:
        onnx.checker.check_model(model)
        print("✓ Cleaned model is valid")
    except Exception as e:
        print(f"⚠️  Validation warning (expected): {e}")
    
    print("✓ All shape hints cleared - tract will infer from inputs only!")


def main():
    if len(sys.argv) != 3:
        print("Usage: python remove_value_info.py <input.onnx> <output.onnx>")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    remove_value_info(input_path, output_path)


if __name__ == "__main__":
    main()
