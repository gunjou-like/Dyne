"""
固定サイズONNXモデルを空間分割するスクリプト

使い方:
    python tools/split_fixed_onnx.py examples/wave_pinn/wave_pinn.onnx --output-dir wasm/dist --num-partitions 2
"""

import onnx
import copy
import argparse
from pathlib import Path


def split_fixed_onnx(model_path: Path, output_dir: Path, num_partitions: int = 2, overlap: int = 2):
    """固定サイズのONNXモデルを空間分割"""
    
    print(f"Loading model: {model_path}")
    model = onnx.load(str(model_path))
    
    # 入力シェイプを取得
    input_shape = model.graph.input[0].type.tensor_type.shape
    total_width = input_shape.dim[2].dim_value
    
    print(f"Total width: {total_width}")
    print(f"Splitting into {num_partitions} parts with overlap {overlap}...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    base_width = total_width // num_partitions
    
    for i in range(num_partitions):
        part_model = copy.deepcopy(model)
        
        # パーティションの範囲を計算
        start_idx = max(0, (i * base_width) - overlap)
        end_idx = min(total_width, ((i + 1) * base_width) + overlap)
        part_width = end_idx - start_idx
        
        print(f"\nPart {i}:")
        print(f"  Grid range: [{start_idx}:{end_idx}]")
        print(f"  Width: {part_width}")
        
        # 入力シェイプを更新
        part_model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = part_width
        
        # 出力シェイプを更新
        part_model.graph.output[0].type.tensor_type.shape.dim[2].dim_value = part_width
        
        # 保存
        output_path = output_dir / f"part_{i}.onnx"
        onnx.save(part_model, str(output_path))
        print(f"  Saved: {output_path}")
        
        # 検証
        try:
            onnx.checker.check_model(part_model)
            print(f"  ✓ Valid")
        except Exception as e:
            print(f"  ⚠️  Validation warning: {e}")
    
    print(f"\n✓ Split complete! {num_partitions} models created in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Split fixed-size ONNX model spatially")
    parser.add_argument("model_path", type=Path, help="Path to input ONNX model")
    parser.add_argument("--output-dir", type=Path, default=Path("wasm/dist"), help="Output directory")
    parser.add_argument("--num-partitions", type=int, default=2, help="Number of partitions")
    parser.add_argument("--overlap", type=int, default=2, help="Overlap cells")
    
    args = parser.parse_args()
    
    if not args.model_path.exists():
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    split_fixed_onnx(args.model_path, args.output_dir, args.num_partitions, args.overlap)


if __name__ == "__main__":
    main()
