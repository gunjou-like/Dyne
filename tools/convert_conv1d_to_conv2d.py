"""
Conv1DをConv2Dに変換してtractと互換性を持たせるスクリプト

使い方:
    python tools/convert_conv1d_to_conv2d.py input.onnx output.onnx
"""

import onnx
import numpy as np
import sys
from pathlib import Path


def convert_conv1d_to_conv2d(input_path: Path, output_path: Path):
    """Conv1DノードをConv2Dに変換"""
    
    print(f"Loading model: {input_path}")
    model = onnx.load(str(input_path))
    
    # 入力を1D -> 2Dに変換 (NCL -> NCHW形式に)
    for input_tensor in model.graph.input:
        shape = input_tensor.type.tensor_type.shape
        if len(shape.dim) == 3:  # [Batch, Channel, Length]
            # 新しい次元を挿入: [Batch, Channel, Height=1, Width=Length]
            new_dim = shape.dim.add()
            # 最後の次元（Length）を一時保存
            last_dim_value = shape.dim[2].dim_value
            last_dim_param = shape.dim[2].dim_param
            # Height = 1 を設定
            shape.dim[2].dim_value = 1
            shape.dim[2].ClearField('dim_param')
            # Width = 元のLength を設定
            if last_dim_value > 0:
                new_dim.dim_value = last_dim_value
            elif last_dim_param:
                new_dim.dim_param = last_dim_param
            else:
                new_dim.dim_value = 52  # デフォルト値
            print(f"  Converted input shape: {input_tensor.name}")
    
    # 出力も同様に変換
    for output_tensor in model.graph.output:
        shape = output_tensor.type.tensor_type.shape
        if len(shape.dim) == 3:
            new_dim = shape.dim.add()
            last_dim_value = shape.dim[2].dim_value
            last_dim_param = shape.dim[2].dim_param
            shape.dim[2].dim_value = 1
            shape.dim[2].ClearField('dim_param')
            if last_dim_value > 0:
                new_dim.dim_value = last_dim_value
            elif last_dim_param:
                new_dim.dim_param = last_dim_param
            else:
                new_dim.dim_value = 52
            print(f"  Converted output shape: {output_tensor.name}")
    
    # Convノードを更新
    for node in model.graph.node:
        if node.op_type == "Conv":
            print(f"  Processing Conv node: {node.name}")
            
            # kernel_shapeを1D -> 2Dに ([3] -> [1, 3])
            for attr in node.attribute:
                if attr.name == "kernel_shape" and len(attr.ints) == 1:
                    kernel_size = attr.ints[0]
                    attr.ints[:] = [1, kernel_size]
                    print(f"    kernel_shape: [1, {kernel_size}]")
                
                elif attr.name == "pads" and len(attr.ints) == 2:
                    # [left, right] -> [top, left, bottom, right]
                    pad_left = attr.ints[0]
                    pad_right = attr.ints[1]
                    attr.ints[:] = [0, pad_left, 0, pad_right]
                    print(f"    pads: [0, {pad_left}, 0, {pad_right}]")
                
                elif attr.name == "strides" and len(attr.ints) == 1:
                    stride = attr.ints[0]
                    attr.ints[:] = [1, stride]
                    print(f"    strides: [1, {stride}]")
                
                elif attr.name == "dilations" and len(attr.ints) == 1:
                    dilation = attr.ints[0]
                    attr.ints[:] = [1, dilation]
                    print(f"    dilations: [1, {dilation}]")
    
    # Initializerの重みを1D -> 2Dに変換
    for init in model.graph.initializer:
        if "weight" in init.name and len(init.dims) == 3:
            # [OutChannels, InChannels, KernelSize] -> [OutChannels, InChannels, 1, KernelSize]
            out_ch, in_ch, kernel = init.dims
            init.dims[:] = [out_ch, in_ch, 1, kernel]
            print(f"  Converted weight: {init.name} -> [{out_ch}, {in_ch}, 1, {kernel}]")
    
    # 保存
    print(f"\nSaving converted model: {output_path}")
    onnx.save(model, str(output_path))
    
    # 検証
    try:
        onnx.checker.check_model(model)
        print("✓ Converted model is valid")
    except Exception as e:
        print(f"⚠️  Validation warning: {e}")
    
    print("✓ Conversion complete!")


def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_conv1d_to_conv2d.py <input.onnx> <output.onnx>")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    convert_conv1d_to_conv2d(input_path, output_path)


if __name__ == "__main__":
    main()
