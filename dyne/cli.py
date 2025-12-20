import click
import onnx
import os
from dyne.compiler.partitioner.simple_split import SimpleSplitter

@click.group()
def main():
    """Dyne: Physics-Aware Edge Runtime Compiler"""
    pass

@main.command()
@click.argument('input_model', type=click.Path(exists=True))
@click.option('--parts', default=2, help='Number of partitions')
@click.option('--overlap', default=2, help='Overlap size (Ghost cells)')
@click.option('--output-dir', default='.', help='Output directory')
def build(input_model, parts, overlap, output_dir):
    """
    Split ONNX model into partitions.
    Example: python -m dyne.cli build wave_pinn.onnx --parts 2
    """
    print(f"🚀 Building Dyne modules from {input_model}...")
    print(f"   Partitions: {parts}, Overlap: {overlap}")

    # 1. モデルロード
    model = onnx.load(input_model)
    onnx.checker.check_model(model)

    # 2. 分割実行
    splitter = SimpleSplitter(num_parts=parts, overlap=overlap)
    sub_models = splitter.split(model)

    # 3. 保存
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, sub_model in enumerate(sub_models):
        filename = f"part_{i}.onnx"
        out_path = os.path.join(output_dir, filename)
        onnx.save(sub_model, out_path)
        print(f"   💾 Saved: {out_path}")
    
    print("✨ Build completed successfully!")

if __name__ == '__main__':
    main()