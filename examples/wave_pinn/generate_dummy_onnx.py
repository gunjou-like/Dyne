import torch
import torch.nn as nn
import torch.onnx
import os

# 1. モデル定義 (neuralwavesimで使用したPINNsモデル構造を再現)
# CNNベースの単純なモデルを想定しています（局所性があるため分割しやすい）
class WavePINN(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        # Conv1d: (in_channels, out_channels, kernel_size, padding)
        # kernel_size=3, padding=1 にすることで、入力と出力の空間サイズ(L)を維持します
        # これが「PDE-aware」な局所結合の最小単位です
        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        self.act1 = nn.Tanh()
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.act2 = nn.Tanh()
        self.head = nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # x shape: [Batch, Channel=1, Spatial=L]
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        out = self.head(x)
        return out

def export_onnx():
    # 2. モデルのインスタンス化
    model = WavePINN()
    model.eval()

    # 3. ダミー入力の作成
    # Batch=1, Channel=1, Length=100 (0.0~1.0の空間を100分割した想定)
    # 固定サイズでエクスポートするため、実際の使用サイズに合わせる
    dummy_input = torch.randn(1, 1, 100)

    # 4. 出力パスの設定
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "wave_pinn.onnx")

    # 5. ONNXエクスポート (Dynamic Axesなし - 完全固定サイズ)
    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,        # 学習済みパラメータを含める
        opset_version=17,          # Tanh対応のため13以上が必要 (17推奨)
        do_constant_folding=True,  # 定数畳み込み最適化
        input_names=['input'],     # 入力ノード名
        output_names=['output'],   # 出力ノード名
        # dynamic_axes を削除 - 完全に固定サイズでエクスポート
    )
    print("✅ Export completed!")
    print(f"   Input shape: {dummy_input.shape} (Fixed size)")
    print("   Output shape: [1, 1, 100] (Fixed size)")
    print("   No dynamic axes - tract compatible!")

if __name__ == "__main__":
    export_onnx()