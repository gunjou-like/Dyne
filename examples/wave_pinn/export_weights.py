import torch
import torch.nn as nn
import json
import os

# 1. モデル定義 (学習時と同じ構造)
class WavePINN(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        self.act1 = nn.Tanh()
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.act2 = nn.Tanh()
        self.head = nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        return self.head(x)

def export_weights():
    # モデルインスタンス作成 (学習済み重みがある場合はここでload_state_dictする)
    # 今回はデモ用にランダム初期化のままでOK
    model = WavePINN()
    model.eval()

    print("Extracting weights...")
    
    # Rustで読みやすい辞書形式に変換
    weights = {
        # flatten()で1次元配列にしてリスト化
        "conv1_w": model.conv1.weight.detach().numpy().flatten().tolist(),
        "conv1_b": model.conv1.bias.detach().numpy().flatten().tolist(),
        "conv2_w": model.conv2.weight.detach().numpy().flatten().tolist(),
        "conv2_b": model.conv2.bias.detach().numpy().flatten().tolist(),
        "head_w":  model.head.weight.detach().numpy().flatten().tolist(),
        "head_b":  model.head.bias.detach().numpy().flatten().tolist(),
        
        "hidden_dim": 64
    }

    # 保存
    output_path = "wave_weights.json"
    with open(output_path, "w") as f:
        json.dump(weights, f)
    
    print(f"✅ Export success: {output_path}")

if __name__ == "__main__":
    export_weights()