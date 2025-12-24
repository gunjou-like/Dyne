import torch
import torch.nn as nn
import os

# 保存先
OUTPUT_PATH = "simple_model.onnx"

class DoublingModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1入力 -> 1出力 の線形層 (重み=2, バイアス=0 に固定)
        self.linear = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.linear.weight.fill_(2.0)

    def forward(self, x):
        return self.linear(x)

# モデル作成
model = DoublingModel()
model.eval()

# ダミー入力 (Batch=1, Input=1)
dummy_input = torch.tensor([[1.0]])

# エクスポート
torch.onnx.export(
    model, 
    dummy_input, 
    OUTPUT_PATH, 
    input_names=['input'], 
    output_names=['output'],
    opset_version=18
)

print(f"✅ Created: {os.path.abspath(OUTPUT_PATH)}")