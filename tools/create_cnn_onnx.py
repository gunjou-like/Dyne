import torch
import torch.nn as nn
import os

OUTPUT_PATH = "simple_cnn.onnx"

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 入力1ch -> 出力1ch, 3x3カーネル, パディング1 (サイズを変えない)
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        
        # 重みを固定: ラプラシアンフィルタ的なもの (エッジ検出)
        # [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        with torch.no_grad():
            kernel = torch.tensor([[[[0., 1., 0.],
                                    [1., -4., 1.],
                                    [0., 1., 0.]]]])
            self.conv.weight.copy_(kernel)

    def forward(self, x):
        return self.conv(x)

# モデル作成
model = SimpleCNN()
model.eval()

# ダミー入力 (Batch=1, Ch=1, H=5, W=5)
# 画像サイズを明示的に指定しないと、ONNX側で動的サイズになり解析が難しくなるため
dummy_input = torch.randn(1, 1, 5, 5)

# エクスポート
torch.onnx.export(
    model, 
    dummy_input, 
    OUTPUT_PATH, 
    input_names=['input'], 
    output_names=['output'],
    opset_version=17
)

print(f"✅ Created CNN Model: {os.path.abspath(OUTPUT_PATH)}")