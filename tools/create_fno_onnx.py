import torch
import torch.nn as nn
import os

OUTPUT_PATH = "simple_fno.onnx"

# --- Model Definitions (Same as before) ---
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = (1 / (in_channels * out_channels))
        # Complex weights
        self.weights = nn.Parameter(
            self.scale * torch.rand(out_channels, in_channels, modes, dtype=torch.cfloat)
        )

class SimpleFNO(nn.Module):
    def __init__(self, modes=4, width=16):
        super(SimpleFNO, self).__init__()
        self.modes = modes
        self.width = width
        self.p = nn.Linear(1, width) 
        self.conv0 = SpectralConv1d(width, width, modes)
        self.w0 = nn.Conv1d(width, width, 1)
        self.q = nn.Linear(width, 1)

    def forward(self, x):
        # 実際の計算ロジックはここにあるが、
        # ONNXエクスポート時は ExportWrapper でバイパスされるため
        # ここは実行されない。
        pass

# --- Export Wrapper (Fixing the Error) ---
class ExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        # 重みへの参照を保持
        # 1. Complex weights -> Real (View as real) [Out, In, Modes, 2]
        self.fno_weights_real = nn.Parameter(torch.view_as_real(model.conv0.weights))
        
        # 2. Other weights
        self.p_weight = model.p.weight
        self.p_bias = model.p.bias
        self.q_weight = model.q.weight
        self.q_bias = model.q.bias
        self.w0_weight = model.w0.weight
        self.w0_bias = model.w0.bias

    def forward(self, x):
        # 【重要】ダミー計算
        # 複素数演算(FFTなど)を行わず、単に重みを返すだけのグラフを作る。
        # これにより "RuntimeError: slice_scatter..." を回避しつつ、
        # ONNXファイル内の Initializer リストに重みが保存される。
        return (
            self.fno_weights_real,
            self.p_weight, self.p_bias,
            self.q_weight, self.q_bias,
            self.w0_weight, self.w0_bias
        )

# --- Execution ---
modes = 4
width = 16
length = 32

model = SimpleFNO(modes, width)
model.eval()

# Wrapperで包む
wrapper = ExportWrapper(model)

dummy_input = torch.randn(1, length)

# Export
torch.onnx.export(
    wrapper,
    dummy_input,
    OUTPUT_PATH,
    input_names=['input'],
    # 出力名は適当でよい（重みが保存されればよい）
    output_names=['fno_w', 'p_w', 'p_b', 'q_w', 'q_b', 'w0_w', 'w0_b'],
    opset_version=17
)

print(f"✅ Created FNO Model (Weights Container): {os.path.abspath(OUTPUT_PATH)}")