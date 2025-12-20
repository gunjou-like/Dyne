import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time

# --- 1. モデル定義 (Runtimeと同じ構造) ---
class WavePINN(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        # Padding=1 でサイズを変えない
        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        self.act1 = nn.Tanh()
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.act2 = nn.Tanh()
        self.head = nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        return self.head(x)

# --- 2. データ生成 (1次元波動方程式の解) ---
# 目標: u_tt = c^2 * u_xx を学習させたいが、
# MVPでは「単純な移動（Advection）」を学習させる方が簡単で、見た目も分かりやすいです。
# u(t+1, x) = u(t, x - c)  (右にずれるだけ)

def generate_training_data(batch_size=1000, seq_len=52):
    # ランダムな波形を作る
    x = np.linspace(0, 10, seq_len)
    inputs = []
    targets = []
    
    for _ in range(batch_size):
        # ガウスパルスをランダムな位置に置く
        center = np.random.uniform(2, 8)
        width = np.random.uniform(0.5, 1.5)
        wave = np.exp(-(x - center)**2 / (2 * width**2))
        
        # 入力: 現在の波 u(t)
        inputs.append(wave)
        
        # 正解: 少し右にずれた波 u(t+1)
        # indexでいうと 1つ右へシフト (左端は0埋め)
        shifted_wave = np.roll(wave, 1)
        shifted_wave[0] = 0 
        targets.append(shifted_wave)
        
    return torch.tensor(inputs, dtype=torch.float32).unsqueeze(1), \
           torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

# --- 3. 学習ループ ---
def train_and_export():
    print("🚀 Training started...")
    model = WavePINN(hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    inputs, targets = generate_training_data(2000, 52)
    
    # 500 Epoch学習
    for epoch in range(500):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

    print("✅ Training finished.")

    # --- 4. 重みのエクスポート (JSON) ---
    print("📦 Exporting weights...")
    weights = {
        "conv1_w": model.conv1.weight.detach().numpy().flatten().tolist(),
        "conv1_b": model.conv1.bias.detach().numpy().flatten().tolist(),
        "conv2_w": model.conv2.weight.detach().numpy().flatten().tolist(),
        "conv2_b": model.conv2.bias.detach().numpy().flatten().tolist(),
        "head_w":  model.head.weight.detach().numpy().flatten().tolist(),
        "head_b":  model.head.bias.detach().numpy().flatten().tolist(),
        "hidden_dim": 64
    }

    with open("wave_weights.json", "w") as f:
        json.dump(weights, f)
    print("✅ wave_weights.json saved.")

if __name__ == "__main__":
    train_and_export()