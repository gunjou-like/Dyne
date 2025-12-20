# Dyne Tests

このディレクトリには、Dyneプロジェクトのテストスクリプトが含まれています。

## Python テスト

### test_onnx_model.py

ONNXモデルの構造とConv1Dパラメータを確認するスクリプト

**実行方法:**
```bash
python tests/test_onnx_model.py
```

**確認項目:**
- モデルの基本情報（IR version, OpSet version など）
- 入力/出力テンソルの形状
- ノードのリストと詳細（特にConv関連）
- Initializer（重みパラメータ）の情報
- モデルの妥当性検証

## Rust テスト

### wasm/tests/test_optimization.rs

tract による最適化の問題を診断するテストスイート

**実行方法:**
```bash
cd wasm
cargo test --release -- --nocapture
```

**テスト内容:**

1. **test_model_loading_without_optimization**
   - 最適化なしでモデルを読み込み
   - 基本的なモデル情報を表示
   - 実行可能プランの作成を確認

2. **test_model_loading_with_optimization**
   - 最適化ありでモデルを読み込み
   - ConvHir での最適化エラーを診断
   - エラーメッセージの詳細を出力

3. **test_tract_version_and_conv_support**
   - tract のバージョン情報を表示
   - Conv1D のサポート状況を確認
   - モデル内のConvノードを検索

4. **test_simple_inference_without_optimization**
   - 最適化なしでの推論テスト
   - ダミーデータで実際に推論を実行
   - 出力の形状を確認

## トラブルシューティング

### Python テストでエラーが出る場合

ONNXパッケージをインストール:
```bash
pip install onnx numpy
```

### Rust テストでファイルが見つからない場合

`wasm` ディレクトリから実行してください:
```bash
cd wasm
cargo test --release -- --nocapture
```

### モデルファイルが存在しない場合

以下のコマンドでモデルを分割してください:
```bash
python -m dyne.compiler.partitioner.simple_split examples/wave_pinn/wave_pinn.onnx --output-dir wasm/dist
```
