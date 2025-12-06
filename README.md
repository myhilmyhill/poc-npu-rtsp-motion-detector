# NPU Object Detection Demo with ONNX Runtime QNN

Qualcomm NPU（Neural Processing Unit）を使用したリアルタイム物体検知のシンプルなデモプロジェクトです。ONNX Runtime QNN Execution Providerを活用して、YOLOv8モデルでの高速推論を実現します。

## 🎯 特徴

- 🚀 **NPUアクセラレーション**: ONNX Runtime QNN Execution Providerによる真のNPU活用
- 🎥 **多様な入力対応**: Webカメラ、画像、動画、RTSPストリーム
- 🎯 **YOLOv8物体検知**: リアルタイムで80クラスのCOCO物体を検知
- ⚡ **高速推論**: NPU使用時は約12ms/フレーム
- 🔧 **シンプル構成**: 3つのファイルのみで動作

## 📋 システム要件

- **OS**: Windows 10/11
- **Python**: 3.10以上
- **ハードウェア**: Qualcomm NPU搭載デバイス（例: Snapdragon X Elite/Plus）
- **必要パッケージ**:
  - `onnxruntime-qnn` (NPU用)
  - `opencv-python`
  - `numpy`
  - `python-dotenv`

## 🚀 クイックスタート

### 1. リポジトリのクローン

```powershell
git clone <repository-url>
cd poc-npu-rtsp-motion-detector
```

### 2. 必要なパッケージをインストール

```powershell
pip install onnxruntime-qnn opencv-python numpy python-dotenv
```

### 3. モデルの確認

`models/yolov8n.onnx` が存在することを確認してください。

### 4. QNN環境の確認

```powershell
python check_qnn.py
```

✅ "QNN Execution Provider is available!" と表示されればOKです。

### 5. デモの実行

```powershell
python simple_demo.py
```

## 📁 プロジェクト構成

```
poc-npu-rtsp-motion-detector/
├── simple_demo.py          # メインのデモプログラム
├── check_qnn.py            # QNN環境確認スクリプト
├── scripts/
│   └── debug_snpe.ps1      # デバッグ用スクリプト
├── models/
│   └── yolov8n.onnx        # YOLOv8 Nanoモデル（ONNX形式）
├── .env.example            # 環境変数のサンプル
├── .env                    # 環境変数（RTSPの設定など）
└── docs/                   # ドキュメント
```

## 🎮 使い方

### check_qnn.py - QNN環境確認

QNN Execution Providerが利用可能か確認します：

```powershell
python check_qnn.py
```

**出力例:**
```
============================================================
ONNX Runtime QNN Status Check
============================================================

📦 ONNX Runtime Location:
   C:\...\site-packages\onnxruntime\__init__.py

🔌 Available Execution Providers:
   🚀 QNNExecutionProvider
      CPUExecutionProvider

✅ SUCCESS: QNN Execution Provider is available!
   Your NPU hardware acceleration is ready to use.
```

### simple_demo.py - 物体検知デモ

リアルタイム物体検知を実行します：

```powershell
python simple_demo.py
```

起動後、入力ソースを選択：

1. **Webカメラ** (デフォルト) - PCのカメラから入力
2. **画像ファイル** - 静止画の推論
3. **動画ファイル** - 動画ファイルの処理
4. **RTSPストリーム** - ネットワークカメラ（`.env`設定必要）

**操作:**
- `q`キーを押して終了

**画面表示:**
- 緑の枠: 検知された物体
- ラベル: クラス名と信頼度
- FPS: フレームレート
- デバイス: NPU (QNN) または CPU

### RTSPストリームの設定

`.env`ファイルを作成（`.env.example`を参考に）：

```env
RTSP_URL=rtsp://192.168.1.100:554/stream1
RTSP_USERNAME=admin
RTSP_PASSWORD=yourpassword
```

- パスワードに特殊文字（`:`など）が含まれていてもOK（自動でURLエンコード）

### scripts/debug_snpe.ps1 - デバッグ用

SNPE環境のデバッグ用PowerShellスクリプトです。

```powershell
.\scripts\debug_snpe.ps1
```

## ⚙️ 技術詳細

### ONNX Runtime QNN

このプロジェクトは `onnxruntime-qnn` パッケージを使用し、Qualcomm QNN (Qualcomm Neural Network) SDKを通じてNPUにアクセスします。

**QNNオプション:**
```python
qnn_options = {
    'backend_path': 'QnnHtp.dll',           # HTP (Hexagon Tensor Processor) バックエンド
    'qnn_context_priority': 'high',         # 優先度
    'profiling_level': 'basic'              # プロファイリングレベル
}
```

### YOLOv8の前処理・後処理

- **入力**: 640x640 RGB画像（正規化済み、0-1）
- **出力**: (1, 84, 8400) - 8400個の検出候補
- **前処理**: アスペクト比を保持したリサイズ + パディング
- **後処理**: 信頼度フィルタリング + NMS (Non-Maximum Suppression)

### 検知可能なクラス

COCO 80クラス（person, car, dog, cat, bicycle, など）

## 📊 パフォーマンス

| デバイス | FPS | 推論時間 | 備考 |
|---------|-----|----------|------|
| NPU (QNN) | ~28-30 | ~12ms | Qualcomm HTP使用 |
| CPU | ~8-10 | ~100-120ms | フォールバック |

## 🔧 トラブルシューティング

### QNNExecutionProviderが利用できない

**症状:**
```
⚠️ QNNが無効です。CPUで実行されます。
```

**解決方法:**
1. `onnxruntime-qnn`がインストールされているか確認:
   ```powershell
   pip list | grep onnxruntime
   ```

2. 正しいバージョンを再インストール:
   ```powershell
   pip uninstall onnxruntime onnxruntime-qnn
   pip install onnxruntime-qnn
   ```

3. NPU搭載デバイスか確認

### RTSPストリームに接続できない

1. URLを確認（`.env`ファイル）
2. ネットワーク接続を確認
3. 認証情報（ユーザー名/パスワード）を確認
4. カメラのRTSPポートが開いているか確認

### モデルが見つからない

```powershell
# modelsディレクトリを確認
ls models/
```

`yolov8n.onnx`が存在しない場合は、別途用意する必要があります。

## 📚 ドキュメント

詳細は`docs/`ディレクトリ内のドキュメントを参照：

- [クイックスタート](docs/クイックスタート.md) - 最初のセットアップ
- [QNN環境設定](docs/QNN環境設定.md) - ONNX Runtime QNNの詳細設定
- [使い方](docs/使い方.md) - 詳細な使用方法

## 🌟 主な変更点

このプロジェクトは以前の複雑な構成から大幅に簡素化されました：

- ✅ 3つのファイルのみで動作
- ✅ ONNX Runtime QNN を直接使用（SNPEから移行）
- ✅ 複雑な依存関係を排除
- ✅ シンプルで理解しやすいコード

## 📝 ライセンス

MIT License

## 🙏 貢献

プルリクエストを歓迎します！
