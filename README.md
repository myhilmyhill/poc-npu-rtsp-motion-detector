# NPU Object Detection Demo with ONNX Runtime QNN

Qualcomm NPU（Neural Processing Unit）を使用したリアルタイム物体検知のシンプルなデモプロジェクトです。ONNX Runtime QNN Execution Providerを活用して、YOLOv8モデルでの高速推論を実現します。

## 🎯 特徴

- 🚀 **NPUアクセラレーション**: ONNX Runtime QNN Execution Providerによる真のNPU活用
- 🎥 **多様な入力対応**: Webカメラ、画像、動画、RTSPストリーム
- 🎯 **YOLOv8物体検知**: リアルタイムで80クラスのCOCO物体を検知
- ⚡ **高速推論**: NPU使用時は約12ms/フレーム
- 🔧 **シンプル構成**: PowerShellスクリプト (`scripts/simple_demo.ps1`) による環境構築と推論の自動化

## 📋 システム要件

- **OS**: Windows 10/11
- **Python**: 3.10 (venvで仮想環境を作成)
- **ハードウェア**: Qualcomm NPU搭載デバイス（例: Snapdragon X Elite/Plus）
- **SDK**: Qualcomm AI Engine Direct (QAIRT) SDK
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

### 2. モデルと環境変数の準備

- `models/yolov8n.onnx` が存在することを確認してください。
- （RTSPストリームを使用する場合）`.env.example` をコピーして `.env` を作成し、接続情報を設定してください。

### 3. スクリプトのパス設定

`scripts/simple_demo.ps1` をエディタで開き、お使いの環境に合わせて1行目のQAIRT SDKパスを修正してください。

```powershell
# 例
pushd C:\path\to\your\qairt\2.44.0.260225\bin
```

### 4. デモの実行

環境のセットアップからデモの実行まで、すべて1つのスクリプトで行います。

```powershell
.\scripts\simple_demo.ps1
```

※ このスクリプトは以下の処理を自動で行います：
1. QAIRT SDKの環境変数セットアップ (`envsetup.ps1` 等の実行)
2. Python 3.10での仮想環境 (`venv`) 作成と有効化
3. SDK依存パッケージのインストールと環境チェック
4. QNN環境の動作確認 (`check_qnn.py` の実行)
5. メインデモ (`simple_demo.py`) の起動

_注意: OpenCVなどのPythonパッケージが不足しているとエラーが出る場合は、仮想環境内で `pip install opencv-python numpy python-dotenv` を実行してください。_

## 📁 プロジェクト構成

```
poc-npu-rtsp-motion-detector/
├── simple_demo.py          # メインのデモプログラム
├── check_qnn.py            # QNN環境確認スクリプト
├── scripts/
│   └── simple_demo.ps1     # 環境構築・実行用統合スクリプト
├── models/
│   └── yolov8n.onnx        # YOLOv8 Nanoモデル（ONNX形式）
├── .env.example            # 環境変数のサンプル
├── .env                    # 環境変数（RTSPの設定など）
└── docs/                   # ドキュメント
```

## 🎮 使い方

### 物体検知デモ (simple_demo.py)

`.\scripts\simple_demo.ps1` 実行時に自動で起動します。起動後、ターミナルで入力ソースを選択します：

1. **Webカメラ** (デフォルト) - PCのカメラから入力
2. **画像ファイル** - 静止画の推論
3. **動画ファイル** - 動画ファイルの処理
4. **RTSPストリーム** - ネットワークカメラ（`.env`設定必要）

**操作:**
- `q`キーを押して画面を終了

**画面表示:**
- 緑の枠: 検知された物体
- ラベル: クラス名と信頼度
- FPS: フレームレート
- デバイス: NPU (QNN) または CPU

### RTSPストリームの設定

`.env`ファイルを作成（`.env.example`を参考に記述）：

```env
RTSP_URL=rtsp://192.168.1.100:554/stream1
RTSP_USERNAME=admin
RTSP_PASSWORD=yourpassword
```

- パスワードに特殊文字（`:`など）が含まれていても自動でURLエンコードして接続します。

## ⚙️ 技術詳細

### ONNX Runtime QNN

このプロジェクトは `onnxruntime-qnn` パッケージを使用し、Qualcomm QNN (Qualcomm Neural Network) SDKを通じてNPUにアクセスします。

**QNNバックエンドの設定例:**
```python
qnn_options = {
    'backend_path': 'QnnHtp.dll',           # HTP (Hexagon Tensor Processor) バックエンド
    'qnn_context_priority': 'high',         # 優先度
    'profiling_level': 'basic'              # プロファイリングレベル
}
```

## 🔧 トラブルシューティング

### QNNExecutionProviderが利用できない

スクリプト実行時に **"⚠️ QNNが無効です。CPUで実行されます。"** と表示される場合は、QNNプロバイダが存在しないか、依存関係の解決に失敗しています。`scripts/simple_demo.ps1` 内に記載されている以下のコマンドを手動で実行（またはコメントアウトを解除）して再インストールを試してください。

```powershell
pip uninstall onnxruntime
pip install onnxruntime-qnn
```

### 実行スクリプト(.ps1)がエラーになる場合
- 実行ポリシーによりPowerShellスクリプトがブロックされている場合は、管理者権限で `Set-ExecutionPolicy RemoteSigned` を実行してください。

### RTSPストリームに接続できない
1. URLを確認（`.env`ファイル）
2. ネットワーク接続を確認
3. 認証情報（ユーザー名/パスワード）を確認

### 📚 ドキュメント
詳細は `docs/` ディレクトリ内のドキュメントを参照してください。
