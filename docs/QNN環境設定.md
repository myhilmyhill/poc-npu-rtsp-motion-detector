# QNN環境設定ガイド

ONNX Runtime QNN Execution Providerの詳細設定とトラブルシューティングガイドです。

## 📚 目次

1. [QNNとは](#qnnとは)
2. [インストール方法](#インストール方法)
3. [環境確認](#環境確認)
4. [QNNオプション詳細](#qnnオプション詳細)
5. [トラブルシューティング](#トラブルシューティング)
6. [パフォーマンスチューニング](#パフォーマンスチューニング)

---

## QNNとは

### Qualcomm Neural Network (QNN) SDK

QNNは、Qualcomm製NPU（Hexagon Tensor Processor）を活用するための開発キットです。

**主な特徴:**
- ✅ NPU（HTP）による高速推論
- ✅ 低消費電力
- ✅ ONNX Runtime統合

### ONNX Runtime QNN Execution Provider

ONNX RuntimeのQNN Execution Providerは、ONNXモデルをQNN経由でNPU上で実行します。

**動作フロー:**
```
ONNXモデル → ONNX Runtime → QNN EP → QNN SDK → NPU (HTP)
```

---

## インストール方法

### 前提条件

- Windows 10/11
- Python 3.10以上
- Qualcomm NPU搭載デバイス（Snapdragon X Elite/Plus）

### 標準インストール

```powershell
pip install onnxruntime-qnn
```

**重要:** `onnxruntime` と `onnxruntime-qnn` は別パッケージです。両方インストールすると競合する場合があります。

### クリーンインストール

既存の`onnxruntime`がある場合、クリーンインストールを推奨：

```powershell
# 既存パッケージを削除
pip uninstall onnxruntime onnxruntime-gpu onnxruntime-directml onnxruntime-qnn

# QNN版をインストール
pip install onnxruntime-qnn
```

### バージョン確認

```powershell
pip show onnxruntime-qnn
```

**出力例:**
```
Name: onnxruntime-qnn
Version: 1.17.0
Location: C:\...\site-packages
Requires: coloredlogs, flatbuffers, numpy, packaging, protobuf, sympy
```

### 依存パッケージ

```powershell
pip install opencv-python numpy python-dotenv
```

---

## 環境確認

### check_qnn.py による確認

```powershell
python check_qnn.py
```

#### 正常な出力

```
============================================================
ONNX Runtime QNN Status Check
============================================================

📦 ONNX Runtime Location:
   C:\Users\...\site-packages\onnxruntime\__init__.py

🔌 Available Execution Providers:
   🚀 QNNExecutionProvider              ← これが重要
      AzureExecutionProvider
      CPUExecutionProvider

✅ SUCCESS: QNN Execution Provider is available!
   Your NPU hardware acceleration is ready to use.

🧪 Testing QNN Session Creation...
Starting stage: Graph Preparation Initializing
Completed stage: Graph Preparation Initializing (454 us)
Starting stage: Graph Transformations and Optimizations
Completed stage: Graph Transformations and Optimizations (831599 us)
Starting stage: Graph Sequencing for Target
 [##################################################] 100%
Completed stage: Graph Sequencing for Target (873212 us)
...
   Active Providers: ['QNNExecutionProvider', 'CPUExecutionProvider']
   ✅ QNN session created successfully!

============================================================
```

**確認ポイント:**
- ✅ `QNNExecutionProvider` が Available Providers に含まれる
- ✅ セッション作成が成功する
- ✅ Graph Sequencing などのステージが表示される（QNNが動作している証拠）

#### 異常な出力

```
🔌 Available Execution Providers:
      CPUExecutionProvider              ← QNNが無い

❌ WARNING: QNN Execution Provider is NOT available.
   Inference will fall back to CPU.
```

**対策:** [トラブルシューティング](#トラブルシューティング) を参照

### Python による手動確認

```python
import onnxruntime as ort

print(f"Version: {ort.__version__}")
print(f"Providers: {ort.get_available_providers()}")

if 'QNNExecutionProvider' in ort.get_available_providers():
    print("✅ QNN is available")
else:
    print("❌ QNN is NOT available")
```

---

## QNNオプション詳細

### 基本オプション

`simple_demo.py` で使用されているQNNオプション:

```python
qnn_options = {
    'backend_path': 'QnnHtp.dll',
    'qnn_context_priority': 'high',
    'profiling_level': 'basic'
}

session = ort.InferenceSession(
    model_path,
    providers=[
        ('QNNExecutionProvider', qnn_options),
        'CPUExecutionProvider'
    ]
)
```

### オプション一覧

#### 1. `backend_path`

QNNバックエンドDLLのパス。

**値:**
- `QnnHtp.dll` - Hexagon Tensor Processor（NPU） **← 推奨**
- `QnnCpu.dll` - CPU（デバッグ用）
- `QnnGpu.dll` - GPU（存在する場合）

**例:**
```python
'backend_path': 'QnnHtp.dll'
```

#### 2. `qnn_context_priority`

QNNコンテキストの実行優先度。

**値:**
- `low` - 省電力優先
- `normal` - バランス
- `high` - 性能優先 **← 推奨**

**影響:**
- 高優先度 → 高速だが消費電力増加
- 低優先度 → 省電力だが速度低下

**例:**
```python
'qnn_context_priority': 'high'
```

#### 3. `profiling_level`

プロファイリング情報の詳細度。

**値:**
- `off` - 無効（最速）
- `basic` - 基本情報（推論時間など） **← 推奨**
- `detailed` - 詳細情報（レイヤーごとの時間など、デバッグ用）

**影響:**
- `off` → 最速だが情報なし
- `detailed` → 詳細情報だが若干遅い

**例:**
```python
'profiling_level': 'basic'
```

#### 4. `enable_htp_fp16_precision`

FP16（半精度浮動小数点）を有効化。

**値:**
- `true` - FP16を使用（高速、省メモリ、若干精度低下）
- `false` - FP32を使用（低速、高精度）

**例:**
```python
'enable_htp_fp16_precision': 'true'
```

**推奨:** 精度に敏感でない場合は `true`

#### 5. `htp_graph_finalization_optimization_mode`

グラフ最適化モード。

**値:**
- `0` - 無効
- `1` - 基本最適化
- `2` - 高度な最適化
- `3` - 最大最適化

**例:**
```python
'htp_graph_finalization_optimization_mode': '3'
```

**推奨:** `3`（最大最適化）

### 推奨設定

#### 高性能モード

```python
qnn_options = {
    'backend_path': 'QnnHtp.dll',
    'qnn_context_priority': 'high',
    'profiling_level': 'off',
    'enable_htp_fp16_precision': 'true',
    'htp_graph_finalization_optimization_mode': '3'
}
```

#### バランスモード（デフォルト）

```python
qnn_options = {
    'backend_path': 'QnnHtp.dll',
    'qnn_context_priority': 'normal',
    'profiling_level': 'basic'
}
```

#### 省電力モード

```python
qnn_options = {
    'backend_path': 'QnnHtp.dll',
    'qnn_context_priority': 'low',
    'profiling_level': 'off',
    'enable_htp_fp16_precision': 'true'
}
```

---

## トラブルシューティング

### ❌ QNNExecutionProviderが見つからない

#### 症状

```
Available Providers: ['CPUExecutionProvider']
```

`QNNExecutionProvider` がリストに含まれない。

#### 原因と対策

| 原因 | 確認方法 | 対策 |
|------|----------|------|
| `onnxruntime-qnn`未インストール | `pip list \| grep onnxruntime` | `pip install onnxruntime-qnn` |
| 通常の`onnxruntime`と競合 | `pip list \| grep onnxruntime` | アンインストール後、QNN版を再インストール |
| NPU非搭載デバイス | デバイスマネージャー確認 | NPU搭載デバイスが必要 |
| 古いバージョン | `pip show onnxruntime-qnn` | 最新版にアップデート |

**確認コマンド:**
```powershell
pip list | Select-String onnxruntime
```

**クリーンインストール:**
```powershell
pip uninstall -y onnxruntime onnxruntime-qnn
pip install onnxruntime-qnn
```

### ❌ セッション作成エラー

#### 症状

```
❌ セッション作成エラー: [ONNXRuntimeError] ...
```

#### 原因と対策

1. **モデルファイルが見つからない**
   ```
   Error: No such file or directory: models/yolov8n.onnx
   ```
   対策: `models/yolov8n.onnx` を配置

2. **モデルが破損**
   ```
   Error: Failed to load model
   ```
   対策: モデルを再ダウンロード

3. **QNNバックエンドDLLが見つからない**
   ```
   Error: QnnHtp.dll not found
   ```
   対策: `onnxruntime-qnn` を再インストール

### ⚠️ CPUにフォールバック

#### 症状

```
Active Providers: ['CPUExecutionProvider']
```

QNNがリクエストされたが、CPUで実行されている。

#### 原因

- モデルがQNNで対応していないオペレーターを含む
- QNNの初期化に失敗

#### 対策

1. `check_qnn.py` でQNN環境を確認
2. モデルがONNX形式で、QNN対応オペレーターのみを使用しているか確認
3. ログを確認して具体的なエラーを特定

### 🐌 推論が遅い

#### 症状

```
FPS: 5-8 (Inf: 150ms)
```

NPU使用時でもFPSが低い。

#### 原因と対策

1. **初回実行**
   - 原因: QNNがモデルをコンパイル中
   - 対策: 2回目以降は高速化される（キャッシュが効く）

2. **優先度が低い**
   ```python
   'qnn_context_priority': 'low'  # これを high に変更
   ```

3. **プロファイリングが有効**
   ```python
   'profiling_level': 'detailed'  # これを off または basic に変更
   ```

4. **入力サイズが大きい**
   ```python
   input_size=(1280, 1280)  # これを (640, 640) に縮小
   ```

5. **実際にCPUで動作している**
   - 画面に「Device: CPU」と表示されている場合
   - `check_qnn.py` でQNN環境を確認

---

## パフォーマンスチューニング

### ベンチマーク

設定ごとのパフォーマンス比較：

| 設定 | FPS | 推論時間 | CPU使用率 | 消費電力 |
|------|-----|----------|-----------|----------|
| **高性能モード** | 30-32 | 10-12ms | 15-20% | 中 |
| **バランスモード** | 28-30 | 12-15ms | 10-15% | 低 |
| **省電力モード** | 25-28 | 15-18ms | 8-12% | 最低 |
| **CPU** | 8-10 | 100-120ms | 80-90% | 高 |

### 最速設定

```python
qnn_options = {
    'backend_path': 'QnnHtp.dll',
    'qnn_context_priority': 'high',
    'profiling_level': 'off',
    'enable_htp_fp16_precision': 'true',
    'htp_graph_finalization_optimization_mode': '3'
}
```

```python
input_size=(320, 320)  # 入力サイズを小さく
conf_threshold=0.7     # 検知閾値を高く（検知数を減らす）
```

**期待値:** 35-40 FPS

### 最高精度設定

```python
qnn_options = {
    'backend_path': 'QnnHtp.dll',
    'qnn_context_priority': 'high',
    'profiling_level': 'off'
}
```

```python
input_size=(1280, 1280)  # 入力サイズを大きく
conf_threshold=0.3       # 検知閾値を低く
```

**期待値:** 15-20 FPS（精度向上）

### バッテリー最適化

```python
qnn_options = {
    'backend_path': 'QnnHtp.dll',
    'qnn_context_priority': 'low',
    'profiling_level': 'off',
    'enable_htp_fp16_precision': 'true'
}
```

```python
input_size=(416, 416)  # 中間サイズ
```

**期待値:** 25-28 FPS、低消費電力

---

## 高度な設定

### キャッシュディレクトリの指定

QNNはコンパイル済みモデルをキャッシュします。

```python
qnn_options = {
    'backend_path': 'QnnHtp.dll',
    'qnn_context_cache_path': 'C:\\path\\to\\cache'
}
```

**メリット:**
- 2回目以降の起動が高速化
- キャッシュを保持することで再コンパイル不要

### デバッグモード

詳細なログを出力：

```python
import logging
logging.basicConfig(level=logging.DEBUG)

qnn_options = {
    'backend_path': 'QnnHtp.dll',
    'profiling_level': 'detailed'
}
```

レイヤーごとの実行時間などが表示されます。

### 複数モデルの使用

複数のONNXモデルを使用する場合、セッションごとにQNNオプションを設定：

```python
session1 = ort.InferenceSession(
    'model1.onnx',
    providers=[('QNNExecutionProvider', qnn_options_high_performance)]
)

session2 = ort.InferenceSession(
    'model2.onnx',
    providers=[('QNNExecutionProvider', qnn_options_power_save)]
)
```

---

## 環境変数

### QNN関連の環境変数

| 変数名 | 説明 | 例 |
|--------|------|-----|
| `QNN_SDK_ROOT` | QNN SDKのルートパス | `C:\Qualcomm\QNN\2.x` |
| `LD_LIBRARY_PATH` (Linux) | ライブラリパス | - |

通常、`onnxruntime-qnn`パッケージには必要なDLLが含まれているため、環境変数の設定は不要です。

---

## まとめ

### ✅ チェックリスト

- [ ] `onnxruntime-qnn`がインストールされている
- [ ] `check_qnn.py`で QNNExecutionProvider が利用可能
- [ ] `simple_demo.py`で「Device: NPU (QNN)」と表示される
- [ ] FPS が 28-30 程度
- [ ] 推論時間が 12-15ms 程度

### 🚀 次のステップ

- [使い方ガイド](使い方.md) - 詳細な使用方法
- [カスタマイズガイド](カスタマイズ.md) - コードのカスタマイズ

---

**問題が解決しない場合は、GitHubのIssueで質問してください！**
