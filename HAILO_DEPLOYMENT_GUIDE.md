# YOLOv8甲虫検出モデル用Hailo 8L NPUデプロイメントガイド

## 📋 概要

本文書は、訓練済みYOLOv8甲虫検出モデルをRaspberry Pi AI Kit（Hailo 8L Neural Processing Unit搭載）にデプロイするための包括的なガイドです。Hailo 8Lは13 TOPSのAI推論性能を提供し、物体検出タスクを大幅に高速化します。

## 🎯 プロジェクトスコープ

このデプロイメントガイドは、現在の昆虫検出訓練プロジェクトを基盤とした**独立したプロジェクト**として設計されています。目標は、CPU基盤の推論からNPU加速推論へ移行し、リアルタイム甲虫検出を実現することです。

## 🏗️ システムアーキテクチャ

### 現在のアーキテクチャ（CPU）
```
PyTorchモデル (.pt) → CPU推論 → 標準的な処理時間
```

### 目標アーキテクチャ（Hailo 8L NPU）
```
PyTorchモデル (.pt) → ONNX (.onnx) → HEF (.hef) → NPU推論 → 大幅高速化
```

**注意**: 具体的な性能値はHailoライセンス規約により非公開

## 📋 前提条件

### ハードウェア要件

#### 開発環境（PC/クラウド）
- **OS**: Ubuntu 20.04/22.04 LTS（WSL2対応）
- **CPU**: マルチコアプロセッサ（Intel i5/i7またはAMD Ryzen 5/7）
- **RAM**: 16GB以上推奨（大規模モデルには32GB）
- **GPU**: NVIDIA GPU（T400以上）- オプションだが最適化に強く推奨
- **ストレージ**: ソフトウェアとデータセット用に50GB以上の空き容量

#### ターゲットデプロイメント環境
- **ハードウェア**: Raspberry Pi 5 + Raspberry Pi AI Kit（Hailo 8L NPU）
- **OS**: Raspberry Pi OS Bookworm（64ビット）
- **メモリ**: 8GB RAM推奨
- **ストレージ**: 32GB以上のmicroSDカード（Class 10以上）

### ソフトウェア要件

#### 開発環境
- **Hailo AI Software Suite**（2024-10以降）
- **Docker**（20.10以上）
- **Python**（3.8-3.10）
- **Hailo Dataflow Compiler（DFC）**
- **Hailo Model Zoo**

#### ターゲット環境
- **Hailoランタイムライブラリ**
- **rpicam-apps**（AIモジュールサポート付き）
- **Python 3.10以上**

## 🔧 インストールとセットアップ

### 1. 開発環境のセットアップ

#### 1.1 Hailo AI Software Suiteのインストール

```bash
# Hailo Developer Zoneから登録・ダウンロード
# https://hailo.ai/developer-zone/

# ダウンロードしたアーカイブを展開
tar -xzf hailo-ai-sw-suite-2024-10.tgz
cd hailo-ai-sw-suite-2024-10

# 提供されたスクリプトを使用してインストール
sudo ./install.sh
```

#### 1.2 Docker環境のセットアップ（推奨）

```bash
# Hailo Dockerコンテナをプル
docker pull hailo/hailo-ai-sw-suite:2024-10

# GPUサポート付きでコンテナを実行（利用可能な場合）
docker run -it --gpus all \
  -v $(pwd):/workspace \
  -v /path/to/your/dataset:/data \
  hailo/hailo-ai-sw-suite:2024-10
```

#### 1.3 Python仮想環境

```bash
# 仮想環境を作成
python3 -m venv hailo-env
source hailo-env/bin/activate

# Hailoツールをインストール
pip install hailo-dataflow-compiler
pip install hailo-model-zoo
```

### 2. Raspberry Pi AI Kitのセットアップ

#### 2.1 OSインストールと設定

```bash
# Raspberry Pi OS Bookworm（64ビット）をSDカードにフラッシュ
# 起動してシステムを更新
sudo apt update && sudo apt upgrade -y

# 最適なパフォーマンスのためにPCIe Gen 3を有効化
sudo raspi-config
# Advanced Options → PCIe Speed → Yes（PCIe Gen 3モードを有効化）
sudo reboot
```

#### 2.2 Hailoソフトウェアのインストール

```bash
# Hailoパッケージをインストール
sudo apt install hailo-all

# インストールを確認
hailo fw-control identify
```

## 🚀 モデル変換ワークフロー

### フェーズ1: PyTorchからONNXへの変換

#### 1.1 訓練済みモデルの準備

```bash
# 訓練済み甲虫検出モデルを確実に用意
# 現在のプロジェクトから: weights/best.pt
cp /path/to/insect-detection-training/weights/best.pt ./
```

#### 1.2 ONNX形式へのエクスポート

```bash
# Ultralytics YOLOを使用
yolo export model=best.pt imgsz=640 format=onnx opset=11

# ONNXモデルを検証
python -c "
import onnx
model = onnx.load('best.onnx')
onnx.checker.check_model(model)
print('ONNXモデルは有効です')
"
```

### フェーズ2: ONNXからHEFへの変換

#### 2.1 キャリブレーションデータセットの準備

```bash
# キャリブレーションデータセットディレクトリを作成
mkdir -p calibration_data

# 訓練データセットから代表的な画像をコピー
# 最適なキャリブレーションのために64-1024枚の多様な画像を使用
cp /path/to/datasets/train/images/*.jpg calibration_data/
ls calibration_data | wc -l  # キャリブレーション画像数を表示
```

#### 2.2 Hailo Model Zoo設定

```bash
# Hailo Model Zooをクローン（スイートに含まれていない場合）
git clone https://github.com/hailo-ai/hailo_model_zoo.git

# YOLOv8設定に移動
cd hailo_model_zoo/cfg/networks/

# 甲虫検出用のカスタム設定を作成
cp yolov8n.yaml yolov8n_beetle.yaml
```

`yolov8n_beetle.yaml`を編集:
```yaml
network:
  network_name: yolov8n_beetle
  primary_input_shape: [1, 3, 640, 640]
  
postprocessing:
  nms:
    classes: 1  # 単一クラス: 甲虫
    bbox_decoders: [...]
    
quantization:
  calib_set_path: /path/to/calibration_data
```

#### 2.3 Hailo DFCでのモデルコンパイル

```bash
# Hailo Model Zooコンパイルコマンドを使用
hailomz compile \
  --ckpt best.onnx \
  --calib-path ./calibration_data/ \
  --yaml hailo_model_zoo/cfg/networks/yolov8n_beetle.yaml \
  --classes 1 \
  --hw-arch hailo8l \
  --output-dir ./compiled_models/

# 代替方法: 手動DFCコンパイル
hailo parser onnx best.onnx \
  --output-dir parsed_model \
  --start-node-names images \
  --end-node-names output0

hailo optimize \
  --model-script parsed_model/yolov8n.hn \
  --calib-path calibration_data \
  --output-dir optimized_model

hailo compiler \
  --model-script optimized_model/yolov8n_optimized.hn \
  --output-dir compiled_model \
  --hw-arch hailo8l
```

#### 2.4 検証

```bash
# HEFファイルの作成を確認
ls compiled_models/*.hef

# 基本的な検証（オプション）
hailo run compiled_models/yolov8n_beetle.hef \
  --input-files test_image.jpg \
  --output-dir validation_results/
```

## 🎯 Raspberry Piでのデプロイメント

### 3.1 Raspberry Piへのモデル転送

```bash
# HEFファイルをRaspberry Piにコピー
scp compiled_models/yolov8n_beetle.hef pi@raspberrypi.local:~/models/
```

### 3.2 実装例

#### 3.2.1 rpicam-appsの使用（組み込みサポート）

```bash
# カメラでのリアルタイム検出
rpicam-hello --post-process-file /usr/share/rpi-camera-assets/hailo_yolov8_inference.json

# 甲虫モデルでのカスタム検出
# 設定ファイルを編集してHEFモデルを指定
sudo nano /usr/share/rpi-camera-assets/hailo_yolov8_inference.json
```

#### 3.2.2 カスタムPython実装

`beetle_detection_hailo.py`を作成:
```python
#!/usr/bin/env python3
"""
Hailo 8L NPUを使用した甲虫検出
Raspberry Pi AI Kit用に最適化されたリアルタイム推論
"""

import cv2
import numpy as np
from hailo_platform import (HEF, VDevice, HailoStreamInterface, 
                          InferVStreams, ConfigureParams)

class BeetleDetectorHailo:
    def __init__(self, hef_path, confidence_threshold=0.25):
        self.confidence_threshold = confidence_threshold
        self.hef = HEF(hef_path)
        
        # Hailoデバイスを設定
        self.target = VDevice()
        self.network_group = self.target.configure(self.hef)
        self.network_group_params = self.network_group.create_params()
        
        # 入力/出力ストリームを設定
        self.input_vstreams_params = self.network_group_params.input_vstreams_params
        self.output_vstreams_params = self.network_group_params.output_vstreams_params
        
    def preprocess_image(self, image):
        """Hailo NPU推論用の画像前処理"""
        # モデル入力サイズにリサイズ（640x640）
        resized = cv2.resize(image, (640, 640))
        
        # BGRからRGBに変換し正規化
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb_image.astype(np.float32) / 255.0
        
        # バッチ次元を追加しNCHW形式に変換
        input_data = np.transpose(normalized, (2, 0, 1))
        input_data = np.expand_dims(input_data, axis=0)
        
        return input_data
    
    def postprocess_results(self, raw_output, original_shape):
        """Hailo NPU出力を処理して甲虫検出結果を抽出"""
        # 実装はHailoの出力形式に依存
        # 実際のHEF出力に基づいて適応する必要があります
        detections = []
        
        # 出力テンソルを解析
        # バウンディングボックス、信頼度スコア、クラス予測を抽出
        # NMSと信頼度フィルタリングを適用
        
        return detections
    
    def detect(self, image):
        """入力画像に対して甲虫検出を実行"""
        start_time = time.time()
        
        # 前処理
        input_data = self.preprocess_image(image)
        
        # Hailo NPUで推論
        with InferVStreams(self.network_group, 
                          self.input_vstreams_params, 
                          self.output_vstreams_params) as infer_pipeline:
            
            # 推論を実行
            raw_output = infer_pipeline.infer(input_data)
        
        # 後処理
        detections = self.postprocess_results(raw_output, image.shape)
        
        processing_time = (time.time() - start_time) * 1000
        return detections, processing_time

def main():
    # 検出器を初期化
    detector = BeetleDetectorHailo("models/yolov8n_beetle.hef")
    
    # カメラを初期化
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 甲虫を検出
        detections, proc_time = detector.detect(frame)
        
        # 結果を描画
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"甲虫: {confidence:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 処理状況表示（性能値は非表示）
        status = "Processing..." if proc_time > 0 else "Ready"
        cv2.putText(frame, f"Status: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('甲虫検出 - Hailo NPU', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

## 📊 パフォーマンス最適化

### 4.1 ベンチマーク比較

| 設定 | 処理時間 | スループット | 消費電力 |
|---------------|----------------|-----|-------------------|
| **CPU（現在）** | 標準的な時間 | 標準的な処理 | 高 |
| **Hailo 8L NPU** | 大幅短縮 | 大幅向上 | 低 |
| **性能向上** | **大幅高速化** | **大幅向上** | **大幅に低減** |

**注意**: 具体的な性能値はHailoライセンス規約により非公開

### 4.2 最適化のコツ

#### 4.2.1 モデル最適化
- 甲虫検出に適切な量子化パラメータを使用
- 入力前処理パイプラインを最適化
- 特定の用途に対してモデルプルーニングを検討

#### 4.2.2 システム最適化
- Raspberry Pi 5でPCIe Gen 3モードを有効化
- 高速microSDカード（UHS-I Class 3）を使用
- カメラキャプチャ設定を最適化

#### 4.2.3 アプリケーション最適化
- 効率的なフレームバッファリングを実装
- カメラキャプチャに非同期処理を使用
- 視覚化とログオーバーヘッドを最適化

## 🔍 トラブルシューティング

### 一般的な問題と解決策

#### 5.1 モデル変換の問題

**問題**: ONNXエクスポートが失敗する
```bash
# 解決策: ONNXバージョンの互換性を確認
pip install onnx==1.12.0
yolo export model=best.pt format=onnx opset=11
```

**問題**: HEFコンパイルが失敗する
```bash
# 解決策: キャリブレーションデータセットを確認
ls calibration_data | head -10  # 画像の存在を確認
file calibration_data/*.jpg     # 画像形式を確認
```

#### 5.2 ランタイムの問題

**問題**: Hailoデバイスが検出されない
```bash
# 解決策: PCIe設定を確認
lspci | grep Hailo
sudo hailo fw-control identify
```

**問題**: 検出性能が低い
```bash
# 解決策: 後処理パラメータを調整
# モデル設定でNMS閾値と信頼度値を編集
```

### 5.3 パフォーマンスの問題

**問題**: NPU加速にもかかわらず処理速度が期待値より低い
- 前処理のボトルネックを確認
- 最適なカメラキャプチャ設定を確認
- システムリソース使用状況を監視

## 📚 参考資料とリソース

### 公式ドキュメント
- [Hailo Developer Zone](https://hailo.ai/developer-zone/)
- [Raspberry Pi AI Kitドキュメント](https://www.raspberrypi.com/documentation/computers/ai-kit.html)
- [Hailo Model Zoo GitHub](https://github.com/hailo-ai/hailo_model_zoo)

### コミュニティリソース
- [Hailoコミュニティフォーラム](https://community.hailo.ai/)
- [Raspberry Piフォーラム - AI Kitセクション](https://forums.raspberrypi.com/)

### サンプルプロジェクト
- [hailo-rpi5-examples](https://github.com/hailo-ai/hailo-rpi5-examples)
- YOLOv8デプロイメント例とチュートリアル

## 🚀 次のステップ

### 即座のアクション
1. **開発環境のセットアップ**: Hailo AI Software Suiteをインストール
2. **モデル変換**: 訓練済み甲虫検出モデルをHEF形式に変換
3. **ハードウェアの取得**: Raspberry Pi 5 + AI Kitを購入

### 将来の機能強化
1. **多種検出**: 異なる昆虫種の検出に拡張
2. **エッジコンピューティング統合**: ローカルデータ処理と保存の実装
3. **IoT統合**: 監視と分析用クラウドサービスへの接続
4. **パフォーマンス監視**: 包括的なベンチマークツールの実装

## 📝 ライセンスと帰属

このデプロイメントガイドは昆虫検出訓練プロジェクトの一部です。
- **ベースプロジェクト**: [insect-detection-training](https://github.com/Murasan201/insect-detection-training)
- **モデルライセンス**: AGPL-3.0（YOLOv8から継承）
- **Hailoソフトウェア**: Hailoライセンス条件に準拠
- **ドキュメントライセンス**: MIT

---

**注意**: このガイドはHailo 8Lデプロイメントの基盤を提供します。具体的な実装詳細は、ソフトウェアバージョンやハードウェア構成によって異なる場合があります。最新情報については、常に最新のHailoドキュメントを参照してください。