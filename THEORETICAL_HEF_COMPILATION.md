# YOLOv8甲虫検出モデル Hailo 8L実装ガイド
## 調査結果に基づく実際のコンパイル手順

*注意: この文書は調査報告書に基づく実際の手順です*

---

## 📋 前提条件確認

### ✅ 準備完了済みアセット
1. **PyTorchモデル**: `weights/best.pt` (6.0MB)
2. **ONNXモデル**: `weights/best.onnx` (11.7MB)
3. **キャリブレーションデータ**: `calibration_data/` (64画像、推奨1024枚以上)
4. **Python環境**: hailo-env with required packages

---

## 🔧 実際のHailo SDK設定

### 1. Hailo SDKコンポーネントインストール

```bash
# 3つの必要コンポーネントをDeveloper Zoneからダウンロード後
sudo dpkg -i hailort_<version>_amd64.deb
pip install hailort-<version>-cp<...>.whl
pip install hailo_dataflow_compiler-<version>-py3-none-linux_x86_64.whl

# Hailo Model Zoo設定
git clone https://github.com/hailo-ai/hailo_model_zoo.git
cd hailo_model_zoo
pip install -e .
```

### 2. 重要な発見事項
- ✅ **無償提供**: 登録ユーザであれば無償でダウンロード・使用可能
- ✅ **商用利用可**: 技術書掲載や商用プロジェクトで追加費用なし
- ⚠️ **EULA制限**: 性能ベンチマーク詳細公開、SDK再配布制限

### 1. Hailo Model Zoo設定ファイル

`configs/yolov8n_beetle.yaml`を作成:

```yaml
network:
  network_name: yolov8n_beetle
  primary_input_shape: [1, 3, 640, 640]
  
paths:
  onnx: weights/best.onnx
  calib_set: calibration_data

postprocessing:
  meta_arch: yolo_v8
  nms:
    classes: 1
    bbox_decoders: [
      {
        name: yolov8_bbox_decoder,
        anchors: [
          [8, 16, 32],      # stride 8 anchors
          [16, 32, 64],     # stride 16 anchors  
          [32, 64, 128]     # stride 32 anchors
        ],
        output_activation: sigmoid
      }
    ]
    
quantization:
  precision: int8
  calib_set_path: calibration_data/
  calib_batch_size: 4
  num_calib_batches: 16
  
optimization:
  cluster_layer_inputs: true
  mixed_precision: false
```

### 2. カスタム前処理設定

`preprocessing/beetle_preprocessing.py`:

```python
import numpy as np
import cv2

def preprocess_func(image_path):
    """Hailo用前処理関数"""
    # 画像読み込み
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 640x640にリサイズ
    image = cv2.resize(image, (640, 640))
    
    # 正規化 [0-1]
    image = image.astype(np.float32) / 255.0
    
    # チャネル順変更 HWC -> CHW
    image = np.transpose(image, (2, 0, 1))
    
    # バッチ次元追加
    image = np.expand_dims(image, axis=0)
    
    return image
```

---

## 🚀 実際のコンパイル手順（3ステップ）

### ステップ1: モデルのパース（HAR生成）

```bash
# 実際のコマンド
hailomz parse --hw-arch hailo8l --ckpt ./best.onnx yolov8s
```

**実行内容:**
- ONNXモデルをHailo形式（HAR: Hailo Archive）に変換
- ネットワーク構造の解析とvalidation
- Hailo 8L NPUとの互換性確認

### ステップ2: モデルの最適化（量子化）

```bash
# 実際のコマンド
hailomz optimize \
  --hw-arch hailo8l \
  --har yolov8s.har \
  --calib-path ./calibration_data/ \
  yolov8s
```

**最適化プロセス:**
1. **量子化キャリブレーション**: FP32→INT8変換
2. **キャリブレーションデータ**: 64画像使用（1024枚以上推奨）
3. **精度vs性能**: バランス調整
4. **NPUアーキテクチャ最適化**: Hailo 8L専用最適化

### ステップ3: モデルのコンパイル（HEF生成）

```bash
# 実際のコマンド
hailomz compile \
  --hw-arch hailo8l \
  --ckpt ./best.onnx \
  --calib-path ./calibration_data/ \
  --yaml hailo_model_zoo/cfg/networks/yolov8s.yaml \
  --classes 1
```

**コンパイルプロセス:**
1. **カスタマイズ**: `--classes 1` で甲虫検出用に調整
2. **Model Zoo設定**: `yolov8s.yaml` をベースに使用
3. **HEF生成**: `*.hef` ファイル出力
4. **デプロイ準備**: Raspberry Pi転送可能形式

---

## 📊 期待される性能改善

### パフォーマンス期待値

| メトリック | CPU (現在) | Hailo 8L NPU | 期待される改善 |
|-----------|------------|--------------|----------------|
| **推論時間** | 標準的な時間 | 大幅短縮 | 大幅な高速化 |
| **スループット** | 標準的な処理 | 実時間処理 | リアルタイム対応 |
| **消費電力** | 高消費電力 | 低消費電力 | エッジ向け最適化 |
| **精度** | mAP@0.5: 97.63% | 軽微な劣化* | 量子化影響考慮 |

*量子化による軽微な精度低下の可能性

**重要**: Hailoライセンス規約により、具体的な性能数値の公開は禁止されています。

### メモリ使用量

```
Original Model (FP32): 11.7MB
Quantized Model (INT8): 大幅削減予想
HEF Runtime: コンパクト化予想
```

**注意**: 具体的なメモリ使用量はHailoライセンス規約により非公開

---

## 🎯 Raspberry Pi統合（実際の手順）

### 1. Raspberry Pi環境セットアップ

```bash
# Raspberry Pi上でのHailo-8Lランタイムインストール
sudo apt update && sudo apt install -y hailo-all
sudo reboot
```

### 2. 公式サンプルコード使用

```bash
# hailo-rpi5-examplesを使用した実行
git clone https://github.com/hailo-ai/hailo-rpi5-examples.git
cd hailo-rpi5-examples
source setup_env.sh

# カスタムモデルでの検出実行
python basic_pipelines/detection.py \
  --labels-json custom.json \
  --hef-path /home/pi/your_model.hef \
  --input usb -f
```

### 3. カスタム統合実装例

```python
# theoretical_beetle_detector_hailo.py
import cv2
import numpy as np
from hailo_platform import (HEF, VDevice, HailoStreamInterface, 
                          InferVStreams, ConfigureParams)

class BeetleDetectorHailo:
    def __init__(self, hef_path="compiled_model/yolov8n_beetle.hef"):
        self.hef_path = hef_path
        self.confidence_threshold = 0.25
        self.nms_threshold = 0.45
        
        # Hailoデバイス初期化
        self._init_hailo_device()
        
    def _init_hailo_device(self):
        """Hailo NPUデバイス初期化"""
        try:
            # HEFファイル読み込み
            self.hef = HEF(self.hef_path)
            
            # デバイス取得
            self.target = VDevice()
            
            # ネットワークグループ設定
            self.network_group = self.target.configure(self.hef)
            self.network_group_params = self.network_group.create_params()
            
            # 入出力ストリーム設定
            self.input_vstreams_params = self.network_group_params.input_vstreams_params
            self.output_vstreams_params = self.network_group_params.output_vstreams_params
            
            print("✅ Hailo NPU initialized successfully")
            
        except Exception as e:
            print(f"❌ Hailo NPU initialization failed: {e}")
            raise
    
    def preprocess_image(self, image):
        """画像前処理 (Hailo NPU用)"""
        # 640x640にリサイズ
        resized = cv2.resize(image, (640, 640))
        
        # BGR→RGB変換
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 正規化 [0-1]
        normalized = rgb_image.astype(np.float32) / 255.0
        
        # HWC→CHW変換
        chw_image = np.transpose(normalized, (2, 0, 1))
        
        # バッチ次元追加
        batched = np.expand_dims(chw_image, axis=0)
        
        return batched
    
    def postprocess_detections(self, raw_output, original_shape):
        """NPU出力後処理"""
        # YOLOv8出力: [1, 5, 8400] -> [8400, 5]
        output = raw_output[0].transpose(1, 0)  # [8400, 5]
        
        detections = []
        h_orig, w_orig = original_shape[:2]
        
        for detection in output:
            x_center, y_center, width, height, confidence = detection
            
            if confidence > self.confidence_threshold:
                # 正規化座標を元画像座標に変換
                x1 = int((x_center - width/2) * w_orig / 640)
                y1 = int((y_center - height/2) * h_orig / 640)
                x2 = int((x_center + width/2) * w_orig / 640)
                y2 = int((y_center + height/2) * h_orig / 640)
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(confidence),
                    'class': 'beetle'
                })
        
        return detections
    
    def detect(self, image):
        """甲虫検出実行"""
        import time
        start_time = time.time()
        
        # 前処理
        input_data = self.preprocess_image(image)
        
        # NPU推論
        with InferVStreams(self.network_group, 
                          self.input_vstreams_params, 
                          self.output_vstreams_params) as infer_pipeline:
            
            raw_output = infer_pipeline.infer(input_data)
        
        # 後処理
        detections = self.postprocess_detections(raw_output, image.shape)
        
        processing_time = (time.time() - start_time) * 1000  # ms
        return detections, processing_time

# 使用例
def main():
    detector = BeetleDetectorHailo()
    
    # リアルタイムカメラ検出
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 甲虫検出
        detections, proc_time = detector.detect(frame)
        
        # 結果描画
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Beetle: {conf:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 処理状況表示（具体的FPS値は非表示）
        status = "Processing..." if proc_time > 0 else "Ready"
        cv2.putText(frame, f"Status: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Beetle Detection - Hailo NPU', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

---

## 🔧 トラブルシューティング

### 一般的な問題と解決策

#### 1. コンパイルエラー

**問題**: ONNX→HEF変換失敗
```bash
Error: Unsupported operator in ONNX model
```

**解決策**: ONNXオペレータ互換性確認
```bash
hailo parser onnx --check-compatibility weights/best.onnx
```

#### 2. キャリブレーション問題

**問題**: 量子化精度低下
```
Warning: Quantization accuracy degradation > 5%
```

**解決策**: キャリブレーションデータ改善
- より多様な画像使用 (64→256枚)
- 実際の訓練データ使用
- バッチサイズ調整

#### 3. NPU実行時エラー

**問題**: Hailo NPU初期化失敗
```
Error: Failed to initialize Hailo device
```

**解決策**: ハードウェア・ドライバ確認
```bash
lspci | grep Hailo
sudo hailo fw-control identify
```

---

## 📚 参考資料

### Hailo公式ドキュメント
- [Hailo Developer Zone](https://hailo.ai/developer-zone/)
- [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo)
- [YOLOv8 Hailo Example](https://github.com/hailo-ai/hailo_model_zoo/tree/master/hailo_model_zoo/cfg/networks)

### 代替実装オプション
- [OpenVINO YOLOv8](https://docs.openvino.ai/latest/notebooks/230-yolov8-optimization-with-output.html)
- [TensorRT YOLOv8](https://github.com/triple-Mu/YOLOv8-TensorRT)
- [ONNX Runtime](https://onnxruntime.ai/)

---

## ⚠️ 重要な免責事項

### ライセンス・規約準拠
- **調査結果**: Hailo SDKは登録ユーザに無償提供、商用利用も可能
- **EULA制限**: 性能ベンチマーク詳細公開、SDK再配布、リバースエンジニアリング制限
- **実装推奨**: 最新のHailo公式ドキュメントとライセンス条項を確認
- **技術書掲載**: 追加費用なしで商用プロジェクトでの使用可能

### 技術的制限事項
- この文書は調査報告書に基づく実際の手順ですが、環境により差異の可能性
- Hailo SDK APIは変更される可能性があります
- 最終的な実装はHailo公式サポートとの連携を推奨します
- キャリブレーション画像は1024枚以上推奨（現在64枚）

---

*最終更新: 2025-07-04 23:20 JST*  
*調査報告書反映: 実際の手順・ライセンス情報更新*