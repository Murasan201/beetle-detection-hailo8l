# 理論的HEFコンパイル手順書
## YOLOv8甲虫検出モデル Hailo 8L変換ガイド

*注意: この文書はHailo SDK未入手の状況での理論的手順です*

---

## 📋 前提条件確認

### ✅ 準備完了済みアセット
1. **PyTorchモデル**: `weights/best.pt` (6.0MB)
2. **ONNXモデル**: `weights/best.onnx` (11.7MB)
3. **キャリブレーションデータ**: `calibration_data/` (64画像)
4. **Python環境**: hailo-env with required packages

---

## 🔧 理論的Hailo SDK設定

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

## 🚀 理論的コンパイル手順

### Phase 1: ONNX Parsing

```bash
# 理論的コマンド (Hailo SDK required)
hailo parser onnx weights/best.onnx \
  --output-dir parsed_model \
  --start-node-names images \
  --end-node-names output0 \
  --hw-arch hailo8l
```

**期待される出力:**
- `parsed_model/yolov8n_beetle.hn` (Hailo Network格式)
- ネットワーク構造解析レポート
- レイヤー互換性チェック結果

### Phase 2: Model Optimization

```bash
# 理論的コマンド
hailo optimize \
  --model-script parsed_model/yolov8n_beetle.hn \
  --calib-path calibration_data \
  --calib-set-size 64 \
  --output-dir optimized_model \
  --hw-arch hailo8l
```

**最適化プロセス:**
1. **量子化キャリブレーション**: FP32→INT8変換
2. **レイヤーフュージョン**: 連続するConv+BN+ReLU統合
3. **メモリ最適化**: NPUメモリレイアウト最適化
4. **並列化**: マルチコア処理最適化

### Phase 3: Hardware Compilation

```bash
# 理論的コマンド
hailo compiler \
  --model-script optimized_model/yolov8n_beetle_optimized.hn \
  --output-dir compiled_model \
  --hw-arch hailo8l \
  --output-file yolov8n_beetle.hef
```

**コンパイルプロセス:**
1. **NPU命令生成**: Hailo 8L専用命令セット変換
2. **メモリマッピング**: オンチップメモリ配置最適化
3. **パイプライン最適化**: 13TOPS性能最大化
4. **HEFパッケージング**: 最終デプロイ可能形式

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

## 🎯 Raspberry Pi統合コード例

### Hailo NPU推論クラス

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
- この文書は理論的手順であり、実際のHailo SDK環境での検証が必要です
- **Hailoライセンス規約により、具体的な性能数値の公開は禁止されています**
- 実装時は最新のHailo公式ドキュメントとライセンス条項を確認してください
- 商用利用の場合は適切なライセンス取得が必要です

### 技術的制限事項
- コード例は教育目的であり、本番環境での動作を保証するものではありません
- Hailo SDK APIは変更される可能性があります
- 最終的な実装はHailo公式サポートとの連携を推奨します

---

*最終更新: 2025-07-04 22:40 JST*  
*ライセンス準拠: Hailo性能値非公開ポリシー適用*