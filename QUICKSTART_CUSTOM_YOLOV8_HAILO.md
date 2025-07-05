# カスタムYOLOv8モデル → Hailo 8L HEFファイル生成 クイックスタートガイド

*カスタム1クラスYOLOv8モデルを30分でHailo 8L NPU対応HEFファイルに変換*

---

## 🎯 前提条件

✅ **必要ファイル**:
- `weights/best.onnx` - カスタムYOLOv8 ONNXモデル
- `calibration_data/` - キャリブレーション用画像（64枚以上）

✅ **環境**:
- Linux (WSL2推奨)
- Docker 20.10+
- Hailo AI Software Suite Docker版

---

## 🚀 5ステップで完了

### ステップ1: Hailo Docker環境起動 (5分)

```bash
cd /path/to/hailo-sdk/hailo_ai_sw_suite_2025-04_docker/
sudo docker load -i hailo_ai_sw_suite_2025-04.tar.gz
chmod +x hailo_ai_sw_suite_docker_run.sh
./hailo_ai_sw_suite_docker_run.sh
```

### ステップ2: プロジェクトファイルをコピー (1分)

```bash
# ホストシステムから
docker cp weights/ hailo_ai_sw_suite_2025-04_container:/local/shared_with_docker/
docker cp calibration_data/ hailo_ai_sw_suite_2025-04_container:/local/shared_with_docker/
```

### ステップ3: 設定ファイル作成 (2分)

**3.1 NMS設定ファイル作成**: `custom_yolov8_nms_config.json`
```json
{
    "nms_scores_th": 0.25,
    "nms_iou_th": 0.45,
    "image_dims": [640, 640],
    "max_proposals_per_class": 100,
    "classes": 1,
    "regression_length": 16,
    "background_removal": false,
    "background_removal_index": 0,
    "bbox_decoders": [
        {
            "name": "bbox_decoder_8",
            "stride": 8,
            "reg_layer": "best/conv41",
            "cls_layer": "best/conv42"
        },
        {
            "name": "bbox_decoder_16",
            "stride": 16,
            "reg_layer": "best/conv52",
            "cls_layer": "best/conv53"
        },
        {
            "name": "bbox_decoder_32",
            "stride": 32,
            "reg_layer": "best/conv62",
            "cls_layer": "best/conv63"
        }
    ]
}
```

**3.2 NMSスクリプト作成**: `custom_nms_script.alls`
```bash
nms_postprocess("custom_yolov8_nms_config.json", meta_arch=yolov8, engine=cpu)
```

**3.3 コンパイル設定作成**: `faster_compilation.alls`
```bash
performance_param(compiler_optimization_level=0)
```

### ステップ4: パース・量子化 (5分)

```bash
# Docker内で実行
cd /local/shared_with_docker

# 4.1 パース
echo 'y' | hailo parser onnx weights/best.onnx --hw-arch hailo8l \
  --har-path yolov8s_no_nms_pure.har --end-node-names '/model.22/Concat_3'

# 4.2 量子化
hailo optimize --hw-arch hailo8l --use-random-calib-set \
  --model-script custom_nms_script.alls yolov8s_no_nms_pure.har
```

### ステップ5: HEFコンパイル (15分)

```bash
# Docker内で実行
hailo compiler --hw-arch hailo8l --model-script faster_compilation.alls \
  --output-dir . best_optimized.har

# 成功確認
ls -la best.hef  # 9.3MB程度のファイルが生成される
```

### ステップ6: ファイル取得 (1分)

```bash
# ホストシステムで実行
docker cp hailo_ai_sw_suite_2025-04_container:/local/shared_with_docker/best.hef ./
```

---

## ✅ 成功確認

**生成されるファイル**:
- ✅ `best.hef` (約9.3MB) - **最終目標ファイル**
- ✅ `best_optimized.har` (約60MB) - 量子化済み中間ファイル

**成功ログの例**:
```
✅ Model Optimization is done
✅ Found valid partition to 3 contexts, Performance improved by 38%
✅ Generated: best.hef
```

---

## 🛠️ トラブルシューティング

### よくあるエラーと解決策

#### エラー1: "Cannot infer bbox conv layers automatically"
**原因**: レイヤー名が間違っている  
**解決**: レイヤー名を確認
```bash
hailo har extract yolov8s_no_nms_pure.har
strings best.hn | grep 'output_layers_order'
# 結果に基づいてNMS設定のreg_layer/cls_layerを修正
```

#### エラー2: "The layer ... doesn't exist in the HN"
**原因**: 間違った内部レイヤー名  
**解決**: 上記コマンドで正しいレイヤー名を確認

#### エラー3: "No argument named ..."
**原因**: .alls構文エラー  
**解決**: 正しい構文を使用
```bash
# ❌ 間違い
nms_postprocess(nms_config_file_path="config.json")

# ✅ 正しい
nms_postprocess("config.json", meta_arch=yolov8, engine=cpu)
```

---

## 📊 カスタマイズポイント

### 他のクラス数での使用
```json
{
    "classes": N,  // ← クラス数を変更
    // ... 他の設定は同じ
}
```

### 検出閾値の調整
```json
{
    "nms_scores_th": 0.25,  // ← 信頼度閾値（低い値 = より多く検出）
    "nms_iou_th": 0.45,     // ← IoU閾値（低い値 = より厳しいNMS）
    // ...
}
```

### 本番用高性能コンパイル
```bash
# faster_compilation.alls を以下に変更
performance_param(compiler_optimization_level=max)
# 注意: コンパイル時間が大幅に増加（1時間以上）
```

---

## 🎯 次のステップ

生成された`best.hef`をRaspberry Pi 5 + Hailo 8Lで使用:

```python
from hailo_platform import HEF, VDevice

# HEFファイル読み込み
hef = HEF("best.hef")
device = VDevice()

# 推論実行
# (詳細は公式Hailo Python APIドキュメント参照)
```

---

**💡 ヒント**: 初回は必ずこの手順通りに実行してください。成功後、個別の設定を調整することをお勧めします。

*作成日: 2025-07-05*  
*対応SDK: Hailo DFC 3.31.0, HailoRT 4.21.0*