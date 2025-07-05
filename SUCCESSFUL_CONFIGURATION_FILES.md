# 成功したHailo設定ファイル集

*このファイルには、カスタム1クラスYOLOv8モデルでHEF生成に成功した実際の設定ファイルを記録しています。*

---

## 📋 使用した設定ファイル

### 1. カスタムNMS設定ファイル: `custom_yolov8_nms_config.json`

```json
{
    "nms_scores_th": 0.25,
    "nms_iou_th": 0.45,
    "image_dims": [
        640,
        640
    ],
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

**重要ポイント**:
- `"classes": 1` でカスタム1クラス対応
- `reg_layer`/`cls_layer`にはHAR内部のレイヤー名(`best/conv*`)を使用
- 3つのストライド(8,16,32)に対応する6つの出力レイヤーを正確に指定

### 2. NMSスクリプトファイル: `custom_nms_script.alls`

```bash
nms_postprocess("custom_yolov8_nms_config.json", meta_arch=yolov8, engine=cpu)
```

**重要ポイント**:
- 第1引数は設定ファイルパス（パラメータ名不要）
- `meta_arch=yolov8` 必須
- `engine=cpu` 必須

### 3. 高速コンパイル設定: `faster_compilation.alls`

```bash
performance_param(compiler_optimization_level=0)
```

**用途**: コンパイル時間短縮（開発・テスト用）

---

## 🔧 実行コマンド履歴

### パース（成功パターン）
```bash
echo 'y' | hailo parser onnx weights/best.onnx --hw-arch hailo8l \
  --har-path yolov8s_no_nms_pure.har --end-node-names '/model.22/Concat_3'
```

### レイヤー名調査
```bash
hailo har extract yolov8s_no_nms_pure.har
strings best.hn | grep 'output_layers_order'
```

### 量子化（成功パターン）
```bash
hailo optimize --hw-arch hailo8l --use-random-calib-set \
  --model-script custom_nms_script.alls yolov8s_no_nms_pure.har
```

### HEFコンパイル（成功パターン）
```bash
hailo compiler --hw-arch hailo8l --model-script faster_compilation.alls \
  --output-dir . best_optimized.har
```

---

## 📊 生成されたファイル

### 最終成果物
- **`best.hef`** (9.3MB) - Hailo 8L NPU実行ファイル
- **`best_optimized.har`** (60MB) - 量子化済みモデル  
- **`best_compiled.har`** (69MB) - コンパイル済みモデル

### 中間ファイル
- **`yolov8s_no_nms_pure.har`** (12.4MB) - パース済みモデル
- **`best_nms_config.json`** - 自動生成NMS設定（参考用）

---

## ⚠️ 注意事項

### レイヤー名について
- **原名**: `/model.22/cv2.*/cv2.*.2/Conv` (ONNX内部)
- **内部名**: `best/conv41-63` (HAR内部)
- **設定ファイルでは内部名を使用する**

### 他のモデルでの適用
- レイヤー名は**モデルごとに異なります**
- 必ず `strings model.hn | grep 'output_layers_order'` で確認
- クラス数も適切に設定 (`"classes": N`)

### パフォーマンス
- `compiler_optimization_level=0`: 高速コンパイル、性能は標準
- `compiler_optimization_level=max`: 低速コンパイル、性能最適（本番用）

---

*最終更新: 2025-07-05*  
*成功環境: Hailo DFC 3.31.0, HailoRT 4.21.0, WSL2 + Docker*