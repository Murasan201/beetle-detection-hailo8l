# Hailo 8L NPU 甲虫検出 リアルタイム推論

このディレクトリには、Raspberry Pi 5上でHailo 8L NPUを使用してカスタム甲虫検出モデル(`best.hef`)でリアルタイム推論を実行するためのスクリプトが含まれています。

## 📁 ファイル構成

```
beetle-detection-hailo8l/
├── best.hef                      # コンパイル済みHailo NPUモデル (9.3MB)
├── hailo_beetle_detection.py     # メイン推論スクリプト
├── hailo_rpi_common.py          # GStreamer共通ユーティリティ
├── detection_pipeline.py        # 検出パイプラインベースクラス
└── HAILO_INFERENCE_README.md    # このファイル
```

## 🔧 前提条件

### ハードウェア要件
- **Raspberry Pi 5** (8GB RAM推奨)
- **Hailo 8L NPU** (Raspberry Pi AI Kit)
- **USBカメラ** または **Raspberry Pi カメラモジュール**

### ソフトウェア要件
- **Raspberry Pi OS** (64-bit, 最新版)
- **Hailo Runtime** (HailoRT)
- **GStreamer** with Hailo plugins
- **Python 3.9+**

## 📦 インストール手順

### 1. Raspberry Pi OSの更新
```bash
sudo apt update && sudo apt upgrade -y
```

### 2. Hailo Runtime環境のインストール
```bash
# Hailo APTリポジトリの追加
echo "deb https://hailo-archive.s3.amazonaws.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hailo.list

# Hailo GPGキーの追加
wget -qO - https://hailo-archive.s3.amazonaws.com/hailo-archive-keyring.gpg | sudo apt-key add -

# Hailo packages のインストール
sudo apt update
sudo apt install -y hailo-all
```

### 3. 必要なPythonパッケージのインストール
```bash
# システムパッケージ
sudo apt install -y python3-pip python3-gi python3-gi-cairo gir1.2-gstreamer-1.0

# Pythonライブラリ
pip3 install --user numpy opencv-python setproctitle
```

### 4. GStreamer Hailo pluginsの確認
```bash
# Hailo プラグインの存在確認
gst-inspect-1.0 hailonet
gst-inspect-1.0 hailofilter
gst-inspect-1.0 hailooverlay
```

### 5. NPUの動作確認
```bash
# Hailo 8L NPUの認識確認
hailortcli scan

# 期待される出力:
# Hailo Devices:
# [-] Device: 0000:01:00.0 (PCIe)
```

## 🚀 実行方法

### 基本的な使用方法

#### USBカメラを使用する場合
```bash
python3 hailo_beetle_detection.py --input usb --device /dev/video0
```

#### Raspberry Pi カメラを使用する場合
```bash
python3 hailo_beetle_detection.py --input rpi
```

#### 動画ファイルを使用する場合
```bash
python3 hailo_beetle_detection.py --input /path/to/video.mp4
```

### コマンドライン引数

| 引数 | 短縮形 | デフォルト | 説明 |
|------|--------|------------|------|
| `--input` | `-i` | `usb` | 入力ソース (`usb`, `rpi`, またはファイルパス) |
| `--hef-path` | - | `best.hef` | HEFモデルファイルのパス |
| `--device` | - | `/dev/video0` | USBカメラデバイスパス |
| `--show-fps` | `-f` | False | FPS表示を有効にする |
| `--verbose` | `-v` | False | 詳細ログを有効にする |

### 使用例

#### 詳細ログでUSBカメラから推論
```bash
python3 hailo_beetle_detection.py --input usb --show-fps --verbose
```

#### カスタムHEFファイルを指定
```bash
python3 hailo_beetle_detection.py --hef-path /path/to/custom.hef --input rpi
```

#### 特定のカメラデバイスを使用
```bash
python3 hailo_beetle_detection.py --input usb --device /dev/video2
```

## 📊 実行時の出力例

```
HEFファイル: best.hef
入力ソース: usb
デバイス: /dev/video0
==================================================
🪲 甲虫検出を開始します...
終了するには Ctrl+C を押してください
==================================================

🪲 甲虫検出: 2匹 (経過時間: 5.3秒)
  検出1: 信頼度 0.87
  検出2: 信頼度 0.92
FPS: 28.5

甲虫は検出されていません
FPS: 29.1
```

## 🔧 トラブルシューティング

### よくある問題と解決方法

#### 1. HEFファイルが見つからない
```
エラー: HEFファイルが見つかりません: best.hef
```
**解決方法**: `best.hef`ファイルがスクリプトと同じディレクトリにあることを確認

#### 2. Hailo NPUが認識されない
```
Error: No Hailo devices found
```
**解決方法**: 
- Raspberry Pi 5とHailo 8L NPUが正しく接続されているか確認
- `sudo reboot`で再起動
- `hailortcli scan`でデバイス認識を確認

#### 3. カメラデバイスが見つからない
```
Error: Could not open video device /dev/video0
```
**解決方法**:
- `ls /dev/video*`でカメラデバイスを確認
- 正しいデバイスパスを`--device`引数で指定
- カメラの接続を確認

#### 4. GStreamerプラグインエラー
```
Error: No such element or plugin 'hailonet'
```
**解決方法**:
- `sudo apt install hailo-all`でHailoパッケージを再インストール
- `gst-inspect-1.0 hailonet`でプラグインの存在確認

#### 5. 権限エラー
```
Error: Permission denied
```
**解決方法**:
- ユーザーを`video`グループに追加: `sudo usermod -a -G video $USER`
- ログアウト・ログインして権限を反映

### パフォーマンス最適化

#### GPU メモリ分割の調整
```bash
# /boot/config.txt に以下を追加
gpu_mem=128
```

#### CPUガバナーの設定
```bash
# パフォーマンスモードに設定
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

## 📈 期待される性能

- **推論速度**: CPU処理と比較して大幅な高速化
- **検出精度**: mAP@0.5: 97.63% (モデル精度)
- **消費電力**: エッジデバイス向け最適化
- **レイテンシ**: ハードウェアアクセラレーション対応

*注意: 性能は使用環境とハードウェア構成により変動します*

## 🛠️ カスタマイズ

### 検出しきい値の調整
`hailo_beetle_detection.py`内の以下の値を変更:
```python
# NMSしきい値（信頼度）
confidence_threshold = 0.25

# IoUしきい値
iou_threshold = 0.45
```

### 表示設定の変更
```python
# 検出ボックスの色変更
box_color = (0, 255, 0)  # 緑色

# テキスト表示の調整
show_confidence = True
show_class_name = True
```

## 📋 ライセンス

- **コード**: MIT License
- **モデル**: AGPL-3.0 (YOLOv8ベース)
- **データセット**: CC BY 4.0

## 🔗 参考リンク

- [Raspberry Pi AI Kit](https://www.raspberrypi.com/documentation/computers/ai-kit.html)
- [Hailo Developer Zone](https://hailo.ai/developer-zone/)
- [元記事](https://murasan-net.com/2024/11/13/raspberry-pi-5-hailo-8l-webcam/)

---

*最終更新: 2025-07-07*