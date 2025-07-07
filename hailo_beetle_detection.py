#!/usr/bin/env python3

"""
Hailo 8L NPU Beetle Detection Script for Raspberry Pi 5
カスタム1クラスYOLOv8甲虫検出モデル用リアルタイム推論スクリプト

Based on: https://murasan-net.com/2024/11/13/raspberry-pi-5-hailo-8l-webcam/
Model: Custom 1-class beetle detection (best.hef)
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import argparse
import multiprocessing
import numpy as np
import setproctitle
import cv2
import time
from hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from detection_pipeline import GStreamerDetectionApp

# 甲虫検出用のクラス定義
BEETLE_CLASS_NAMES = ["beetle"]
BEETLE_CLASS_ID = 0

class BeetleDetectionApp(GStreamerDetectionApp):
    """カスタム甲虫検出アプリケーション"""
    
    def __init__(self, args, user_data):
        super().__init__(args, user_data)
        self.beetle_count = 0
        self.detection_start_time = time.time()
        
    def app_callback(self, pad, info, user_data):
        """甲虫検出コールバック関数"""
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
            
        # フレーム情報取得
        string_to_print = ""
        format, width, height = get_caps_from_pad(pad)
        
        # NumPy配列に変換
        frame = None
        if format is not None:
            frame = get_numpy_from_buffer(buffer, format, width, height)
            
        # ROIメタデータから検出結果を取得
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
        
        # 甲虫検出の処理
        beetle_detections = []
        current_time = time.time()
        
        for detection in detections:
            label = detection.get_label()
            bbox = detection.get_bbox()
            confidence = detection.get_confidence()
            
            # 甲虫のみフィルタリング（クラス0のみ）
            if label == BEETLE_CLASS_NAMES[BEETLE_CLASS_ID]:
                beetle_detections.append({
                    'label': label,
                    'confidence': confidence,
                    'bbox': bbox
                })
                
        # 検出結果の表示
        if beetle_detections:
            self.beetle_count = len(beetle_detections)
            elapsed_time = current_time - self.detection_start_time
            
            string_to_print += f"🪲 甲虫検出: {self.beetle_count}匹 "
            string_to_print += f"(経過時間: {elapsed_time:.1f}秒)\n"
            
            for i, det in enumerate(beetle_detections):
                string_to_print += f"  検出{i+1}: 信頼度 {det['confidence']:.2f}\n"
        else:
            string_to_print += "甲虫は検出されていません\n"
            
        # フレームレート情報
        fps = self.get_fps()
        if fps > 0:
            string_to_print += f"FPS: {fps:.1f}\n"
            
        # 結果出力
        if string_to_print:
            print(string_to_print)
            
        return Gst.PadProbeReturn.OK

def get_pipeline_string(args):
    """
    カスタム甲虫検出用GStreamerパイプライン文字列を生成
    """
    # HEFファイルパスの設定
    hef_path = args.hef_path or "best.hef"
    
    if args.input == 'usb':
        # USBカメラからの入力
        source_element = f"v4l2src device={args.device} name=src_0 ! "
        source_element += "video/x-raw, format=YUY2, width=640, height=480, framerate=30/1 ! "
        source_element += "videoflip video-direction=horiz ! "
        source_element += "videoconvert ! "
        source_element += "video/x-raw, format=RGB, pixel-aspect-ratio=1/1 ! "
        
    elif args.input == 'rpi':
        # Raspberry Pi カメラからの入力
        source_element = "libcamerasrc name=src_0 ! "
        source_element += "video/x-raw, format=RGB, width=640, height=480, framerate=30/1 ! "
        
    else:
        # ファイルからの入力
        source_element = f"filesrc location={args.input} ! "
        source_element += "qtdemux ! queue ! "
        source_element += "h264parse ! v4l2h264dec ! "
        source_element += "videoconvert ! "
        source_element += "videoscale ! "
        source_element += "video/x-raw, format=RGB, width=640, height=480 ! "
        
    # Hailo推論パイプライン
    pipeline_string = source_element
    pipeline_string += "queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! "
    pipeline_string += f"hailonet hef-path={hef_path} batch-size=1 ! "
    pipeline_string += "queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! "
    pipeline_string += f"hailofilter so-path={get_hailofilter_path()} qos=false ! "
    pipeline_string += "queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! "
    pipeline_string += "hailooverlay ! "
    pipeline_string += "queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! "
    pipeline_string += "videoconvert ! "
    pipeline_string += "fpsdisplaysink video-sink=autovideosink name=hailo_display sync=false text-overlay=false "
    
    return pipeline_string

def get_hailofilter_path():
    """HailoFilterライブラリのパスを取得"""
    # Hailo RPi Examples の標準パス
    hailofilter_path = "/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgsthailo_filter.so"
    
    # 代替パス
    if not os.path.exists(hailofilter_path):
        hailofilter_path = "/opt/hailo/tappas/apps/gstreamer/libs/post_processes/libgsthailo_filter.so"
    
    return hailofilter_path

def parse_arguments():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(
        description="Hailo 8L NPU 甲虫検出リアルタイム推論"
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="usb",
        help="入力ソース: 'usb' (USBカメラ), 'rpi' (RPiカメラ), またはファイルパス"
    )
    
    parser.add_argument(
        "--hef-path",
        type=str,
        default="best.hef",
        help="HEFモデルファイルのパス (デフォルト: best.hef)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="/dev/video0",
        help="USBカメラデバイスパス (デフォルト: /dev/video0)"
    )
    
    parser.add_argument(
        "--show-fps", "-f",
        action="store_true",
        help="FPS表示を有効にする"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="詳細ログを有効にする"
    )
    
    return parser.parse_args()

def main():
    """メイン関数"""
    # プロセス名設定
    setproctitle.setproctitle("hailo-beetle-detection")
    
    # 引数解析
    args = parse_arguments()
    
    # HEFファイルの存在確認
    if not os.path.exists(args.hef_path):
        print(f"エラー: HEFファイルが見つかりません: {args.hef_path}")
        print("best.hef ファイルがカレントディレクトリにあることを確認してください。")
        return 1
    
    # ログ設定
    if args.verbose:
        print(f"HEFファイル: {args.hef_path}")
        print(f"入力ソース: {args.input}")
        print(f"デバイス: {args.device}")
        print("=" * 50)
        print("🪲 甲虫検出を開始します...")
        print("終了するには Ctrl+C を押してください")
        print("=" * 50)
    
    # GStreamer初期化
    Gst.init(None)
    
    # パイプライン文字列取得
    pipeline_string = get_pipeline_string(args)
    
    if args.verbose:
        print(f"パイプライン: {pipeline_string}")
    
    # ユーザーデータ設定
    user_data = {
        'show_fps': args.show_fps,
        'verbose': args.verbose,
        'beetle_class_id': BEETLE_CLASS_ID
    }
    
    try:
        # 甲虫検出アプリケーション起動
        app = BeetleDetectionApp(args, user_data)
        app.run()
        
    except KeyboardInterrupt:
        print("\n終了シグナルを受信しました。アプリケーションを終了します...")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())