#!/usr/bin/env python3

"""
Hailo 8L NPU Beetle Detection Script - Fixed Version
カスタム1クラスYOLOv8甲虫検出モデル用リアルタイム推論スクリプト（修正版）

Based on: 根本原因調査結果
Model: Custom 1-class beetle detection (best.hef)
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import argparse
import time
import numpy as np
import cv2
import hailo
import logging
from datetime import datetime
from hailo_rpi_common import get_caps_from_pad, get_numpy_from_buffer

# 甲虫検出用のクラス定義
BEETLE_CLASS_NAMES = ["beetle"]
BEETLE_CLASS_ID = 0

class SimpleBeetleDetectionApp:
    """
    シンプルな甲虫検出アプリケーション（修正版）
    """
    
    def __init__(self, args):
        self.args = args
        self.pipeline = None
        self.loop = None
        self.beetle_count = 0
        self.detection_start_time = time.time()
        self.frame_buffer = None
        self.display_window = None
        self.frame_count = 0
        self.total_detections = 0
        
        # ログファイル設定
        self.setup_logging()
        
        # OpenCVウィンドウ初期化（表示モードの場合）
        if args.display:
            cv2.namedWindow("Beetle Detection", cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow("Beetle Detection", 100, 100)
        
        # 初期ログ出力
        self.logger.info("=" * 60)
        self.logger.info(f"Beetle Detection Session Started")
        self.logger.info(f"HEF File: {args.hef_path}")
        self.logger.info(f"Input Source: {args.input}")
        self.logger.info(f"Device: {args.device}")
        self.logger.info(f"Display Mode: {args.display}")
        self.logger.info("=" * 60)
        
    def get_pipeline_string(self):
        """
        動作することが確認されたパイプライン文字列を生成
        """
        # HEFファイルパス
        hef_path = self.args.hef_path or "best.hef"
        
        if self.args.input == 'usb':
            # USBカメラからの入力（640x640解像度でhailonetと互換性確保）
            source_element = f"v4l2src device={self.args.device} ! "
            source_element += "videoconvert ! "
            source_element += "videoscale ! "
            source_element += "video/x-raw,format=RGB,width=640,height=640 ! "
            source_element += "videoflip video-direction=horiz ! "
            
        elif self.args.input == 'rpi':
            # Raspberry Pi カメラからの入力
            source_element = "libcamerasrc ! "
            source_element += "videoconvert ! "
            source_element += "videoscale ! "
            source_element += "video/x-raw,format=RGB,width=640,height=640 ! "
            
        else:
            # ファイルからの入力
            source_element = f"filesrc location={self.args.input} ! "
            source_element += "qtdemux ! queue ! "
            source_element += "h264parse ! v4l2h264dec ! "
            source_element += "videoconvert ! "
            source_element += "videoscale ! "
            source_element += "video/x-raw,format=RGB,width=640,height=640 ! "
            
        # Hailo推論パイプライン（表示モード対応）
        pipeline_string = source_element
        pipeline_string += "queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! "
        pipeline_string += f"hailonet hef-path={hef_path} batch-size=1 name=hailonet0 ! "
        pipeline_string += "queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! "
        
        if self.args.display:
            # 表示モード: teeでデータを分岐してappsinkでフレーム取得
            pipeline_string += "tee name=t ! "
            pipeline_string += "queue ! appsink name=appsink0 emit-signals=true sync=false max-buffers=2 drop=true "
            pipeline_string += "t. ! queue ! fakesink name=fakesink0"
        else:
            # ヘッドレスモード: 従来通りfakesink
            pipeline_string += "fakesink name=fakesink0"
        
        return pipeline_string
        
    def app_callback(self, pad, info, user_data):
        """甲虫検出コールバック関数"""
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
            
        try:
            self.frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - self.detection_start_time
            
            # フレーム処理ログ（5秒ごと）
            if self.frame_count % 150 == 0:  # 約5秒ごと (30fps想定)
                self.logger.info(f"Frame #{self.frame_count} - Processing at {elapsed_time:.1f}s")
            
            # ROIメタデータから検出結果を取得
            roi = hailo.get_roi_from_buffer(buffer)
            detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
            
            # 全検出結果をログ記録
            all_detections_count = len(detections)
            
            # 甲虫検出の処理
            beetle_detections = []
            
            # 全検出結果の詳細ログ
            if all_detections_count > 0:
                self.logger.info(f"Frame #{self.frame_count}: Total detections found: {all_detections_count}")
                
                for i, detection in enumerate(detections):
                    label = detection.get_label()
                    bbox = detection.get_bbox()
                    confidence = detection.get_confidence()
                    
                    # 全検出をログ記録
                    self.logger.info(f"  Detection {i+1}: Label='{label}', Confidence={confidence:.3f}, BBox=({bbox.xmin():.3f},{bbox.ymin():.3f},{bbox.xmax():.3f},{bbox.ymax():.3f})")
                    
                    # 甲虫のみフィルタリング（クラス0のみ）
                    if label == BEETLE_CLASS_NAMES[BEETLE_CLASS_ID]:
                        beetle_detections.append({
                            'label': label,
                            'confidence': confidence,
                            'bbox': bbox
                        })
                        self.logger.info(f"  -> BEETLE DETECTED! Confidence: {confidence:.3f}")
            else:
                # 検出なしの場合は詳細ログのみ
                if self.frame_count % 150 == 0:
                    self.logger.info(f"Frame #{self.frame_count}: No detections found")
            
            # フレーム情報を保存（表示モード用）
            if self.args.display:
                # フレームサイズを取得
                format, width, height = get_caps_from_pad(pad)
                if format and width and height:
                    # フレームデータを取得
                    frame = get_numpy_from_buffer(buffer, format, width, height)
                    if frame is not None:
                        # バウンディングボックスを描画
                        frame_with_boxes = self.draw_detections(frame, beetle_detections, width, height)
                        self.frame_buffer = frame_with_boxes
                        
                        # フレーム情報をログ記録
                        if self.frame_count % 150 == 0:
                            self.logger.info(f"Frame #{self.frame_count}: Video frame {width}x{height}, format={format}")
                    
            # 検出結果の表示とログ記録
            if beetle_detections:
                self.beetle_count = len(beetle_detections)
                self.total_detections += self.beetle_count
                
                detection_msg = f"🪲 甲虫検出: {self.beetle_count}匹 (経過時間: {elapsed_time:.1f}秒)"
                print(detection_msg)
                self.logger.info(f"BEETLE SUCCESS: {self.beetle_count} beetles detected at {elapsed_time:.1f}s (Total: {self.total_detections})")
                
                for i, det in enumerate(beetle_detections):
                    detail_msg = f"  検出{i+1}: 信頼度 {det['confidence']:.2f}"
                    print(detail_msg)
                    bbox = det['bbox']
                    self.logger.info(f"  Beetle {i+1}: Confidence={det['confidence']:.3f}, BBox=({bbox.xmin():.3f},{bbox.ymin():.3f},{bbox.xmax():.3f},{bbox.ymax():.3f})")
            else:
                if not self.args.display:  # 表示モードでない場合のみ出力
                    if self.frame_count % 150 == 0:
                        print("甲虫は検出されていません")
                
        except Exception as e:
            error_msg = f"コールバックエラー: {e}"
            print(error_msg)
            self.logger.error(f"Frame #{self.frame_count}: Callback error - {e}")
            
        return Gst.PadProbeReturn.OK
        
    def bus_call(self, bus, message, loop):
        """GStreamer バスメッセージハンドラ"""
        message_type = message.type
        
        if message_type == Gst.MessageType.EOS:
            print("End of stream")
            loop.quit()
            
        elif message_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}, Debug: {debug}")
            loop.quit()
            
        elif message_type == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            print(f"Warning: {err}, Debug: {debug}")
            
        return True
    
    def setup_logging(self):
        """ログファイル設定を初期化"""
        # タイムスタンプでファイル名生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"detection_log_{timestamp}.txt"
        log_path = os.path.join("detection_logs", log_filename)
        
        # ログディレクトリが存在しない場合は作成
        os.makedirs("detection_logs", exist_ok=True)
        
        # ログ設定
        self.logger = logging.getLogger('beetle_detection')
        self.logger.setLevel(logging.INFO)
        
        # ファイルハンドラー
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # フォーマッター
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # ハンドラーを追加
        self.logger.addHandler(file_handler)
        
        # コンソールにもログファイルパスを出力
        print(f"📝 検出ログファイル: {log_path}")
        
    def draw_detections(self, frame, detections, width, height):
        """フレームにバウンディングボックスを描画"""
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            # バウンディングボックス座標（正規化座標から実座標に変換）
            x1 = int(bbox.xmin() * width)
            y1 = int(bbox.ymin() * height)
            x2 = int(bbox.xmax() * width)
            y2 = int(bbox.ymax() * height)
            
            # バウンディングボックスを描画（緑色）
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # ラベルと信頼度を描画
            label_text = f"Beetle: {confidence:.2f}"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            
            # ラベル背景を描画
            cv2.rectangle(frame_bgr, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            
            # ラベルテキストを描画
            cv2.putText(frame_bgr, label_text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        return frame_bgr
    
    def appsink_callback(self, appsink):
        """appsinkからフレームを取得してOpenCVで表示"""
        try:
            sample = appsink.emit('pull-sample')
            if sample:
                buffer = sample.get_buffer()
                caps = sample.get_caps()
                
                # フレームバッファがある場合は表示
                if self.frame_buffer is not None:
                    cv2.imshow("Beetle Detection", self.frame_buffer)
                    # ESCキーで終了
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESCキー
                        self.loop.quit()
                        
        except Exception as e:
            print(f"appsinkコールバックエラー: {e}")
            
        return Gst.FlowReturn.OK
        
    def run(self):
        """アプリケーションを実行"""
        # GStreamer初期化
        Gst.init(None)
        
        # パイプライン文字列を取得
        pipeline_string = self.get_pipeline_string()
        
        if self.args.verbose:
            print(f"HEFファイル: {self.args.hef_path}")
            print(f"入力ソース: {self.args.input}")
            print(f"デバイス: {self.args.device}")
            print("=" * 50)
            print("🪲 甲虫検出を開始します...")
            print("終了するには Ctrl+C を押してください")
            print("=" * 50)
            print(f"パイプライン: {pipeline_string}")
        
        try:
            # パイプラインを作成
            self.pipeline = Gst.parse_launch(pipeline_string)
            
            # GMainLoopを作成
            self.loop = GLib.MainLoop()
            
            # バス設定
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self.bus_call, self.loop)
            
            # hailonet要素からhailooutput padを取得してコールバックを設定
            hailonet = self.pipeline.get_by_name("hailonet0")
            if hailonet:
                # hailonetの出力パッドにプローブを設定
                pad = hailonet.get_static_pad("src")
                if pad:
                    pad.add_probe(
                        Gst.PadProbeType.BUFFER,
                        self.app_callback,
                        None
                    )
                    if self.args.verbose:
                        print("検出コールバックを設定しました")
            
            # 表示モードの場合、appsinkコールバックを設定
            if self.args.display:
                appsink = self.pipeline.get_by_name("appsink0")
                if appsink:
                    appsink.connect('new-sample', self.appsink_callback)
                    if self.args.verbose:
                        print("表示コールバックを設定しました")
            
            # パイプラインを開始
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                print("パイプラインの開始に失敗しました")
                return False
                
            # メインループを開始
            self.loop.run()
            
        except KeyboardInterrupt:
            print("\\n終了シグナルを受信しました。アプリケーションを終了します...")
            
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            return False
            
        finally:
            # 最終統計をログ記録
            final_time = time.time() - self.detection_start_time
            self.logger.info("=" * 60)
            self.logger.info(f"Session Summary:")
            self.logger.info(f"  Total Runtime: {final_time:.1f} seconds")
            self.logger.info(f"  Total Frames Processed: {self.frame_count}")
            self.logger.info(f"  Total Beetle Detections: {self.total_detections}")
            if self.frame_count > 0:
                self.logger.info(f"  Average FPS: {self.frame_count / final_time:.1f}")
            self.logger.info("Session Ended")
            self.logger.info("=" * 60)
            
            # クリーンアップ
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
            
            # OpenCVウィンドウを閉じる
            if self.args.display:
                cv2.destroyAllWindows()
                
        return True

def parse_arguments():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(
        description="Hailo 8L NPU 甲虫検出リアルタイム推論（修正版）"
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
        "--verbose", "-v",
        action="store_true",
        help="詳細ログを有効にする"
    )
    
    parser.add_argument(
        "--display", "-d",
        action="store_true",
        help="GUIウィンドウでリアルタイム検出表示を有効にする"
    )
    
    return parser.parse_args()

def main():
    """メイン関数"""
    # 引数解析
    args = parse_arguments()
    
    # HEFファイルの存在確認
    if not os.path.exists(args.hef_path):
        print(f"エラー: HEFファイルが見つかりません: {args.hef_path}")
        print("best.hef ファイルがカレントディレクトリにあることを確認してください。")
        return 1
    
    try:
        # 甲虫検出アプリケーション起動
        app = SimpleBeetleDetectionApp(args)
        success = app.run()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\\n終了シグナルを受信しました。アプリケーションを終了します...")
        return 0
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return 1

if __name__ == "__main__":
    exit(main())