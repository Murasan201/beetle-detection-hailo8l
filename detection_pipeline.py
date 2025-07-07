#!/usr/bin/env python3

"""
GStreamer Detection Pipeline Base Class
Hailo 8L NPU用の検出パイプラインベースクラス

Based on Hailo RPi Examples
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import time
import threading
from hailo_rpi_common import app_callback_class

class GStreamerDetectionApp:
    """
    GStreamer検出アプリケーションのベースクラス
    """
    
    def __init__(self, args, user_data):
        self.args = args
        self.user_data = user_data
        self.pipeline = None
        self.loop = None
        self.fps_counter = app_callback_class()
        self.running = False
        
        # パイプライン文字列を取得
        self.pipeline_string = self.get_pipeline_string()
        
    def get_pipeline_string(self):
        """
        パイプライン文字列を取得（サブクラスでオーバーライド）
        """
        raise NotImplementedError("Subclass must implement get_pipeline_string()")
        
    def app_callback(self, pad, info, user_data):
        """
        アプリケーションコールバック（サブクラスでオーバーライド）
        """
        raise NotImplementedError("Subclass must implement app_callback()")
        
    def bus_call(self, bus, message, loop):
        """
        GStreamer バスメッセージハンドラ
        """
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
            
        elif message_type == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old_state, new_state, pending_state = message.parse_state_changed()
                if self.args.verbose:
                    print(f"Pipeline state changed from {old_state.value_nick} to {new_state.value_nick}")
                    
        return True
        
    def create_pipeline(self):
        """
        GStreamerパイプラインを作成
        """
        try:
            # パイプラインを作成
            self.pipeline = Gst.parse_launch(self.pipeline_string)
            
            # バス設定
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self.bus_call, self.loop)
            
            # パイプライン要素を取得
            self.setup_pipeline_elements()
            
            # コールバック設定
            self.setup_callbacks()
            
            return True
            
        except Exception as e:
            print(f"パイプライン作成エラー: {e}")
            return False
            
    def setup_pipeline_elements(self):
        """
        パイプライン要素の設定
        """
        # hailonet要素の設定
        hailonet = self.pipeline.get_by_name("hailonet0")
        if hailonet:
            hailonet.set_property("batch-size", 1)
            if self.args.verbose:
                print("hailonet 要素を設定しました")
                
        # hailofilter要素の設定
        hailofilter = self.pipeline.get_by_name("hailofilter0")
        if hailofilter and self.args.verbose:
            print("hailofilter 要素を設定しました")
            
        # 表示要素の設定
        display_sink = self.pipeline.get_by_name("hailo_display")
        if display_sink:
            display_sink.set_property("sync", False)
            if self.args.verbose:
                print("表示要素を設定しました")
                
    def setup_callbacks(self):
        """
        コールバック関数の設定
        """
        # hailofilter要素のパッドにプローブを設定
        hailofilter = self.pipeline.get_by_name("hailofilter0")
        if hailofilter:
            pad = hailofilter.get_static_pad("src")
            if pad:
                pad.add_probe(
                    Gst.PadProbeType.BUFFER,
                    self.app_callback,
                    self.user_data
                )
                if self.args.verbose:
                    print("コールバック関数を設定しました")
                    
    def get_fps(self):
        """
        現在のFPSを取得
        """
        return self.fps_counter.get_fps()
        
    def start_pipeline(self):
        """
        パイプラインを開始
        """
        if not self.pipeline:
            return False
            
        # パイプラインを開始
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("パイプラインの開始に失敗しました")
            return False
            
        self.running = True
        if self.args.verbose:
            print("パイプラインを開始しました")
            
        return True
        
    def stop_pipeline(self):
        """
        パイプラインを停止
        """
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            self.running = False
            if self.args.verbose:
                print("パイプラインを停止しました")
                
    def run(self):
        """
        アプリケーションを実行
        """
        # GMainLoopを作成
        self.loop = GLib.MainLoop()
        
        # パイプラインを作成
        if not self.create_pipeline():
            print("パイプラインの作成に失敗しました")
            return False
            
        # パイプラインを開始
        if not self.start_pipeline():
            print("パイプラインの開始に失敗しました")
            return False
            
        try:
            # メインループを開始
            print("アプリケーションを実行中...")
            self.loop.run()
            
        except KeyboardInterrupt:
            print("\nキーボード割り込みを受信しました")
            
        finally:
            # クリーンアップ
            self.stop_pipeline()
            
        return True
        
    def get_pipeline_string(self):
        """
        デフォルトのパイプライン文字列
        """
        # HEFファイルパス
        hef_path = getattr(self.args, 'hef_path', 'best.hef')
        
        # 基本的なパイプライン
        pipeline_string = f"""
        videotestsrc pattern=0 ! 
        video/x-raw, format=RGB, width=640, height=480, framerate=30/1 ! 
        queue ! 
        hailonet hef-path={hef_path} batch-size=1 name=hailonet0 ! 
        queue ! 
        hailofilter so-path=/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgsthailo_filter.so name=hailofilter0 ! 
        queue ! 
        hailooverlay ! 
        queue ! 
        videoconvert ! 
        fpsdisplaysink video-sink=autovideosink name=hailo_display sync=false text-overlay=false
        """
        
        return pipeline_string.strip()

class SimpleBeetleDetectionApp(GStreamerDetectionApp):
    """
    シンプルな甲虫検出アプリケーション
    """
    
    def __init__(self, args, user_data):
        super().__init__(args, user_data)
        
    def app_callback(self, pad, info, user_data):
        """
        甲虫検出用のコールバック
        """
        # FPSカウンタを更新
        self.fps_counter.update_fps()
        
        # バッファを取得
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
            
        # 検出結果を処理
        # この部分は実際のHailo推論結果に応じて実装
        
        return Gst.PadProbeReturn.OK
        
    def get_pipeline_string(self):
        """
        甲虫検出用のパイプライン文字列
        """
        hef_path = getattr(self.args, 'hef_path', 'best.hef')
        
        if self.args.input == 'usb':
            # USBカメラ
            source = f"v4l2src device={getattr(self.args, 'device', '/dev/video0')} ! "
            source += "video/x-raw, format=YUY2, width=640, height=480, framerate=30/1 ! "
            source += "videoconvert ! "
            source += "video/x-raw, format=RGB ! "
            
        elif self.args.input == 'rpi':
            # Raspberry Pi カメラ
            source = "libcamerasrc ! "
            source += "video/x-raw, format=RGB, width=640, height=480, framerate=30/1 ! "
            
        else:
            # テストパターン
            source = "videotestsrc pattern=0 ! "
            source += "video/x-raw, format=RGB, width=640, height=480, framerate=30/1 ! "
            
        pipeline_string = source
        pipeline_string += "queue ! "
        pipeline_string += f"hailonet hef-path={hef_path} batch-size=1 name=hailonet0 ! "
        pipeline_string += "queue ! "
        pipeline_string += "hailofilter so-path=/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgsthailo_filter.so name=hailofilter0 ! "
        pipeline_string += "queue ! "
        pipeline_string += "hailooverlay ! "
        pipeline_string += "queue ! "
        pipeline_string += "videoconvert ! "
        pipeline_string += "fpsdisplaysink video-sink=autovideosink name=hailo_display sync=false text-overlay=false"
        
        return pipeline_string