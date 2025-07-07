#!/usr/bin/env python3

"""
Hailo Raspberry Pi Common Utilities
GStreamer パイプライン用の共通ユーティリティ関数

Based on Hailo RPi Examples
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject
import numpy as np
import cv2
from typing import Tuple, Optional

def get_caps_from_pad(pad):
    """
    GStreamer パッドからキャップス情報を取得
    
    Args:
        pad: GStreamer パッド
        
    Returns:
        tuple: (format, width, height)
    """
    caps = pad.get_current_caps()
    if not caps:
        return None, None, None
        
    structure = caps.get_structure(0)
    if not structure:
        return None, None, None
        
    format_str = structure.get_value('format')
    width = structure.get_value('width')
    height = structure.get_value('height')
    
    return format_str, width, height

def get_numpy_from_buffer(buffer, format_str: str, width: int, height: int) -> Optional[np.ndarray]:
    """
    GStreamer バッファから NumPy 配列を取得
    
    Args:
        buffer: GStreamer バッファ
        format_str: ピクセルフォーマット
        width: 画像幅
        height: 画像高さ
        
    Returns:
        numpy.ndarray: 画像データ
    """
    if not buffer:
        return None
        
    # バッファからメモリを取得
    success, map_info = buffer.map(Gst.MapFlags.READ)
    if not success:
        return None
        
    try:
        # NumPy配列に変換
        if format_str == 'RGB':
            # RGB format
            frame = np.frombuffer(map_info.data, dtype=np.uint8)
            frame = frame.reshape((height, width, 3))
        elif format_str == 'YUY2':
            # YUY2 format
            frame = np.frombuffer(map_info.data, dtype=np.uint8)
            frame = frame.reshape((height, width * 2))
            # YUY2からRGBに変換
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_YUY2)
        else:
            # その他のフォーマット（グレースケールなど）
            frame = np.frombuffer(map_info.data, dtype=np.uint8)
            frame = frame.reshape((height, width))
            
        return frame
        
    finally:
        buffer.unmap(map_info)

class app_callback_class:
    """
    アプリケーションコールバック用のベースクラス
    """
    
    def __init__(self):
        self.frame_count = 0
        self.fps_counter = FPSCounter()
        
    def increment_frame_count(self):
        """フレームカウンタを増加"""
        self.frame_count += 1
        
    def get_frame_count(self) -> int:
        """現在のフレーム数を取得"""
        return self.frame_count
        
    def get_fps(self) -> float:
        """現在のFPSを取得"""
        return self.fps_counter.get_fps()
        
    def update_fps(self):
        """FPSカウンタを更新"""
        self.fps_counter.update()

class FPSCounter:
    """
    FPS計算用のカウンタクラス
    """
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = []
        self.last_time = None
        
    def update(self):
        """フレーム時間を更新"""
        import time
        current_time = time.time()
        
        if self.last_time is not None:
            frame_time = current_time - self.last_time
            self.frame_times.append(frame_time)
            
            # ウィンドウサイズを超えた場合は古いデータを削除
            if len(self.frame_times) > self.window_size:
                self.frame_times.pop(0)
                
        self.last_time = current_time
        
    def get_fps(self) -> float:
        """現在のFPSを計算"""
        if len(self.frame_times) < 2:
            return 0.0
            
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        if avg_frame_time <= 0:
            return 0.0
            
        return 1.0 / avg_frame_time
        
    def reset(self):
        """カウンタをリセット"""
        self.frame_times = []
        self.last_time = None

def gst_buffer_with_caps_to_numpy(buffer, caps):
    """
    キャップス付きGStreamerバッファをNumPy配列に変換
    
    Args:
        buffer: GStreamer バッファ
        caps: GStreamer キャップス
        
    Returns:
        numpy.ndarray: 変換された画像データ
    """
    structure = caps.get_structure(0)
    width = structure.get_value('width')
    height = structure.get_value('height')
    format_str = structure.get_value('format')
    
    return get_numpy_from_buffer(buffer, format_str, width, height)

def print_gst_caps(caps):
    """
    GStreamer キャップスの情報を表示
    
    Args:
        caps: GStreamer キャップス
    """
    if not caps:
        print("Caps: None")
        return
        
    structure = caps.get_structure(0)
    if not structure:
        print("Caps: No structure")
        return
        
    print(f"Caps: {structure.get_name()}")
    print(f"  Format: {structure.get_value('format')}")
    print(f"  Width: {structure.get_value('width')}")
    print(f"  Height: {structure.get_value('height')}")
    print(f"  Framerate: {structure.get_value('framerate')}")