#!/usr/bin/env python3

"""
Original Pipeline String Test
元のhailo_beetle_detection.pyのパイプライン文字列をテスト
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import time

def test_original_pipeline_string():
    """
    元のhailo_beetle_detection.pyと同じパイプライン文字列をテスト
    """
    # GStreamer初期化
    Gst.init(None)
    
    # 元のコードのパイプライン文字列（hailo_beetle_detection.pyから）
    pipeline_string = "v4l2src device=/dev/video0 name=src_0 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=640,height=640 ! videoflip video-direction=horiz ! queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! hailonet hef-path=best.hef batch-size=1 ! queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! fakesink "
    
    print("元のパイプライン文字列:")
    print(pipeline_string)
    print("=" * 80)
    
    try:
        # パイプラインを作成
        print("パイプライン作成中...")
        pipeline = Gst.parse_launch(pipeline_string)
        print("✅ パイプライン作成成功")
        
        # パイプラインを開始
        print("パイプライン開始中...")
        ret = pipeline.set_state(Gst.State.PLAYING)
        
        if ret == Gst.StateChangeReturn.FAILURE:
            print("❌ パイプライン開始失敗")
            return False
        else:
            print("✅ パイプライン開始成功")
            
        # 5秒間実行
        print("5秒間実行中...")
        time.sleep(5)
        
        # 停止
        pipeline.set_state(Gst.State.NULL)
        print("✅ パイプライン停止")
        
        return True
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

def test_working_pipeline_string():
    """
    動作することが確認されたパイプライン文字列をテスト
    """
    # GStreamer初期化
    Gst.init(None)
    
    # 動作するパイプライン文字列
    pipeline_string = "v4l2src device=/dev/video0 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=640,height=640 ! videoflip video-direction=horiz ! queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! hailonet hef-path=best.hef batch-size=1 ! queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! fakesink"
    
    print("動作するパイプライン文字列:")
    print(pipeline_string)
    print("=" * 80)
    
    try:
        # パイプラインを作成
        print("パイプライン作成中...")
        pipeline = Gst.parse_launch(pipeline_string)
        print("✅ パイプライン作成成功")
        
        # パイプラインを開始
        print("パイプライン開始中...")
        ret = pipeline.set_state(Gst.State.PLAYING)
        
        if ret == Gst.StateChangeReturn.FAILURE:
            print("❌ パイプライン開始失敗")
            return False
        else:
            print("✅ パイプライン開始成功")
            
        # 3秒間実行
        print("3秒間実行中...")
        time.sleep(3)
        
        # 停止
        pipeline.set_state(Gst.State.NULL)
        print("✅ パイプライン停止")
        
        return True
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

if __name__ == "__main__":
    print("Pipeline String Comparison Test")
    print("=" * 80)
    
    print("\n1. 元のパイプライン文字列テスト:")
    result1 = test_original_pipeline_string()
    
    print("\n2. 動作するパイプライン文字列テスト:")
    result2 = test_working_pipeline_string()
    
    print("\n結果:")
    print(f"元のパイプライン: {'成功' if result1 else '失敗'}")
    print(f"動作するパイプライン: {'成功' if result2 else '失敗'}")
    
    if result1 and result2:
        print("\n🎉 両方とも成功！問題は別の場所にあります")
    elif not result1 and result2:
        print("\n🔍 元のパイプライン文字列に問題があります")
    else:
        print("\n💥 何か他の問題があります")