#!/usr/bin/env python3

"""
Simple GStreamer Pipeline Test
最小構成でGst.parse_launch()をテスト
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import time

def test_simple_pipeline():
    """
    動作するコマンドラインと同じパイプラインをPythonでテスト
    """
    # GStreamer初期化
    Gst.init(None)
    
    # 動作することが確認されたパイプライン文字列
    pipeline_string = """
    v4l2src device=/dev/video0 !
    videoconvert !
    videoscale !
    video/x-raw,format=RGB,width=640,height=640 !
    videoflip video-direction=horiz !
    queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 !
    hailonet hef-path=best.hef batch-size=1 !
    queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 !
    fakesink
    """
    
    print("パイプライン文字列:")
    print(pipeline_string)
    print("=" * 50)
    
    try:
        # パイプラインを作成
        print("パイプラインを作成中...")
        pipeline = Gst.parse_launch(pipeline_string)
        print("✅ パイプライン作成成功")
        
        # パイプラインを開始
        print("パイプラインを開始中...")
        ret = pipeline.set_state(Gst.State.PLAYING)
        
        if ret == Gst.StateChangeReturn.FAILURE:
            print("❌ パイプラインの開始に失敗")
            return False
        else:
            print("✅ パイプライン開始成功")
            
        # 5秒間実行
        print("5秒間実行中...")
        time.sleep(5)
        
        # パイプラインを停止
        print("パイプラインを停止中...")
        pipeline.set_state(Gst.State.NULL)
        print("✅ パイプライン停止成功")
        
        return True
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

if __name__ == "__main__":
    print("Simple GStreamer Pipeline Test")
    print("=" * 50)
    
    if test_simple_pipeline():
        print("\n🎉 テスト成功！")
    else:
        print("\n💥 テスト失敗")