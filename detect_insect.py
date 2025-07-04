#!/usr/bin/env python3
"""
YOLOv8を使用した昆虫検出アプリケーション

CPUベースの昆虫検出アプリケーション。画像をバッチ処理し、
包括的なログと共に可視化結果を出力します。

作成者: Generated with Claude Code
ライセンス: MIT
"""

# 標準ライブラリのインポート
import argparse     # コマンドライン引数解析
import csv          # CSVファイル処理
import logging      # ログ出力
import os           # OS機能
import sys          # システム機能
import time         # 時刻計測
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# 外部ライブラリのインポート
import cv2          # OpenCV コンピュータビジョン
import numpy as np  # 数値計算
from ultralytics import YOLO  # YOLOv8 モデル


def setup_logging(log_dir: Path) -> Tuple[logging.Logger, str]:
    """
    コンソールとファイル出力の両方に対してログ設定を初期化します。
    
    Args:
        log_dir: ログファイルを保存するディレクトリ
        
    Returns:
        ロガーインスタンスとCSVログファイル名のタプル
    """
    # ログディレクトリが存在しない場合は作成
    log_dir.mkdir(exist_ok=True)
    
    # コンソールログの設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)  # 標準出力へのログ出力
        ]
    )
    logger = logging.getLogger(__name__)
    
    # タイムスタンプ付きCSVログファイル名を作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"detection_log_{timestamp}.csv"
    csv_path = log_dir / csv_filename
    
    # CSVヘッダーを作成
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'detected', 'count', 'time_ms'])
    
    logger.info(f"ログ初期化完了。CSVログ: {csv_path}")
    return logger, str(csv_path)


def load_model(model_path: str, logger: logging.Logger) -> YOLO:
    """
    推論用のYOLOv8モデルを読み込みます。
    
    Args:
        model_path: モデル重みファイルのパス
        logger: ロガーインスタンス
        
    Returns:
        読み込まれたYOLOモデル
    """
    try:
        logger.info(f"YOLOv8モデルを読み込み中: {model_path}")
        model = YOLO(model_path)
        logger.info(f"モデルの読み込みが成功しました。クラス数: {len(model.names)}")
        return model
    except Exception as e:
        logger.error(f"モデルの読み込みに失敗しました: {e}")
        sys.exit(1)


def get_image_files(input_dir: Path, logger: logging.Logger) -> List[Path]:
    """
    入力ディレクトリからすべての有効な画像ファイルを取得します。
    
    Args:
        input_dir: 入力ディレクトリパス
        logger: ロガーインスタンス
        
    Returns:
        画像ファイルパスのリスト
    """
    # サポートする画像フォーマット
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    # 大文字・小文字両方の拡張子で検索
    for ext in valid_extensions:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    image_files.sort()  # ファイル名でソート
    logger.info(f"{input_dir} で {len(image_files)} 個の画像ファイルを発見")
    
    if not image_files:
        logger.warning(f"{input_dir} で画像ファイルが見つかりません")
    
    return image_files


def detect_objects(
    model: YOLO, 
    image_path: Path, 
    confidence_threshold: float = 0.25
) -> Tuple[np.ndarray, List[dict], bool]:
    """
    単一の画像に対して物体検出を実行します。
    
    Args:
        model: YOLOモデルインスタンス
        image_path: 入力画像のパス
        confidence_threshold: 検出の最低信頼度
        
    Returns:
        (注釈付き画像、検出結果リスト、検出有無)のタプル
    """
    # OpenCVを使用して画像ファイルを読み込み（BGR形式）
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"画像を読み込めません: {image_path}")
    
    # YOLOv8モデルで推論を実行
    # CPUデバイスを明示的に指定してGPUとの競合を回避
    results = model.predict(
        source=image,                    # 入力画像データ
        device='cpu',                    # 推論実行デバイス（CPU専用）
        conf=confidence_threshold,       # 信頼度閾値（この値以下は除外）
        verbose=False                    # 推論中の詳細ログを抑制
    )
    
    # 検出結果を格納するリストと検出フラグを初期化
    detections = []
    has_detections = False
    
    # YOLOv8の推論結果を解析（通常は1つの結果オブジェクト）
    for result in results:
        # バウンディングボックスと信頼度を描画した注釈付き画像を生成
        annotated_image = result.plot()
        
        # 検出されたオブジェクトがあるかチェック
        if result.boxes is not None and len(result.boxes) > 0:
            has_detections = True
            
            # 検出結果から各要素を抽出（GPUテンソルをCPU NumPy配列に変換）
            boxes = result.boxes.xyxy.cpu().numpy()        # バウンディングボックス座標 [x1,y1,x2,y2]
            classes = result.boxes.cls.cpu().numpy()       # 検出クラスのID番号
            confidences = result.boxes.conf.cpu().numpy()  # 各検出の信頼度スコア
            
            # 検出された各オブジェクトについて情報を整理
            for box, cls, conf in zip(boxes, classes, confidences):
                # バウンディングボックス座標を整数値に変換（ピクセル単位）
                x1, y1, x2, y2 = map(int, box)
                # クラスIDから実際のクラス名を取得（例: 0 -> "beetle"）
                class_name = model.names[int(cls)]
                
                # 検出情報を構造化して辞書に保存
                detection = {
                    'class': class_name,           # 検出されたオブジェクトのクラス名
                    'confidence': float(conf),     # 検出の信頼度（0.0～1.0）
                    'bbox': [x1, y1, x2, y2]       # バウンディングボックス座標
                }
                detections.append(detection)
        else:
            # 何も検出されなかった場合は元の画像をそのまま使用
            annotated_image = image
    
    return annotated_image, detections, has_detections


def save_result_image(
    annotated_image: np.ndarray, 
    output_path: Path, 
    logger: logging.Logger
) -> bool:
    """
    注釈付き画像を出力ディレクトリに保存します。
    
    Args:
        annotated_image: バウンディングボックスが描画された画像データ
        output_path: 出力ファイルのパス
        logger: ロガーインスタンス
        
    Returns:
        保存が成功した場合True、失敗した場合False
    """
    try:
        # OpenCVを使用して画像をPNG形式で保存
        # cv2.imwriteは自動的にファイル拡張子から形式を判断
        cv2.imwrite(str(output_path), annotated_image)
        return True
    except Exception as e:
        # ファイル保存エラーをログに記録
        logger.error(f"画像の保存に失敗しました {output_path}: {e}")
        return False


def log_detection_result(
    csv_path: str,
    filename: str,
    detected: bool,
    count: int,
    processing_time: float,
    logger: logging.Logger
) -> None:
    """
    検出結果をCSVファイルにログ出力します。
    
    Args:
        csv_path: CSVログファイルのパス
        filename: 画像ファイル名
        detected: オブジェクトが検出されたかどうか
        count: 検出数
        processing_time: 処理時間（ミリ秒）
        logger: ロガーインスタンス
    """
    try:
        # CSVファイルを追記モードで開き、UTF-8エンコーディングで書き込み
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # ファイル名、検出有無、検出数、処理時間をCSV行として書き込み
            writer.writerow([filename, detected, count, f"{processing_time:.1f}"])
    except Exception as e:
        # CSVファイル書き込みエラーをログに記録
        logger.error(f"CSVログの書き込みに失敗しました: {e}")


def process_images(
    model: YOLO,
    input_dir: Path,
    output_dir: Path,
    csv_log_path: str,
    logger: logging.Logger,
    confidence_threshold: float = 0.25
) -> None:
    """
    入力ディレクトリ内のすべての画像を処理します。
    
    Args:
        model: YOLOモデルインスタンス
        input_dir: 入力ディレクトリのパス
        output_dir: 出力ディレクトリのパス
        csv_log_path: CSVログファイルのパス
        logger: ロガーインスタンス
        confidence_threshold: 検出の最低信頼度
    """
    # 出力ディレクトリが存在しない場合は作成
    output_dir.mkdir(exist_ok=True)
    
    # 入力ディレクトリからすべての画像ファイルを取得
    image_files = get_image_files(input_dir, logger)
    
    # 処理対象画像がない場合は警告を出して終了
    if not image_files:
        logger.warning("処理対象の画像がありません")
        return
    
    # 処理統計用の変数を初期化
    total_images = len(image_files)          # 総画像数
    successful_processed = 0                 # 正常処理数
    total_detections = 0                     # 総検出数
    
    logger.info(f"{total_images} 枚の画像のバッチ処理を開始します...")
    
    # 各画像ファイルを順次処理
    for i, image_path in enumerate(image_files, 1):
        try:
            # 処理開始時刻を記録（処理時間計測用）
            start_time = time.time()
            
            logger.info(f"処理中 [{i}/{total_images}]: {image_path.name}")
            
            # YOLOv8モデルで物体検出を実行
            annotated_image, detections, has_detections = detect_objects(
                model, image_path, confidence_threshold
            )
            
            # 処理時間を計算（秒からミリ秒に変換）
            processing_time = (time.time() - start_time) * 1000
            
            # 出力ファイル名を生成（入力形式に関係なくPNGで保存）
            output_filename = image_path.stem + ".png"
            output_path = output_dir / output_filename
            
            # 注釈付き画像を保存
            if save_result_image(annotated_image, output_path, logger):
                # 正常処理カウンターを増加
                successful_processed += 1
                detection_count = len(detections)
                total_detections += detection_count
                
                # 処理結果をCSVログに記録
                log_detection_result(
                    csv_log_path,
                    image_path.name,
                    has_detections,
                    detection_count,
                    processing_time,
                    logger
                )
                
                # コンソールに処理結果を表示
                if has_detections:
                    # 検出されたクラス名のリストを作成（重複除去）
                    classes_detected = [d['class'] for d in detections]
                    logger.info(
                        f"✓ {detection_count} 個のオブジェクトを検出: "
                        f"{', '.join(set(classes_detected))} "
                        f"(処理時間: {processing_time:.1f}ms)"
                    )
                else:
                    logger.info(f"✓ オブジェクトは検出されませんでした (処理時間: {processing_time:.1f}ms)")
            
        except Exception as e:
            # 個別ファイルの処理エラーをログに記録（処理は継続）
            logger.error(f"{image_path.name} の処理に失敗しました: {e}")
            # 失敗した処理もCSVログに記録
            log_detection_result(
                csv_log_path,
                image_path.name,
                False,      # 検出なし
                0,          # 検出数0
                0.0,        # 処理時間0
                logger
            )
    
    # 処理完了後の統計情報を表示
    logger.info("=" * 60)
    logger.info("処理結果サマリー")
    logger.info("=" * 60)
    logger.info(f"総画像数: {total_images}")
    logger.info(f"正常処理数: {successful_processed}")
    logger.info(f"失敗数: {total_images - successful_processed}")
    logger.info(f"総検出数: {total_detections}")
    logger.info(f"CSVログ保存先: {csv_log_path}")
    logger.info("=" * 60)


def main():
    """昆虫検出アプリケーションを実行するメイン関数。"""
    parser = argparse.ArgumentParser(
        description="YOLOv8を使用した昆虫検出アプリケーション",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python detect_insect.py --input input_images/ --output output_images/
  python detect_insect.py --input input_images/ --output results/ --model yolov8s.pt
        """
    )
    
    # 必須引数
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='処理する画像を含む入力ディレクトリ'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='処理済み画像の出力ディレクトリ'
    )
    
    # オプション引数
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        help='YOLOv8モデル重みファイルのパス (デフォルト: yolov8n.pt)'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='検出の信頼度闾値 (デフォルト: 0.25)'
    )
    
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='ログファイル用ディレクトリ (デフォルト: logs)'
    )
    
    args = parser.parse_args()
    
    # コマンドライン引数をPathオブジェクトに変換（パス操作を簡潔にするため）
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    log_dir = Path(args.log_dir)
    
    # 入力ディレクトリの存在確認と検証
    if not input_dir.exists():
        print(f"エラー: 入力ディレクトリ '{input_dir}' が存在しません")
        sys.exit(1)
    
    if not input_dir.is_dir():
        print(f"エラー: '{input_dir}' はディレクトリではありません")
        sys.exit(1)
    
    # ロギングシステムを初期化（コンソール出力とCSVファイル出力を設定）
    logger, csv_log_path = setup_logging(log_dir)
    
    # 指定されたパスからYOLOv8モデルを読み込み
    model = load_model(args.model, logger)
    
    # モデル情報と処理パラメータをログに出力
    logger.info(f"検出可能クラス: {list(model.names.values())}")
    logger.info(f"入力ディレクトリ: {input_dir}")
    logger.info(f"出力ディレクトリ: {output_dir}")
    logger.info(f"信頼度閾値: {args.conf}")
    
    # メイン画像処理ループを実行
    try:
        process_images(
            model=model,                      # 読み込み済みYOLOモデル
            input_dir=input_dir,              # 処理対象画像の入力ディレクトリ
            output_dir=output_dir,            # 結果画像の出力ディレクトリ
            csv_log_path=csv_log_path,        # 処理ログのCSVファイルパス
            logger=logger,                    # ロガーインスタンス
            confidence_threshold=args.conf    # 検出の最低信頼度閾値
        )
        logger.info("すべての処理が正常に完了しました")
    except KeyboardInterrupt:
        # Ctrl+Cによる中断を適切に処理
        logger.info("ユーザーによって処理が中断されました")
        sys.exit(1)
    except Exception as e:
        # 予期しないエラーをログに記録して終了
        logger.error(f"処理中に予期しないエラーが発生しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()