"""
エラーハンドリングユーティリティ
"""
import logging
import traceback
from functools import wraps
from typing import Callable, Any, Optional

class MarketReportError(Exception):
    """マーケットレポート生成に関するカスタム例外"""
    pass

class DataFetchError(MarketReportError):
    """データ取得エラー"""
    pass

class ChartGenerationError(MarketReportError):
    """チャート生成エラー"""
    pass

class CommentaryGenerationError(MarketReportError):
    """コメント生成エラー"""
    pass

class ReportGenerationError(MarketReportError):
    """レポート生成エラー"""
    pass

def handle_step_error(step_name: str, error_class: type = MarketReportError):
    """
    処理ステップのエラーハンドリングデコレータ
    
    Args:
        step_name: 処理ステップ名
        error_class: 発生させるエラークラス
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_message = f"{step_name}でエラーが発生しました: {str(e)}"
                logging.error(error_message)
                logging.error(traceback.format_exc())
                raise error_class(error_message) from e
        return wrapper
    return decorator

def log_and_reraise(error: Exception, step_name: str, logger: Optional[logging.Logger] = None):
    """
    エラーをログに記録して再発生させる
    
    Args:
        error: 発生したエラー
        step_name: 処理ステップ名
        logger: ロガー（指定されない場合はデフォルトロガーを使用）
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    error_message = f"{step_name}でエラーが発生しました: {str(error)}"
    logger.error(error_message)
    logger.error(traceback.format_exc())
    raise

def create_error_summary(errors: list) -> str:
    """
    エラーリストからサマリーを作成
    
    Args:
        errors: エラーのリスト
        
    Returns:
        エラーサマリー文字列
    """
    if not errors:
        return "エラーはありません"
    
    summary = "以下のエラーが発生しました:\n"
    for i, error in enumerate(errors, 1):
        summary += f"{i}. {str(error)}\n"
    
    return summary