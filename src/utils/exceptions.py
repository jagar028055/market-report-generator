"""
カスタム例外クラス定義
"""

class MarketReportException(Exception):
    """マーケットレポート生成関連の基底例外クラス"""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self):
        error_msg = super().__str__()
        if self.error_code:
            error_msg = f"[{self.error_code}] {error_msg}"
        return error_msg


class DataFetchError(MarketReportException):
    """データ取得関連のエラー"""
    pass


class MarketDataError(DataFetchError):
    """市場データ取得エラー"""
    pass


class NewsDataError(DataFetchError):
    """ニュースデータ取得エラー"""
    pass


class EconomicDataError(DataFetchError):
    """経済指標データ取得エラー"""
    pass


class ChartGenerationError(MarketReportException):
    """チャート生成関連のエラー"""
    pass


class ConfigurationError(MarketReportException):
    """設定関連のエラー"""
    pass


class ValidationError(MarketReportException):
    """データ検証エラー"""
    pass


class NetworkError(MarketReportException):
    """ネットワーク関連のエラー"""
    pass


class TimeoutError(MarketReportException):
    """タイムアウトエラー"""
    pass


class APIError(MarketReportException):
    """API関連のエラー"""
    
    def __init__(self, message: str, status_code: int = None, response_data: dict = None, **kwargs):
        super().__init__(message, **kwargs)
        self.status_code = status_code
        self.response_data = response_data or {}


class FileOperationError(MarketReportException):
    """ファイル操作関連のエラー"""
    pass


class TemplateError(MarketReportException):
    """テンプレート処理関連のエラー"""
    pass


class WebScrapingError(MarketReportException):
    """Webスクレイピング関連のエラー"""
    pass


class AuthenticationError(MarketReportException):
    """認証エラー"""
    pass


class RateLimitError(MarketReportException):
    """レート制限エラー"""
    
    def __init__(self, message: str, retry_after: int = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class DataProcessingError(MarketReportException):
    """データ処理関連のエラー"""
    pass


class ChartConfigurationError(ChartGenerationError):
    """チャート設定関連のエラー"""
    pass


class FontError(ChartGenerationError):
    """フォント関連のエラー"""
    pass


class ReportGenerationError(MarketReportException):
    """レポート生成関連のエラー"""
    pass


class ResourceError(MarketReportException):
    """リソース関連のエラー（メモリ、ディスク等）"""
    pass


class DatabaseError(MarketReportException):
    """データベース関連のエラー"""
    pass


class CacheError(MarketReportException):
    """キャッシュ関連のエラー"""
    pass


class BackupError(MarketReportException):
    """バックアップ関連のエラー"""
    pass


class UnexpectedResponseError(APIError):
    """予期しないレスポンスエラー"""
    pass


class ServiceUnavailableError(MarketReportException):
    """サービス利用不可エラー"""
    pass


class InsufficientDataError(DataFetchError):
    """データ不足エラー"""
    pass


class InvalidDataFormatError(ValidationError):
    """データフォーマット不正エラー"""
    pass


class DependencyError(MarketReportException):
    """依存関係エラー"""
    pass