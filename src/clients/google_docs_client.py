from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pytz
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    title: str
    url: str
    published_jst: datetime
    body: str
    category: str = "Google Docs"
    country: str = "JP"

class GoogleDocsClient:
    """Google Docs APIを使用してニュース記事を取得するクライアント"""
    
    SCOPES = ['https://www.googleapis.com/auth/documents.readonly']
    
    def __init__(self, credentials_path: str = None, token_path: str = None):
        """
        Google Docs クライアントを初期化
        
        Args:
            credentials_path: Google Cloud Platformの認証情報ファイルパス
            token_path: 保存されたトークンファイルパス
        """
        self.credentials_path = credentials_path or 'credentials.json'
        self.token_path = token_path or 'token.json'
        self.service = None
        self._authenticate()
    
    def _authenticate(self) -> None:
        """Google API認証を実行"""
        if not os.path.exists(self.credentials_path):
            logger.warning(f"認証情報ファイルが見つかりません: {self.credentials_path}")
            logger.info("公開ドキュメントとしてアクセスを試行します")
            self.service = None
            return
        
        try:
            # サービスアカウント認証を試行
            creds = ServiceAccountCredentials.from_service_account_file(
                self.credentials_path, scopes=self.SCOPES)
            self.service = build('docs', 'v1', credentials=creds)
            logger.info("サービスアカウント認証に成功しました")
            return
            
        except Exception as sa_error:
            logger.warning(f"サービスアカウント認証失敗: {sa_error}")
            
            # OAuth認証にフォールバック
            try:
                creds = None
                
                # 既存のトークンファイルを確認
                if os.path.exists(self.token_path):
                    creds = Credentials.from_authorized_user_file(self.token_path, self.SCOPES)
                
                # 認証情報が無効または存在しない場合
                if not creds or not creds.valid:
                    if creds and creds.expired and creds.refresh_token:
                        creds.refresh(Request())
                    else:
                        flow = InstalledAppFlow.from_client_secrets_file(
                            self.credentials_path, self.SCOPES)
                        creds = flow.run_local_server(port=0)
                    
                    # トークンを保存
                    with open(self.token_path, 'w') as token:
                        token.write(creds.to_json())
                
                self.service = build('docs', 'v1', credentials=creds)
                logger.info("OAuth認証に成功しました")
                
            except Exception as oauth_error:
                logger.error(f"OAuth認証も失敗: {oauth_error}")
                logger.info("公開ドキュメントとしてアクセスを試行します")
                self.service = None
    
    def get_document_content(self, document_id: str) -> str:
        """
        Googleドキュメントの内容を取得
        
        Args:
            document_id: GoogleドキュメントのID
            
        Returns:
            ドキュメントのテキスト内容
        """
        try:
            # 最初にGoogle Docs APIを使用
            if self.service:
                logger.info("Google Docs APIでドキュメントを取得中...")
                document = self.service.documents().get(documentId=document_id).execute()
                content = ""
                
                for element in document.get('body', {}).get('content', []):
                    if 'paragraph' in element:
                        paragraph = element['paragraph']
                        for text_run in paragraph.get('elements', []):
                            if 'textRun' in text_run:
                                content += text_run['textRun']['content']
                
                logger.info(f"Google Docs APIでテキストを取得しました（文字数: {len(content)}）")
                return content
            else:
                raise Exception("Google Docs API service not available")
                
        except HttpError as error:
            logger.warning(f"Google Docs API エラー: {error}")
            
            # フォールバック: 公開ドキュメントとしてプレーンテキストで取得を試行
            try:
                import requests
                
                # Google Docs の公開エクスポートURLを試行
                export_url = f"https://docs.google.com/document/d/{document_id}/export?format=txt"
                
                response = requests.get(export_url)
                response.raise_for_status()
                
                content = response.text
                logger.info(f"公開ドキュメントとしてテキストを取得しました（文字数: {len(content)}）")
                return content
                
            except requests.exceptions.HTTPError as e:
                logger.error(f"公開ドキュメントアクセスも失敗: {e}")
                raise
        
        except Exception as error:
            logger.error(f"ドキュメント取得エラー: {error}")
            raise
    
    def parse_news_content(self, content: str, hours_limit: int = 24) -> List[NewsArticle]:
        """
        Googleドキュメントの内容をニュース記事形式にパース
        
        Args:
            content: ドキュメントのテキスト内容
            hours_limit: 何時間以内の記事を対象とするか
            
        Returns:
            パースされたニュース記事のリスト
        """
        articles = []
        lines = content.strip().split('\n')
        
        # 1行目の更新時刻を取得（参考用）
        if lines:
            first_line = lines[0].strip()
            logger.info(f"ドキュメント更新時刻: {first_line}")
        
        current_time = datetime.now(pytz.timezone('Asia/Tokyo'))
        cutoff_time = current_time - timedelta(hours=hours_limit)
        
        i = 1  # 1行目は更新時刻なのでスキップ
        while i < len(lines):
            line = lines[i].strip()
            
            # 記事エントリの開始を検出 (YYYY-MM-DD HH:MM) タイトル の形式
            time_title_match = re.match(r'^\((\d{4}-\d{2}-\d{2} \d{2}:\d{2})\)\s*(.+)$', line)
            if time_title_match:
                time_str = time_title_match.group(1)
                title = time_title_match.group(2).strip()
                
                # 時刻をパース
                try:
                    published_time = self._parse_time(time_str, current_time)
                except ValueError as e:
                    logger.warning(f"時刻パースエラー: {time_str}, エラー: {e}")
                    i += 1
                    continue
                
                # 時間制限チェック
                if published_time < cutoff_time:
                    logger.debug(f"時間制限外の記事をスキップ: {title}")
                    i += 1
                    continue
                
                # 次の行でURLを取得
                url = ""
                body = ""
                
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line.startswith('http'):
                        url = next_line
                        i += 2  # タイトル行とURL行をスキップ
                        
                        # 記事本文を取得（次の記事まで）
                        body_lines = []
                        while i < len(lines):
                            body_line = lines[i].strip()
                            
                            # 次の記事の開始を検出または区切り線を検出
                            if re.match(r'^\(\d{4}-\d{2}-\d{2} \d{2}:\d{2}\)', body_line) or body_line.startswith('--'):
                                break
                            
                            if body_line:  # 空行は無視
                                body_lines.append(body_line)
                            i += 1
                        
                        body = '\n'.join(body_lines)
                        
                        # 記事オブジェクトを作成
                        if title and body:
                            article = NewsArticle(
                                title=title,
                                url=url,
                                published_jst=published_time,
                                body=body
                            )
                            articles.append(article)
                            logger.debug(f"記事を追加: {title[:50]}...")
                        
                        continue  # while ループの次の反復へ
                
            i += 1
        
        logger.info(f"Google Docsから {len(articles)} 件の記事を取得しました")
        return articles
    
    def _parse_time(self, time_str: str, reference_time: datetime) -> datetime:
        """
        時刻文字列をdatetimeオブジェクトに変換
        
        Args:
            time_str: 時刻文字列（例: "2025-07-14 23:11", "14:30", "昨日 15:45"）
            reference_time: 基準時刻
            
        Returns:
            パースされたdatetime（JST）
        """
        jst = pytz.timezone('Asia/Tokyo')
        
        # フル日時形式 YYYY-MM-DD HH:MM
        full_datetime_match = re.match(r'(\d{4})-(\d{2})-(\d{2}) (\d{1,2}):(\d{2})', time_str)
        if full_datetime_match:
            year = int(full_datetime_match.group(1))
            month = int(full_datetime_match.group(2))
            day = int(full_datetime_match.group(3))
            hour = int(full_datetime_match.group(4))
            minute = int(full_datetime_match.group(5))
            return jst.localize(datetime(year, month, day, hour, minute))
        
        # "昨日" を含む場合
        if '昨日' in time_str:
            time_part = re.search(r'(\d{1,2}):(\d{2})', time_str)
            if time_part:
                hour = int(time_part.group(1))
                minute = int(time_part.group(2))
                yesterday = reference_time.date() - timedelta(days=1)
                return jst.localize(datetime.combine(yesterday, datetime.min.time().replace(hour=hour, minute=minute)))
        
        # "今日" を含む場合または時刻のみの場合
        time_match = re.search(r'(\d{1,2}):(\d{2})', time_str)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2))
            today = reference_time.date()
            return jst.localize(datetime.combine(today, datetime.min.time().replace(hour=hour, minute=minute)))
        
        # その他の形式
        raise ValueError(f"サポートされていない時刻形式: {time_str}")
    
    def fetch_news_articles(self, document_id: str, hours_limit: int = 24) -> List[Dict]:
        """
        Googleドキュメントからニュース記事を取得し、DataFetcher互換形式で返す
        
        Args:
            document_id: GoogleドキュメントのID
            hours_limit: 何時間以内の記事を対象とするか
            
        Returns:
            DataFetcher互換形式のニュース記事リスト
        """
        try:
            content = self.get_document_content(document_id)
            articles = self.parse_news_content(content, hours_limit)
            
            # DataFetcher互換形式に変換
            news_data = []
            for article in articles:
                news_data.append({
                    'title': article.title,
                    'url': article.url,
                    'published_jst': article.published_jst,
                    'category': article.category,
                    'country': article.country,
                    'body': article.body
                })
            
            return news_data
            
        except Exception as e:
            logger.error(f"Google Docsからのニュース取得エラー: {e}")
            return []

# テスト用のメイン関数
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    logging.basicConfig(level=logging.INFO)
    
    client = GoogleDocsClient(credentials_path='service_account.json')
    document_id = os.getenv("GOOGLE_DOCS_ID")
    
    if not document_id:
        raise ValueError("環境変数 GOOGLE_DOCS_ID が設定されていません。")

    # デバッグ: ドキュメントの最初の部分を確認
    content = client.get_document_content(document_id)
    print("=== ドキュメント内容の最初の1000文字 ===")
    print(repr(content[:1000]))
    print("\n=== 改行で分割した最初の10行 ===")
    lines = content.split('\n')
    for i, line in enumerate(lines[:10]):
        print(f"{i+1}: {repr(line)}")
    
    articles = client.fetch_news_articles(document_id)
    print(f"\n取得した記事数: {len(articles)}")
    
    for i, article in enumerate(articles[:3], 1):
        print(f"\n--- 記事 {i} ---")
        print(f"タイトル: {article['title']}")
        print(f"URL: {article['url']}")
        print(f"時刻: {article['published_jst']}")
        print(f"本文: {article['body'][:100]}...")
