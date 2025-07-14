#!/usr/bin/env python3
"""
チャート更新用の簡易Webサーバー
"""

import json
import subprocess
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import os
from pathlib import Path

class ChartUpdateHandler(BaseHTTPRequestHandler):
    
    def do_POST(self):
        """POST リクエストを処理"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/regenerate_charts':
            self._handle_regenerate_charts()
        else:
            self._send_error(404, "Not Found")
    
    def do_OPTIONS(self):
        """CORSプリフライトリクエストを処理"""
        self._send_cors_headers()
        self.end_headers()
    
    def _handle_regenerate_charts(self):
        """チャート再生成リクエストを処理"""
        try:
            # リクエストボディを読み取り
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            # JSONデータを解析
            chart_settings = json.loads(post_data.decode('utf-8'))
            
            # update_charts.pyスクリプトを実行
            script_path = Path(__file__).parent / "update_charts.py"
            
            # Pythonスクリプトを実行
            result = subprocess.run([
                sys.executable, str(script_path), 
                json.dumps(chart_settings)
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                # 成功
                response_data = {
                    "success": True,
                    "message": "チャートが正常に更新されました",
                    "settings": chart_settings,
                    "output": result.stdout
                }
                self._send_json_response(200, response_data)
            else:
                # エラー
                response_data = {
                    "success": False,
                    "message": "チャート更新中にエラーが発生しました",
                    "error": result.stderr,
                    "output": result.stdout
                }
                self._send_json_response(500, response_data)
                
        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON")
        except subprocess.TimeoutExpired:
            self._send_error(504, "Request timeout")
        except Exception as e:
            self._send_error(500, f"Internal server error: {str(e)}")
    
    def _send_json_response(self, status_code, data):
        """JSON レスポンスを送信"""
        response_json = json.dumps(data, ensure_ascii=False, indent=2)
        
        self.send_response(status_code)
        self._send_cors_headers()
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(response_json.encode('utf-8'))))
        self.end_headers()
        
        self.wfile.write(response_json.encode('utf-8'))
    
    def _send_cors_headers(self):
        """CORSヘッダーを送信"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
    
    def _send_error(self, status_code, message):
        """エラーレスポンスを送信"""
        error_data = {
            "success": False,
            "error": message
        }
        self._send_json_response(status_code, error_data)
    
    def log_message(self, format, *args):
        """ログメッセージを出力"""
        print(f"[{self.date_time_string()}] {format % args}")

def run_server(port=8000):
    """サーバーを起動"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, ChartUpdateHandler)
    
    print(f"チャート更新サーバーを起動しています...")
    print(f"URL: http://localhost:{port}")
    print("Ctrl+C で停止")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nサーバーを停止しています...")
        httpd.shutdown()

if __name__ == "__main__":
    # ポート番号をコマンドライン引数から取得
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("無効なポート番号です。デフォルトの8000を使用します。")
    
    run_server(port)