# Market Report Generator - 新UI実装完了

## 概要

`feature/enhanced-ui-individual-stocks`ブランチで、高度な分析機能（予測・リスク分析）を活用する新しいWebUI が完全実装されました。

## 実装された機能

### 🎯 サマリー + 詳細ページ形式UI
- **サマリーページ**: 主要指数、AIコメント、銘柄分析の概要
- **予測分析ページ**: 複数モデル予測結果の比較・可視化
- **リスク分析ページ**: VaR・CVaR・モンテカルロシミュレーション

### 🔧 技術仕様
- **バックエンド**: Flask + 既存analytics modules
- **フロントエンド**: Bootstrap 5 + Plotly.js
- **API**: RESTful endpoints (`/api/summary`, `/api/predict`, `/api/risk`)
- **フォールバック**: Analytics modules利用不可時の代替実装

## 🚀 使用方法

### 1. 環境準備
```bash
# 依存関係インストール
pip install -r requirements.txt

# 環境変数設定（必要に応じて）
export OPENAI_API_KEY="your-api-key"
export ALPHA_VANTAGE_API_KEY="your-api-key"
```

### 2. Webサーバー起動
```bash
# デフォルトポート5000で起動
python main.py server

# カスタムポートで起動
python main.py server 8080
```

### 3. アクセス
ブラウザで以下のURLにアクセス:
- **サマリー**: http://localhost:5000/
- **予測分析**: http://localhost:5000/predict
- **リスク分析**: http://localhost:5000/risk

## 📊 ページ詳細

### サマリーページ (`/`)
- 主要指数（S&P500、NASDAQ、DOW30）のリアルタイム表示
- AIによる市況コメント自動生成
- 主要銘柄（AAPL、GOOGL等）の予測・リスクサマリー
- 詳細分析ページへのナビゲーション

### 予測分析ページ (`/predict/<ticker>`)
- 複数予測モデルの選択・比較（Ensemble、XGBoost、ARIMA）
- 予測期間の設定（1週間、1ヶ月、3ヶ月）
- インタラクティブな予測結果チャート
- モデル精度評価テーブル（RMSE、MAPE、R²）

### リスク分析ページ (`/risk/<ticker>`)
- VaR（Value at Risk）・CVaR計算と表示
- 信頼度レベルの設定（90%、95%、99%）
- モンテカルロシミュレーション結果の可視化
- リスク評価サマリーと解釈説明

## 🔗 API仕様

### `/api/summary`
- **メソッド**: GET
- **機能**: サマリーページ用データ取得
- **レスポンス**: 主要指数、AI コメント、銘柄サマリー

### `/api/predict/<ticker>`
- **メソッド**: GET
- **パラメータ**: 
  - `models`: 予測モデル（カンマ区切り）
  - `period`: 期間（1w、1m、3m）
- **レスポンス**: 予測結果、精度評価、チャートデータ

### `/api/risk/<ticker>`
- **メソッド**: GET  
- **パラメータ**:
  - `confidence`: 信頼度（0.90、0.95、0.99）
  - `days`: シミュレーション日数
- **レスポンス**: VaR/CVaR、モンテカルロシミュレーション結果

## 🧪 テスト実行

```bash
# 基本機能テスト
python test_web_server.py

# 従来のレポート生成（動作確認）
python main.py enhanced
```

## ⚡ パフォーマンス特徴

### フォールバック実装
- 既存`src/analytics/`モジュール利用可能時: 高度な分析実行
- モジュール利用不可時: 軽量なフォールバック実装に自動切替
- エラー時でも基本機能は維持される堅牢性

### レスポンシブデザイン  
- デスクトップ・モバイル対応
- BootstrapによるモダンなUI/UX
- Plotly.jsによるインタラクティブチャート

## 📁 実装ファイル構成

```
market-report-generator/
├── src/api/                    # 新API実装
│   ├── __init__.py
│   └── routes.py              # APIエンドポイント + フォールバック実装
├── templates/                  # 新UIテンプレート
│   ├── summary_page.html      # サマリーページ
│   ├── predict_page.html      # 予測分析ページ
│   ├── risk_page.html         # リスク分析ページ
│   └── error.html             # エラーページ
├── main.py                    # Webサーバー機能追加
├── requirements.txt           # Flask依存関係追加
└── test_web_server.py         # 統合テスト
```

## 🔄 既存機能との併用

新しいWebUIと従来のHTMLレポート生成は独立して動作:

```bash
# 新WebUI起動
python main.py server

# 従来のレポート生成
python main.py enhanced
```

## 🎉 推奨用途

1. **日次マーケット監視**: サマリーページで市況を迅速把握
2. **詳細銘柄分析**: 予測・リスクページで投資判断サポート  
3. **専門的リスク評価**: VaR/CVaRによる定量的リスク管理
4. **教育・デモ**: インタラクティブUIによる金融分析の学習

---

**実装完了日**: 2024-12-25  
**実装者**: Claude Code Assistant  
**ブランチ**: feature/enhanced-ui-individual-stocks