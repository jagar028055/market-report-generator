# 米国マーケットレポート自動生成システム

このプロジェクトは、前日の米国市場動向を自動で収集・解析し、チャート・AI コメント付きの HTML レポートを生成します。

---

## 1. 処理フロー

1. **データ取得** (`data_fetcher.py`)
   * 主要株価指数・金利・商品・為替レート
   * 米経済指標（過去 24h / 未来 24h）
   * セクター ETF の前日比 (%)
   * ロイター日本語ニュース（カテゴリ・キーワードでフィルタ）
2. **チャート生成** (`chart_generator.py`)
   * イントラデイ & 1 年ローソク足 (mplfinance)
   * セクター ETF 変化率横棒グラフ (matplotlib)
3. **AI コメント生成** (`commentary_generator.py`)
   * Google Gemini API で「株式・金利・為替」の 3 段落コメントを生成
4. **HTML レポート生成** (`html_generator.py`)
   * Jinja2 でテンプレート `templates/report_template.html` を描画し `market_report.html` を出力

---

## 2. ディレクトリ構成

```
market_report_generator/
├─ charts/             # 生成された PNG チャート
├─ static/             # CSS など静的ファイル
├─ templates/          # Jinja2 テンプレート
├─ main.py             # エントリーポイント
├─ data_fetcher.py     # データ収集
├─ chart_generator.py  # チャート描画
├─ commentary_generator.py  # Gemini でコメント生成
├─ html_generator.py   # HTML 出力
├─ requirements.txt    # 依存ライブラリ
└─ README.md           # (本ファイル)
```

---

## 3. セットアップ

1. Python 3.9+ 推奨
2. 依存ライブラリをインストール
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows は .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. 環境変数を設定（`.env` も可）
   ```bash
   export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
   ```
4. ChromeDriver が PATH 上にあることを確認（Selenium 用）。

---

## 4. 実行方法

```bash
python main.py
```

完了後、プロジェクト直下に `market_report.html` が生成されます。ブラウザで開いてレポートを閲覧してください。

---

## 5. 出力物

| ファイル/フォルダ | 説明 |
|-------------------|------|
| `market_report.html` | 完成したマーケットレポート (日本語) |
| `charts/` | イントラデイ/長期チャート、セクターパフォーマンスチャート (PNG) |
| `execution.log` | 実行時ログ (任意) |

---

## 6. 注意事項

* Gemini API キーは必須です。無料枠の制限に注意してください。
* 週末でもレポート生成できるよう、経済指標・ニュースの取得時間幅を拡張済みです。
* `*_2.py` というファイルは旧版バックアップです。実際の呼び出しは無印ファイルを使用します。
* Mac の場合、日本語フォントが見つからないとチャートの日本語が文字化けします。必要に応じて `chart_generator.py` のフォントパスを編集してください。

---

## 7. ライセンス

MIT License (予定)。
