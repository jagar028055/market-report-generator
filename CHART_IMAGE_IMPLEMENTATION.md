# ポップアップチャート画像表示機能 要件定義

## 1. 概要

主要指標カードクリック時に表示されるポップアップ（モーダルウィンドウ）内のチャート表示が崩れる問題を解決する。
現在のインタラクティブチャート（HTML埋め込み）から静的チャート画像（PNG）表示に変更することで、表示崩れを防ぎ、視認性を向上させる。

## 2. 要件

### 2.1. チャート生成

-   **静的チャート画像の追加生成:**
    -   現在生成されているインタラクティブなHTMLチャート (`*_intraday.html`, `*_longterm.html`) に加えて、同じデータソースから静的なPNG画像チャート (`*_intraday.png`, `*_longterm.png`) を生成する。
    -   画像は `charts/` ディレクトリに保存する。
-   **既存チャートの維持:**
    -   レポート本体の「チャート」セクションで使用されているインタラクティブなHTMLチャートは、引き続き生成・利用する。

### 2.2. HTML・フロントエンド

-   **ポップアップ内の表示要素変更:**
    -   `templates/report_template.html` 内のポップアップ（モーダル）部分にある `<iframe>` 要素を `<img>` 要素に変更する。
-   **データ属性の更新:**
    -   各指標カード (`.market-card`) が持つ `data-chart-intraday` および `data-chart-longterm` 属性の値を、新しく生成されるPNG画像のパス (`charts/INDICATOR_intraday.png` など) に更新する。
-   **JavaScriptロジックの修正:**
    -   指標カードクリック時に、ポップアップ内の `<img>` タグの `src` 属性を、クリックされたカードの `data-chart-*` 属性値（PNG画像のパス）に設定するよう修正する。
    -   ポップアップ内の「イントラデイ」「長期」タブをクリックした際に、表示する `<img>` の `src` を対応するPNG画像のパスに切り替えるよう修正する。

## 3. 実装タスクリスト

### フェーズ1: バックエンド改修 (チャート生成ロジック)

1.  **`src/chart_generators/candlestick_chart_generator.py` の修正:**
    -   `generate_intraday_chart` と `generate_longterm_chart` メソッドに、生成タイプ (`'interactive'` or `'static'`) を指定する `chart_type` 引数を追加する。
    -   `chart_type` の値に応じて、適切な内部メソッド (`generate_interactive_chart` or `generate_static_chart`) を呼び出すように変更する。

2.  **`src/async_processors/async_chart_generator.py` の修正:**
    -   ポップアップ用の静的チャートを生成するため、`_generate_intraday_chart_async` と `_generate_longterm_chart_async` 内で `candlestick_generator` のメソッドを `chart_type='static'` で呼び出す処理を追加する。
    -   ファイル名を `.png` にし、戻り値の `interactive` フラグを `False` に設定する。
    -   **注:** 非同期処理では、HTMLとPNGの両方を生成するタスクを並行して実行する必要がある。

3.  **`main.py` の修正 (`enhanced_main`, `original_main`):**
    -   チャート生成ループ内で、各指標に対してインタラクティブなHTMLチャートと静的なPNGチャートの両方を生成する。
        -   `generate_..._chart_interactive()` を呼び出してHTMLを生成 (既存のロジック)。
        -   `generate_..._chart(..., chart_type='static')` を呼び出してPNGを生成 (新規追加)。
    -   `grouped_charts` にはHTMLチャートの情報を渡し、ポップアップ用のデータ属性にはPNGチャートのパスを渡すように `HTMLGenerator` へのデータ渡しを調整する。

### フェーズ2: フロントエンド改修 (HTMLテンプレートとJavaScript)

4.  **`templates/report_template.html` の修正:**
    -   各指標カードの `div.market-card` に、静的画像用のデータ属性を追加する。
        -   例: `data-chart-intraday-img="charts/{{ name }}_intraday.png"`
        -   例: `data-chart-longterm-img="charts/{{ name }}_longterm.png"`
    -   ポップアップモーダル (`#marketModal`) 内のチャート表示部分を以下のように変更する。
        -   `<iframe id="modalChartFrame" ...>` を `<img id="modalChartFrame" src="" alt="Chart" style="width: 100%; height: auto;">` に置き換える。
    -   JavaScript (`<script>` タグ内) を修正する。
        -   `marketCards.forEach` のクリックイベントリスナー内で、`iframe` の `src` を設定している部分を、`img` の `src` を設定するように変更する。
        -   `chartTabs.forEach` のクリックイベントリスナー内で、タブ切り替え時に `img` の `src` を対応するPNG画像のパスに切り替えるように変更する。

### フェーズ3: テストと確認

5.  **動作確認:**
    -   レポート生成スクリプトを実行し、エラーなく完了することを確認する。
    -   `charts/` ディレクトリに `.html` ファイルと `.png` ファイルの両方が生成されていることを確認する。
    -   生成された `market_report.html` をブラウザで開く。
    -   指標カードをクリックしてポップアップが表示され、チャート画像が正しく表示されることを確認する。
    -   ポップアップ内の「イントラデイ」「長期」タブを切り替えて、画像が正しく切り替わることを確認する。
    -   レポート本体の「チャート」セクションのインタラクティブチャートが引き続き正常に動作することを確認する。
