# チャートカスタマイズ機能

このプロジェクトでは、移動平均線のカスタマイズ機能が実装されています。

## 設定方法

### 1. config.py での設定

`config.py` ファイルで移動平均の設定を変更できます：

```python
# 移動平均設定
MOVING_AVERAGES: Dict[str, Dict] = {
    "short": {"period": 25, "color": "blue", "label": "MA25"},
    "medium": {"period": 50, "color": "orange", "label": "MA50"}, 
    "long": {"period": 75, "color": "red", "label": "MA75"}
}

# デフォルトで表示する移動平均（キーのリスト）
DEFAULT_MA_DISPLAY: List[str] = ["short", "long"]

# 移動平均タイプ設定
MA_TYPES: Dict[str, str] = {
    "SMA": "Simple Moving Average",
    "EMA": "Exponential Moving Average", 
    "WMA": "Weighted Moving Average"
}

# デフォルト移動平均タイプ
DEFAULT_MA_TYPE: str = "SMA"
```

### 2. プログラムでの使用

チャート生成時に移動平均をカスタマイズできます：

```python
from chart_generator import ChartGenerator
from config import Config

config = Config()
generator = ChartGenerator(config=config)

# カスタム移動平均でチャート生成
generator.generate_longterm_chart_interactive(
    data=data,
    ticker_name="S&P500",
    filename="custom_chart.html",
    ma_keys=["short", "medium"],  # 25日線と50日線のみ表示
    ma_type="EMA"  # 指数移動平均を使用
)
```

## 移動平均タイプ

### SMA (Simple Moving Average) - 単純移動平均
期間内の終値の単純平均。最も基本的な移動平均。

### EMA (Exponential Moving Average) - 指数移動平均
最近の価格により重きを置く移動平均。価格変動に敏感に反応。

### WMA (Weighted Moving Average) - 加重移動平均
期間内で線形的に重み付けされた移動平均。最新の価格が最も重要視される。

## HTMLレポートでの操作

生成されたHTMLレポートでは、以下の操作が可能です：

1. **移動平均タイプの選択**: SMA、EMA、WMAから選択
2. **表示する移動平均の選択**: 25日線、50日線、75日線の表示/非表示
3. **設定のリセット**: デフォルト設定に戻す

注意：HTMLでの設定変更は表示用のUIのみで、実際のチャート再生成にはサーバーサイドでの処理が必要です。

## カスタマイズ例

### 短期トレード用設定
```python
MOVING_AVERAGES = {
    "short": {"period": 5, "color": "blue", "label": "MA5"},
    "medium": {"period": 20, "color": "green", "label": "MA20"},
    "long": {"period": 50, "color": "red", "label": "MA50"}
}
DEFAULT_MA_DISPLAY = ["short", "medium"]
DEFAULT_MA_TYPE = "EMA"
```

### 長期投資用設定
```python
MOVING_AVERAGES = {
    "medium": {"period": 50, "color": "blue", "label": "MA50"},
    "long": {"period": 200, "color": "red", "label": "MA200"},
    "extra_long": {"period": 300, "color": "purple", "label": "MA300"}
}
DEFAULT_MA_DISPLAY = ["medium", "long"]
DEFAULT_MA_TYPE = "SMA"
```

## トラブルシューティング

### 移動平均が表示されない場合
1. `config.py` の設定を確認
2. データに十分な期間があるか確認（移動平均期間より長い期間のデータが必要）
3. `ma_keys` パラメータが正しく設定されているか確認

### 色が正しく表示されない場合
1. 色名またはHEXコードが正しいか確認
2. Plotlyでサポートされている色名を使用しているか確認

### エラーが発生する場合
1. `config.py` の構文エラーがないか確認
2. 移動平均期間が正の整数であるか確認
3. ログファイルでエラーメッセージを確認