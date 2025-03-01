# 多市場回測系統開發文檔

## 目標
開發一個簡單且高效的多市場回測系統，該系統能夠支援不同策略的回測，並提供清晰的交易訊號和回測分析，增強準確的回測計算和策略績效的儲存與對比，並且能模擬真實市場的情況。

## 功能需求

1. 多市場回測支持
   - 支援多市場數據並能進行回測分析

2. 回測計算
   - 準確計算回測（最大回測），並與實際市場情況接近

3. 策略設置
   - 支援不同的策略（如 RSI、布林通道等）

4. 績效指標儲存與對比
   - 支援每次回測完成後儲存策略績效，並能對不同策略進行績效對比

5. 可視化圖表
   - 支援顯示進出場位置、資金曲線、最大回測等指標

6. 策略報告匯出
   - 支援將回測結果匯出，並顯示最大回測、連續虧損、期望值、最終資金等指標

7. 數據清洗與處理
   - 進行數據清洗，處理缺失值、異常數據等

8. 真實市場模擬
   - 支援滑點、手續費、資金管理等真實市場特徵，確保回測結果的真實性

9. 資金管理
   - 最大持倉比例設為 10%

## 技術選擇

1. 編程語言：Python
   - Python 是回測開發的標準語言，擁有強大的數據處理和計算程式庫

2. 數據來源：Yahoo Finance
   - 使用 yfinance 庫獲取市場數據，免費且能夠獲取多市場數據

3. 可視化庫：Plotly、Streamlit
   - 使用 Plotly 生成互動性圖表，並使用 Streamlit 來建立使用者介面

4. 回測框架：Pandas、Talib
   - Pandas 用於處理數據和進行回測，Talib 用於計算技術指標

## 專案結構
```
multi_market_backtest/
│
├── data/
│   └── market_data.py              # 數據下載與處理
│
├── strategies/
│   ├── rsi_strategy.py             # RSI 策略
│   └── bollinger_strategy.py       # 布林帶策略
│
├── backtest/
│   ├── backtest.py                 # 回測邏輯
│   ├── performance.py              # 績效計算與儲存
│   └── capital_management.py       # 資金管理（最大持倉比等）
│
├── ui/
│   ├── app.py                      # Streamlit 應用界面
│   └── charts.py                   # 可視化圖表
│
└── utils/
    ├── data_cleaning.py            # 數據清洗與處理
    ├── utils.py                    # 通用函數
    └── report.py                   # 報告生成與儲存
```

## 核心模組功能說明

### 1. market_data.py（數據下載與處理）
```python
import yfinance as yf

def download_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    # 數據清洗
    data = data.dropna()  # 刪除缺失值
    return data
```

### 2. rsi_strategy.py（RSI 策略）
```python
import talib
import pandas as pd

def rsi_strategy(data, period=14, overbought=70, oversold=30):
    rsi = talib.RSI(data['Close'], timeperiod=period)
    signals = pd.Series(index=data.index, dtype='float64')
    
    # 進場條件：RSI < 30
    signals[rsi < oversold] = 1  # 進場
    # 退出條件：RSI > 70
    signals[rsi > overbought] = -1  # 退出
    
    return signals
```

### 3. backtest.py（回測邏輯）
```python
import pandas as pd

def backtest(data, signals, initial_capital=10000):
    capital = initial_capital
    position = 0
    history = []
    
    for i in range(1, len(data)):
        if signals[i] == 1 and position == 0:  # 進場
            position = capital / data['Close'][i]
        elif signals[i] == -1 and position > 0:  # 退出
            capital = position * data['Close'][i]
            position = 0
        
        history.append(capital + position * data['Close'][i])
    
    final_capital = capital + position * data['Close'][-1]
    return history, final_capital
```

### 4. performance.py（績效計算與儲存）
```python
import pandas as pd

def calculate_performance(history):
    max_drawdown = (min(history) - max(history)) / max(history)
    return max_drawdown

def save_performance(strategy_name, max_drawdown, final_capital):
    performance_data = {
        'Strategy': strategy_name,
        'Max Drawdown': max_drawdown,
        'Final Capital': final_capital
    }
    df = pd.DataFrame([performance_data])
    df.to_csv('performance_results.csv', mode='a', header=False, index=False)
```

### 5. capital_management.py（資金管理）
```python
def apply_position_size(capital, price, max_position=0.1):
    position_size = capital * max_position / price
    return position_size
```

### 6. app.py（Streamlit 應用界面）
```python
import streamlit as st
import plotly.graph_objects as go

def display_charts(data, strategy_signals):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index,
                                open=data['Open'],
                                high=data['High'],
                                low=data['Low'],
                                close=data['Close'], name='Candles'))
    
    # 顯示進出場點
    buy_signals = data[strategy_signals == 1]
    sell_signals = data[strategy_signals == -1]
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close']))
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close']))
    
    st.plotly_chart(fig)
```

## 性能優化
- 數據加載：使用多線程或批次加載數據提高速度
- 回測優化：使用向量化操作加速回測過程
- 結果儲存：用 CSV 或數據庫儲存績效數據，便於分析 