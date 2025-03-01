import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path
import time
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Any
from plotly.subplots import make_subplots
import logging
from functools import reduce

# 添加專案根目錄到 Python 路徑
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

# 確保 data 目錄存在
data_dir = root_dir / "data"
if not data_dir.exists():
    data_dir.mkdir(parents=True)

# 確保 cache 目錄存在
cache_dir = data_dir / "cache"
if not cache_dir.exists():
    cache_dir.mkdir(parents=True)

# 從其他模組導入所需的函數和類
try:
    from data.market_data import MarketDataLoader, calculate_technical_indicators, MultiMarketDataLoader
    from strategies.ma_cross_strategy import ma_cross_strategy, MACrossStrategy
    from backtest.backtest import BacktestEngine
    from backtest.capital_management import PositionConfig
    from backtest.performance import calculate_metrics
    from ui.charts import plot_portfolio_performance, plot_single_market_performance, plot_combined_results
    from visualization.plotter import BacktestPlotter
    from reporting.report_generator import ReportGenerator, BacktestReport
except ImportError as e:
    st.error(f"導入模組時發生錯誤: {str(e)}")
    st.info("請確保專案結構正確，且所有必要的模組都已安裝")
    logging.error(f"模組導入錯誤: {str(e)}", exc_info=True)
    raise

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 初始化數據加載器（使用緩存）
data_loader = MarketDataLoader(cache_dir=str(cache_dir))

def load_data_with_progress(ticker, start_date, end_date):
    """帶進度顯示的數據加載函數"""
    try:
        # 創建進度顯示區域
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        with progress_placeholder.container():
            progress_bar = st.progress(0)
            
            # 階段 1: 連接數據源
            status_placeholder.info('正在連接數據源...')
            time.sleep(0.5)
            progress_bar.progress(10)
            
            # 階段 2: 下載數據
            status_placeholder.info(f'正在下載 {ticker} 的歷史數據...')
            
            # 將 date 轉換為 datetime
            start_datetime = datetime.combine(start_date, datetime.min.time())
            end_datetime = datetime.combine(end_date, datetime.min.time())
            
            market_data = data_loader.download_data(
                ticker=ticker,
                start_date=start_datetime,
                end_date=end_datetime
            )
            
            if market_data is None:
                raise ValueError(f"無法獲取 {ticker} 的數據")
            progress_bar.progress(40)
            
            # 階段 3: 數據清洗
            status_placeholder.info('正在清洗數據...')
            time.sleep(0.5)
            progress_bar.progress(60)
            
            # 階段 4: 計算技術指標
            status_placeholder.info('正在計算技術指標...')
            try:
                data_with_indicators = calculate_technical_indicators(market_data)
                if 'MA50' not in data_with_indicators.columns or 'MA200' not in data_with_indicators.columns:
                    raise ValueError("無法計算均線指標")
            except Exception as e:
                raise ValueError(f"計算技術指標失敗: {str(e)}")
            
            progress_bar.progress(90)
            
            # 完成
            status_placeholder.success('數據處理完成！')
            progress_bar.progress(100)
            time.sleep(0.5)
            
            # 清理進度顯示
            progress_placeholder.empty()
            status_placeholder.empty()
            
            return data_with_indicators
            
    except Exception as e:
        status_placeholder.error(f'數據處理失敗: {str(e)}')
        progress_placeholder.empty()
        return None

def format_number(value: float, precision: int = 2) -> str:
    """格式化數字顯示"""
    if abs(value) >= 1_000_000:
        return f'{value/1_000_000:.{precision}f}M'
    elif abs(value) >= 1_000:
        return f'{value/1_000:.{precision}f}K'
    else:
        return f'{value:.{precision}f}'

def plot_chart(data: pd.DataFrame, signals: pd.Series, trades: pd.DataFrame = None):
    """繪製增強版走勢圖"""
    # 創建子圖
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03,
                       row_heights=[0.7, 0.3])
    
    # 主圖 - 價格和均線
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Close'],
                  name='收盤價',
                  line=dict(color='#1f77b4')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=data.index, y=data['MA50'],
                  name='50日均線',
                  line=dict(color='#ff7f0e')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=data.index, y=data['MA200'],
                  name='200日均線',
                  line=dict(color='#2ca02c')),
        row=1, col=1
    )
    
    # 添加交易點
    if trades is not None:
        # 買入點
        fig.add_trace(
            go.Scatter(
                x=trades['進場時間'],
                y=trades['進場價格'],
                mode='markers',
                name='買入',
                marker=dict(
                    symbol='triangle-up',
                    size=10,
                    color='red'
                )
            ),
            row=1, col=1
        )
        
        # 賣出點
        fig.add_trace(
            go.Scatter(
                x=trades['出場時間'],
                y=trades['出場價格'],
                mode='markers',
                name='賣出',
                marker=dict(
                    symbol='triangle-down',
                    size=10,
                    color='green'
                )
            ),
            row=1, col=1
        )
    
    # 副圖 - 成交量
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'],
               name='成交量',
               marker_color='#1f77b4'),
        row=2, col=1
    )
    
    # 更新布局
    fig.update_layout(
        title='股價走勢圖',
        xaxis_title='日期',
        yaxis_title='價格',
        height=800,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def calculate_performance_metrics(data: pd.DataFrame, initial_capital: float = 1000000) -> Dict[str, float]:
    """計算績效指標"""
    try:
        # 計算每日報酬率
        daily_returns = data['Close'].pct_change()
        
        # 計算累積報酬
        cumulative_returns = (1 + daily_returns).cumprod()
        total_return = (cumulative_returns.iloc[-1] - 1) * 100
        
        # 計算最終資金
        final_capital = initial_capital * (1 + total_return/100)
        
        # 計算獲利/虧損天數
        profitable_days = (daily_returns > 0).sum()
        losing_days = (daily_returns < 0).sum()
        win_rate = (profitable_days / len(daily_returns)) * 100
        
        # 計算連續獲利/虧損天數
        consecutive_wins = 0
        consecutive_losses = 0
        current_streak = 0
        for ret in daily_returns:
            if ret > 0:
                if current_streak > 0:
                    current_streak += 1
                else:
                    current_streak = 1
                consecutive_wins = max(consecutive_wins, current_streak)
            elif ret < 0:
                if current_streak < 0:
                    current_streak -= 1
                else:
                    current_streak = -1
                consecutive_losses = min(consecutive_losses, current_streak)
        
        # 計算其他指標
        annual_return = (daily_returns.mean() * 252) * 100
        annual_volatility = (daily_returns.std() * np.sqrt(252)) * 100
        risk_free_rate = 0.02
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        
        # 計算最大回撤
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() * 100
        
        # 計算平均獲利/虧損
        avg_gain = daily_returns[daily_returns > 0].mean() * 100
        avg_loss = daily_returns[daily_returns < 0].mean() * 100
        
        return {
            '起始資金': f'{initial_capital:,.0f}元',
            '最終資金': f'{final_capital:,.0f}元',
            '總報酬率': f'{total_return:.2f}%',
            '年化報酬率': f'{annual_return:.2f}%',
            '年化波動率': f'{annual_volatility:.2f}%',
            '夏普比率': f'{sharpe_ratio:.2f}',
            '最大回撤': f'{abs(max_drawdown):.2f}%',
            '勝率': f'{win_rate:.2f}%',
            '獲利天數': f'{profitable_days}天',
            '虧損天數': f'{losing_days}天',
            '最長連續獲利': f'{consecutive_wins}天',
            '最長連續虧損': f'{abs(consecutive_losses)}天',
            '平均每日獲利': f'{avg_gain:.2f}%',
            '平均每日虧損': f'{abs(avg_loss):.2f}%'
        }
    except Exception as e:
        logger.error(f"計算績效指標時出錯: {str(e)}")
        return {}

def get_ma_crossover_points(data: pd.DataFrame) -> pd.DataFrame:
    """識別均線交叉點"""
    # 計算快線是否穿越慢線
    data['Signal'] = 0
    data.loc[data['MA50'] > data['MA200'], 'Signal'] = 1
    
    # 找出交叉點
    crossover = data['Signal'].diff()
    entry_points = data[crossover == 1]
    
    return entry_points

def display_performance_metrics(metrics: Dict[str, Any]):
    """顯示績效指標"""
    try:
        st.subheader('績效指標')
        
        # 使用四列布局
        col1, col2, col3, col4 = st.columns(4)
        
        # 資金指標
        with col1:
            st.markdown("### 資金績效")
            st.metric("初始資金", format_number(metrics.get('initial_capital', 0)))
            st.metric("最終資金", format_number(metrics.get('final_capital', 0)))
            st.metric("總報酬率", f"{metrics.get('total_return_pct', 0):.2f}%")
            st.metric("年化複合成長率", f"{metrics.get('年化複合成長率', 0):.2f}%")
        
        # 風險指標
        with col2:
            st.markdown("### 風險指標")
            st.metric("最大回撤", f"{metrics.get('最大回撤(%)', 0):.2f}%")
            st.metric("回報/風險比", f"{metrics.get('回報/風險比', 0):.2f}")
            st.metric("破產風險", f"{metrics.get('破產風險(%)', 0):.2f}%")
            st.metric("期望值E[R]", f"{metrics.get('期望值E[R]', 0):.2f}%")
        
        # 交易統計
        with col3:
            st.markdown("### 交易統計")
            st.metric("總交易次數", str(metrics.get('total_trades', 0)))
            st.metric("獲利次數", str(metrics.get('winning_trades', 0)))
            st.metric("虧損次數", str(metrics.get('losing_trades', 0)))
            st.metric("勝率", f"{metrics.get('win_rate', 0):.2f}%")
        
        # 其他指標
        with col4:
            st.markdown("### 其他指標")
            st.metric("平均獲利", format_number(float(metrics.get('avg_profit_per_trade', 0))))
            st.metric("平均虧損", format_number(float(metrics.get('avg_loss_per_trade', 0))))
            st.metric("盈虧比", f"{metrics.get('profit_loss_ratio', 0):.2f}")
            st.metric("平均持倉天數", f"{metrics.get('avg_holding_period', 0):.1f}天")
            
    except Exception as e:
        st.error(f"顯示績效指標時發生錯誤: {str(e)}")

def create_chinese_table(data: pd.DataFrame) -> pd.DataFrame:
    """創建中文列名的數據表"""
    chinese_columns = {
        'Close': '收盤價',
        'Dividends': '股息',
        'High': '最高價',
        'Low': '最低價',
        'Open': '開盤價',
        'Stock Splits': '股票分割',
        'Volume': '成交量',
        'MA5': '5日均線',
        'MA20': '20日均線',
        'MA50': '50日均線',
        'MA200': '200日均線',
        'Volume_MA5': '5日均量',
        'Volatility': '波動率',
        'RSI': 'RSI指標',
        'MACD': 'MACD指標'
    }
    
    # 複製數據以避免修改原始數據
    df = data.copy()
    # 重命名列
    df.columns = [chinese_columns.get(col, col) for col in df.columns]
    return df

def display_metrics(metrics: Dict[str, float]):
    """顯示績效指標"""
    st.subheader('績效指標')
    
    # 使用四列布局
    col1, col2, col3, col4 = st.columns(4)
    
    # 收益指標
    with col1:
        st.markdown("### 收益指標")
        st.metric("原始資金", f"{metrics['initial_capital']}")
        st.metric("最終資金", f"{metrics['final_capital']}")
        st.metric("總報酬率", f"{metrics['total_return']:.2f}%")
        st.metric("年化收益率", f"{metrics['annual_return']:.2f}%")
    
    # 風險指標
    with col2:
        st.markdown("### 風險指標")
        st.metric("最大回撤", f"{metrics['max_drawdown']:.2f}%")
        st.metric("波動率", f"{metrics['annual_volatility']:.2f}%")
        st.metric("夏普比率", f"{metrics['sharpe_ratio']:.2f}")
        st.metric("交易成本比率", f"{metrics['transaction_cost_ratio']:.2f}%")
    
    # 交易統計
    with col3:
        st.markdown("### 交易統計")
        st.metric("交易次數", f"{metrics['total_trades']}")
        st.metric("獲利交易次數", f"{metrics['winning_trades']}")
        st.metric("虧損交易次數", f"{metrics['losing_trades']}")
        st.metric("勝率", f"{metrics['win_rate']:.2f}%")
    
    # 交易效率
    with col4:
        st.markdown("### 交易效率")
        st.metric("平均獲利", f"{metrics['avg_profit_per_trade']:.2f}")
        st.metric("平均虧損", f"{metrics['avg_loss_per_trade']:.2f}")
        st.metric("獲利因子", f"{metrics['profit_factor']:.2f}")
        st.metric("平均持倉天數", f"{metrics['avg_holding_days']:.1f} 天")

def display_trade_history(trades: pd.DataFrame):
    """顯示交易歷史"""
    if trades.empty:
        st.warning("沒有交易記錄")
        return
    
    st.subheader("交易記錄")
    
    # 格式化交易記錄
    formatted_trades = trades.copy()
    formatted_trades['進場時間'] = pd.to_datetime(formatted_trades['進場時間']).dt.strftime('%Y-%m-%d')
    formatted_trades['出場時間'] = pd.to_datetime(formatted_trades['出場時間']).dt.strftime('%Y-%m-%d')
    formatted_trades['進場價格'] = formatted_trades['進場價格'].round(2)
    formatted_trades['出場價格'] = formatted_trades['出場價格'].round(2)
    formatted_trades['獲利/虧損'] = formatted_trades['獲利/虧損'].round(2)
    formatted_trades['手續費'] = formatted_trades['手續費'].round(2)
    formatted_trades['滑價成本'] = formatted_trades['滑價成本'].round(2)
    
    # 添加收益率列
    formatted_trades['收益率'] = (
        formatted_trades['獲利/虧損'] / 
        (formatted_trades['進場價格'] * formatted_trades['position_size'])
    ) * 100
    formatted_trades['收益率'] = formatted_trades['收益率'].round(2).astype(str) + '%'
    
    # 設置顯示列順序
    columns = [
        '進場時間', '進場價格', '出場時間', '出場價格',
        '持倉數量', '獲利/虧損', '收益率', '持倉天數',
        '手續費', '滑價成本'
    ]
    
    # 使用 st.dataframe 顯示交易記錄
    st.dataframe(
        formatted_trades[columns],
        use_container_width=True,
        hide_index=True
    )

def process_date_input(date_str: str) -> datetime:
    """處理日期輸入，確保返回正確的datetime對象"""
    try:
        # 如果輸入的是字符串格式的日期
        if isinstance(date_str, str):
            return datetime.strptime(date_str, '%Y/%m/%d')
        # 如果已經是datetime對象
        elif isinstance(date_str, datetime):
            return date_str
        else:
            raise ValueError(f"不支持的日期格式: {type(date_str)}")
    except Exception as e:
        st.error(f"日期格式錯誤: {str(e)}")
        raise

def plot_portfolio_performance(market_results: Dict[str, Dict]):
    """繪製投資組合績效圖表"""
    if not market_results:
        st.warning("沒有可用的回測結果")
        return
        
    # 創建子圖
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=('投資組合淨值', '各市場淨值'),
                       vertical_spacing=0.15)
    
    # 合併所有市場的資金曲線
    combined_equity = pd.Series(dtype='float64')
    for market, result in market_results.items():
        if 'history' in result and len(result['history']) > 0:
            market_equity = pd.Series(result['history'])
            if len(combined_equity) == 0:
                combined_equity = market_equity
            else:
                # 確保索引對齊
                combined_equity = combined_equity.add(market_equity, fill_value=0)
    
    # 繪製總資金曲線
    if len(combined_equity) > 0:
        fig.add_trace(
            go.Scatter(
                x=combined_equity.index,
                y=combined_equity.values,
                name='總資金曲線',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
    
    # 繪製各市場資金曲線
    colors = ['red', 'green', 'purple', 'orange']
    for i, (market, result) in enumerate(market_results.items()):
        if 'history' in result and len(result['history']) > 0:
            market_equity = pd.Series(result['history'])
            fig.add_trace(
                go.Scatter(
                    x=market_equity.index,
                    y=market_equity.values,
                    name=f'{market}資金曲線',
                    line=dict(color=colors[i % len(colors)])
                ),
                row=2, col=1
            )
    
    # 更新布局
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="投資組合績效分析"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_backtest_results(market_results: Dict[str, Dict]):
    """繪製回測結果圖表"""
    try:
        if not market_results:
            st.warning("沒有可用的回測結果")
            return
        
        # 創建子圖
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('資金曲線', '交易信號'),
            vertical_spacing=0.15,
            row_heights=[0.6, 0.4]
        )
        
        # 繪製資金曲線
        for market, result in market_results.items():
            if 'history' in result and not result['history'].empty:  # 修改判斷條件
                # 確保資金曲線是 Series 類型
                if isinstance(result['history'], (list, np.ndarray)):
                    equity = pd.Series(result['history'], index=result['data'].index[:len(result['history'])])
                else:
                    equity = result['history']
                
                fig.add_trace(
                    go.Scatter(
                        x=equity.index,
                        y=equity.values,
                        name=f'{market}資金曲線',
                        mode='lines'
                    ),
                    row=1, col=1
                )
        
        # 繪製交易信號
        for market, result in market_results.items():
            if 'data' in result and 'signals' in result:
                data = result['data']
                signals = result['signals']
                
                # 添加K線圖
                fig.add_trace(
                    go.Candlestick(
                        x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name=f'{market} K線'
                    ),
                    row=2, col=1
                )
                
                # 添加均線
                for ma in ['MA50', 'MA200']:
                    if ma in data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=data.index,
                                y=data[ma],
                                name=f'{market} {ma}',
                                line=dict(
                                    color='blue' if ma == 'MA50' else 'red',
                                    width=1
                                )
                            ),
                            row=2, col=1
                        )
                
                # 添加買賣信號
                buy_signals = data[signals == 1]
                sell_signals = data[signals == -1]
                
                if not buy_signals.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=buy_signals.index,
                            y=buy_signals['Close'],
                            mode='markers',
                            name=f'{market} 買入信號',
                            marker=dict(symbol='triangle-up', size=10, color='red')
                        ),
                        row=2, col=1
                    )
                
                if not sell_signals.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=sell_signals.index,
                            y=sell_signals['Close'],
                            mode='markers',
                            name=f'{market} 賣出信號',
                            marker=dict(symbol='triangle-down', size=10, color='green')
                        ),
                        row=2, col=1
                    )
        
        # 更新布局
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="回測結果分析",
            xaxis_rangeslider_visible=False
        )
        
        # 顯示圖表
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"繪製圖表時發生錯誤: {str(e)}")
        logging.error(f"圖表繪製錯誤詳情: {str(e)}", exc_info=True)

def main():
    st.set_page_config(page_title="量化交易回測系統", layout="wide")
    
    # 側邊欄 - 參數設置
    with st.sidebar:
        st.header("參數設置")
        
        # 1. 市場選擇
        st.subheader("市場選擇")
        selected_symbols = st.multiselect(
            "選擇交易標的",
            ["AAPL", "GOOGL", "MSFT", "AMZN", "META"],
            default=["AAPL"]
        )
        
        # 2. 回測區間
        st.subheader("回測區間")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "開始日期",
                datetime.now() - timedelta(days=365)
        )
    with col2:
        end_date = st.date_input(
            "結束日期",
                datetime.now()
            )
        
        # 3. 策略參數
        st.subheader("策略參數")
        short_window = st.slider("短期均線", 5, 50, 5)
        long_window = st.slider("長期均線", 20, 200, 20)
        
        # 4. 資金設置
        st.subheader("資金設置")
        initial_capital = st.number_input(
            "初始資金",
            min_value=10000,
            value=100000,
            step=10000
        )
        
        # 執行回測按鈕
        run_backtest = st.button("執行回測")
    
    # 主界面
    st.title("量化交易回測系統")
    
    if run_backtest:
        try:
            # 1. 加載數據
            with st.spinner("正在加載市場數據..."):
                loader = MultiMarketDataLoader()
                market_data = loader.load_data(
                    symbols=selected_symbols,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d")
                )
            
            if not market_data:
                st.error("無法加載市場數據")
                return
            
            # 2. 執行回測
            results = {}
            for symbol, data in market_data.items():
                with st.spinner(f"正在回測 {symbol}..."):
                    strategy = MACrossStrategy(
                        short_window=short_window,
                        long_window=long_window
                    )
                    engine = BacktestEngine(initial_capital=initial_capital)
                    equity_curve, final_capital = engine.run_backtest(data, strategy)
                    
                    results[symbol] = {
                            'data': data,
                        'equity_curve': equity_curve,
                        'trades': engine.trades,
                        'metrics': engine.calculate_metrics()
                    }
            
            # 3. 顯示結果
            for symbol, result in results.items():
                st.header(f"{symbol} 回測結果")
                
                # 顯示主要指標
                metrics = result['metrics']
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("總收益率", f"{metrics['total_return']:.2%}")
                with col2:
                    st.metric("年化收益率", f"{metrics['annual_return']:.2%}")
                with col3:
                    st.metric("夏普比率", f"{metrics['sharpe_ratio']:.2f}")
                with col4:
                    st.metric("最大回撤", f"{metrics['max_drawdown']:.2%}")
                
                # 顯示圖表
                plotter = BacktestPlotter(
                    data=result['data'],
                    equity_curve=result['equity_curve'],
                    trades=result['trades']
                )
                fig = plotter.plot_results()
                st.plotly_chart(fig, use_container_width=True)
                
                # 顯示交易記錄
                if result['trades']:
                st.subheader("交易記錄")
                    trades_df = pd.DataFrame([
                        {
                            '入場時間': t.entry_date,
                            '入場價格': t.entry_price,
                            '出場時間': t.exit_date,
                            '出場價格': t.exit_price,
                            '收益': t.profit_loss,
                            '方向': '做多' if t.direction > 0 else '做空'
                        }
                        for t in result['trades']
                    ])
                    st.dataframe(trades_df)
            
            # 4. 生成報告
            if st.button("生成詳細報告"):
                report_generator = ReportGenerator()
                for symbol, result in results.items():
                    backtest_report = BacktestReport(
                        strategy_name=f"MA Cross ({short_window}, {long_window})",
                        start_date=start_date,
                        end_date=end_date,
                        initial_capital=initial_capital,
                        final_capital=result['equity_curve'][-1],
                        total_return=result['metrics']['total_return'],
                        annual_return=result['metrics']['annual_return'],
                        sharpe_ratio=result['metrics']['sharpe_ratio'],
                        max_drawdown=result['metrics']['max_drawdown'],
                        win_rate=result['metrics']['win_rate'],
                        trade_count=len(result['trades']),
                        trades=result['trades'],
                        metrics=result['metrics']
                    )
                    report_path = report_generator.generate_report(backtest_report)
                    st.success(f"{symbol} 報告已生成: {report_path}")
                
        except Exception as e:
            st.error(f"回測過程中發生錯誤: {str(e)}")

if __name__ == "__main__":
    main() 