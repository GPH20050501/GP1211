import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict
import streamlit as st

def plot_trading_signals(data, signals, ma_short=None, ma_long=None):
    """
    繪製交易信號圖表
    
    Parameters:
        data (pd.DataFrame): 股票數據
        signals (pd.Series): 交易信號
        ma_short (pd.Series): 短期移動平均線
        ma_long (pd.Series): 長期移動平均線
    """
    # 創建子圖
    fig = make_subplots(rows=2, cols=1, 
                       shared_xaxes=True,
                       vertical_spacing=0.03,
                       row_heights=[0.7, 0.3])

    # 添加K線圖
    fig.add_trace(go.Candlestick(x=data.index,
                                open=data['Open'],
                                high=data['High'],
                                low=data['Low'],
                                close=data['Close'],
                                name='K線'),
                 row=1, col=1)

    # 添加移動平均線
    if ma_short is not None:
        fig.add_trace(go.Scatter(x=data.index,
                               y=ma_short,
                               name='MA50',
                               line=dict(color='orange')),
                     row=1, col=1)
    
    if ma_long is not None:
        fig.add_trace(go.Scatter(x=data.index,
                               y=ma_long,
                               name='MA200',
                               line=dict(color='blue')),
                     row=1, col=1)

    # 添加交易信號
    buy_signals = data[signals == 1]
    sell_signals = data[signals == -1]

    fig.add_trace(go.Scatter(x=buy_signals.index,
                            y=buy_signals['Close'],
                            name='買入',
                            mode='markers',
                            marker=dict(symbol='triangle-up',
                                      size=10,
                                      color='red')),
                 row=1, col=1)

    fig.add_trace(go.Scatter(x=sell_signals.index,
                            y=sell_signals['Close'],
                            name='賣出',
                            mode='markers',
                            marker=dict(symbol='triangle-down',
                                      size=10,
                                      color='green')),
                 row=1, col=1)

    # 添加成交量圖
    fig.add_trace(go.Bar(x=data.index,
                        y=data['Volume'],
                        name='成交量'),
                 row=2, col=1)

    # 更新布局
    fig.update_layout(
        title='交易信號圖表',
        yaxis_title='價格',
        yaxis2_title='成交量',
        xaxis_rangeslider_visible=False
    )

    return fig

def plot_portfolio_performance(market_results: Dict, initial_capital: float):
    """繪製投資組合績效圖表"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=('投資組合表現', '市場資金配置')
    )
    
    # 1. 資金曲線和回撤圖
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 確保所有市場的資金歷史長度一致
    min_length = min(len(result['data'].index) for result in market_results.values())
    
    # 初始化總資金序列
    first_market = next(iter(market_results.values()))
    index = first_market['data'].index[:min_length]
    total_equity = pd.Series(initial_capital, index=index)
    
    # 添加每個市場的資金曲線
    for (market, result), color in zip(market_results.items(), colors):
        market_equity = pd.Series(
            result['history'][:min_length], 
            index=result['data'].index[:min_length]
        )
        fig.add_trace(
            go.Scatter(
                x=market_equity.index,
                y=market_equity,
                name=f'{market}資金曲線',
                line=dict(color=color, width=1),
                hovertemplate=f"{market}<br>日期: %{x}<br>資金: %{y:,.0f} 元<extra></extra>",
                yaxis='y1'
            ),
            row=1, col=1
        )
        total_equity = total_equity.add(market_equity - initial_capital, fill_value=0)
    
    # 添加總資金曲線
    fig.add_trace(
        go.Scatter(
            x=total_equity.index,
            y=total_equity,
            name='總資金曲線',
            line=dict(color='black', width=2),
            hovertemplate="總資金<br>日期: %{x}<br>資金: %{y:,.0f} 元<extra></extra>",
            yaxis='y1'
        ),
        row=1, col=1
    )
    
    # 添加回撤曲線
    cummax = total_equity.cummax()
    drawdown = (total_equity - cummax) / cummax * 100
    
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown,
            name='回撤率',
            line=dict(color='red', width=1),
            fill='tozeroy',
            opacity=0.3,
            yaxis='y2',
            hovertemplate="回撤<br>日期: %{x}<br>回撤率: %{y:.2f}%<extra></extra>"
        ),
        row=1, col=1
    )
    
    # 2. 市場資金配置圖
    market_allocations = {}
    for market, result in market_results.items():
        market_allocations[market] = result['history'][-1] - initial_capital
    
    fig.add_trace(
        go.Bar(
            x=list(market_allocations.keys()),
            y=list(market_allocations.values()),
            name='市場資金配置',
            marker_color=colors[:len(market_allocations)],
            text=[f'{v:,.0f} 元' for v in market_allocations.values()],
            textposition='auto',
            hovertemplate="市場: %{x}<br>配置金額: %{y:,.0f} 元<extra></extra>"
        ),
        row=2, col=1
    )
    
    # 更新布局
    fig.update_layout(
        title=dict(
            text='投資組合績效分析',
            x=0.5,
            y=0.95
        ),
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis2=dict(
            title="回撤率 (%)",
            overlaying='y',
            side='right',
            showgrid=False
        )
    )
    
    # 更新軸標籤
    fig.update_yaxes(title_text="資金", row=1, col=1)
    fig.update_yaxes(title_text="資金配置", row=2, col=1)
    fig.update_xaxes(title_text="交易日期", row=2, col=1)
    
    return fig

def plot_single_market_performance(data: pd.DataFrame, signals: pd.Series, 
                                 history: pd.Series, market: str):
    """繪製單一市場的回測表現"""
    
    # 創建子圖
    fig = make_subplots(rows=3, cols=1,
                       subplot_titles=(f'{market} - 價格和交易訊號', '資金曲線', '回撤分析'),
                       vertical_spacing=0.1,
                       row_heights=[0.5, 0.25, 0.25])

    # 1. K線圖和交易訊號
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='K線'
        ),
        row=1, col=1
    )
    
    # 添加均線
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['MA50'],
            name='MA50',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['MA200'],
            name='MA200',
            line=dict(color='red', width=1)
        ),
        row=1, col=1
    )
    
    # 添加交易訊號
    buy_signals = signals[signals == 1].index
    sell_signals = signals[signals == -1].index
    
    if len(buy_signals) > 0:
        fig.add_trace(
            go.Scatter(
                x=buy_signals,
                y=data.loc[buy_signals, 'Close'],
                mode='markers',
                name='買入訊號',
                marker=dict(
                    symbol='triangle-up',
                    size=10,
                    color='red'
                )
            ),
            row=1, col=1
        )
    
    if len(sell_signals) > 0:
        fig.add_trace(
            go.Scatter(
                x=sell_signals,
                y=data.loc[sell_signals, 'Close'],
                mode='markers',
                name='賣出訊號',
                marker=dict(
                    symbol='triangle-down',
                    size=10,
                    color='green'
                )
            ),
            row=1, col=1
        )
    
    # 2. 資金曲線
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history.values,
            name='資金曲線',
            line=dict(color='blue')
        ),
        row=2, col=1
    )
    
    # 3. 回撤分析
    rolling_max = history.expanding().max()
    drawdown = (history - rolling_max) / rolling_max * 100
    
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            name='回撤率',
            line=dict(color='red'),
            fill='tozeroy'
        ),
        row=3, col=1
    )
    
    # 更新布局
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text=f"{market} 市場回測分析",
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_combined_results(total_equity: pd.Series, market_results: Dict):
    """
    繪製合併後的回測結果圖表
    
    Parameters:
        total_equity (pd.Series): 合併後的總資金曲線
        market_results (Dict): 各市場的回測結果
    """
    # 創建子圖
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('總資金曲線', '回撤分析'),
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3]
    )
    
    # 添加總資金曲線
    fig.add_trace(
        go.Scatter(
            x=total_equity.index,
            y=total_equity.values,
            name='總資金',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    # 計算回撤
    drawdown = (total_equity - total_equity.cummax()) / total_equity.cummax() * 100
    
    # 添加回撤圖
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            name='回撤率',
            line=dict(color='red'),
            fill='tozeroy'
        ),
        row=2, col=1
    )
    
    # 更新布局
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="多市場綜合回測分析",
        xaxis_rangeslider_visible=False
    )
    
    # 更新軸標籤
    fig.update_yaxes(title_text="資金", row=1, col=1)
    fig.update_yaxes(title_text="回撤率 (%)", row=2, col=1)
    fig.update_xaxes(title_text="交易日期", row=2, col=1)
    
    # 顯示圖表
    st.plotly_chart(fig, use_container_width=True) 