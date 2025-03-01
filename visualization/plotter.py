import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class BacktestPlotter:
    """回測結果可視化"""
    def __init__(self, data: pd.DataFrame, equity_curve: pd.Series, trades: list):
        self.data = data
        self.equity_curve = equity_curve
        self.trades = trades
        
    def plot_results(self) -> go.Figure:
        """繪製回測結果圖表"""
        try:
            # 創建子圖
            fig = make_subplots(rows=2, cols=1, 
                              shared_xaxes=True,
                              vertical_spacing=0.03,
                              subplot_titles=('價格與交易', '資金曲線'),
                              row_heights=[0.7, 0.3])
            
            # 添加價格線
            fig.add_trace(
                go.Candlestick(
                    x=self.data.index,
                    open=self.data['Open'],
                    high=self.data['High'],
                    low=self.data['Low'],
                    close=self.data['Close'],
                    name='價格'
                ),
                row=1, col=1
            )
            
            # 添加交易點
            for trade in self.trades:
                # 入場點
                fig.add_trace(
                    go.Scatter(
                        x=[trade.entry_date],
                        y=[trade.entry_price],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-up' if trade.direction > 0 else 'triangle-down',
                            size=10,
                            color='green' if trade.direction > 0 else 'red'
                        ),
                        name=f"{'做多' if trade.direction > 0 else '做空'} 入場"
                    ),
                    row=1, col=1
                )
                
                # 出場點
                if trade.exit_date and trade.exit_price:
                    fig.add_trace(
                        go.Scatter(
                            x=[trade.exit_date],
                            y=[trade.exit_price],
                            mode='markers',
                            marker=dict(
                                symbol='x',
                                size=10,
                                color='black'
                            ),
                            name='出場'
                        ),
                        row=1, col=1
                    )
            
            # 添加資金曲線
            fig.add_trace(
                go.Scatter(
                    x=self.equity_curve.index,
                    y=self.equity_curve.values,
                    name='資金曲線',
                    line=dict(color='blue')
                ),
                row=2, col=1
            )
            
            # 更新布局
            fig.update_layout(
                title='回測結果',
                xaxis_title='日期',
                yaxis_title='價格',
                yaxis2_title='資金',
                showlegend=True,
                height=800
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"繪製圖表失敗: {str(e)}")
            raise 