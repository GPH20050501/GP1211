import pytest
import pandas as pd
import numpy as np
from visualization.plotter import BacktestPlotter
from backtest.backtest_engine import Trade
import plotly.graph_objects as go

class TestVisualization:
    @pytest.fixture
    def sample_data(self):
        """生成測試用的價格數據"""
        dates = pd.date_range('2024-01-01', periods=100)
        data = pd.DataFrame({
            'Open': np.random.normal(100, 10, 100),
            'High': np.random.normal(105, 10, 100),
            'Low': np.random.normal(95, 10, 100),
            'Close': np.random.normal(100, 10, 100),
            'Volume': np.random.normal(1000000, 100000, 100)
        }, index=dates)
        return data
        
    @pytest.fixture
    def sample_trades(self):
        """生成測試用的交易記錄"""
        return [
            Trade(
                entry_date=pd.Timestamp('2024-01-05'),
                entry_price=100.0,
                position_size=1.0,
                direction=1,
                exit_date=pd.Timestamp('2024-01-10'),
                exit_price=105.0,
                profit_loss=5.0
            ),
            Trade(
                entry_date=pd.Timestamp('2024-01-15'),
                entry_price=105.0,
                position_size=1.0,
                direction=-1,
                exit_date=pd.Timestamp('2024-01-20'),
                exit_price=95.0,
                profit_loss=10.0
            )
        ]
        
    def test_plotter_initialization(self, sample_data, sample_trades):
        """測試繪圖器初始化"""
        equity_curve = pd.Series(np.linspace(100000, 110000, 100))
        plotter = BacktestPlotter(sample_data, equity_curve, sample_trades)
        
        assert isinstance(plotter.data, pd.DataFrame)
        assert isinstance(plotter.equity_curve, pd.Series)
        assert isinstance(plotter.trades, list)
        
    def test_plot_generation(self, sample_data, sample_trades):
        """測試圖表生成"""
        equity_curve = pd.Series(np.linspace(100000, 110000, 100))
        plotter = BacktestPlotter(sample_data, equity_curve, sample_trades)
        
        fig = plotter.plot_results()
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0  # 確保圖表包含數據
        assert fig.layout.title.text == '回測結果'

if __name__ == "__main__":
    pytest.main(['-v', __file__]) 