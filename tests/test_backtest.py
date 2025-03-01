import pytest
import pandas as pd
import numpy as np
from backtest.backtest_engine import BacktestEngine
from strategies.ma_cross_strategy import MACrossStrategy

class TestBacktest:
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
        
    def test_backtest_initialization(self):
        """測試回測引擎初始化"""
        engine = BacktestEngine(initial_capital=100000)
        assert engine.initial_capital == 100000
        assert engine.current_capital == 100000
        assert len(engine.trades) == 0
        
    def test_backtest_execution(self, sample_data):
        """測試回測執行"""
        engine = BacktestEngine(initial_capital=100000)
        strategy = MACrossStrategy()
        
        equity_curve, final_capital = engine.run_backtest(sample_data, strategy)
        
        assert isinstance(equity_curve, pd.Series)
        assert len(equity_curve) == len(sample_data)
        assert final_capital > 0
        assert len(engine.trades) >= 0
        
    def test_metrics_calculation(self, sample_data):
        """測試績效指標計算"""
        engine = BacktestEngine(initial_capital=100000)
        strategy = MACrossStrategy()
        
        engine.run_backtest(sample_data, strategy)
        metrics = engine.calculate_metrics()
        
        # 驗證績效指標
        assert isinstance(metrics, dict)
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert -1 <= metrics['total_return'] <= 10
        assert -5 <= metrics['sharpe_ratio'] <= 5
        assert -1 <= metrics['max_drawdown'] <= 0
        
    def test_transaction_costs(self, sample_data):
        """測試交易成本計算"""
        engine = BacktestEngine(initial_capital=100000)
        strategy = MACrossStrategy()
        
        _, final_capital = engine.run_backtest(sample_data, strategy)
        
        # 驗證交易成本的影響
        assert final_capital < engine.initial_capital * 1.5  # 確保收益合理
        assert all(trade.commission > 0 for trade in engine.trades if trade.commission)
        assert all(trade.slippage > 0 for trade in engine.trades if trade.slippage)

    def test_equity_curve_dtype(self, sample_data):
        """測試資金曲線的數據類型"""
        engine = BacktestEngine(initial_capital=100000)
        strategy = MACrossStrategy()
        
        equity_curve, _ = engine.run_backtest(sample_data, strategy)
        
        # 驗證資金曲線的數據類型
        assert equity_curve.dtype == np.float64, "資金曲線應該是 float64 類型"
        assert not equity_curve.isna().any(), "資金曲線不應該包含 NaN 值"
        assert all(isinstance(v, float) for v in equity_curve), "所有值應該是浮點數"

if __name__ == "__main__":
    pytest.main(['-v', __file__]) 