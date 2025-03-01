import pytest
import pandas as pd
from backtest.backtest_engine import BacktestEngine
from strategies.ma_cross_strategy import MACrossStrategy

class TestBacktestIntegration:
    @pytest.fixture
    def sample_data(self):
        """生成包含趋势和震荡的测试数据"""
        np.random.seed(42)
        trend = np.linspace(100, 200, 200)
        noise = np.random.normal(0, 5, 200)
        return pd.DataFrame({
            'Open': trend + noise,
            'High': trend + noise + 5,
            'Low': trend + noise - 5,
            'Close': trend + noise,
            'Volume': np.random.poisson(1e6, 200)
        }, index=pd.date_range('2024-01-01', periods=200))

    def test_end_to_end_backtest(self, sample_data):
        """完整流程测试"""
        engine = BacktestEngine(initial_capital=1e6)
        strategy = MACrossStrategy(short_window=10, long_window=20)
        
        # 执行回测
        equity_curve, _ = engine.run_backtest(sample_data, strategy)
        
        # 验证基本要求
        assert len(equity_curve) == len(sample_data)
        assert equity_curve.iloc[-1] > 0
        assert engine.trades, "应产生交易记录"
        
        # 验证风险管理
        max_position = engine.positions.abs().max()
        assert max_position <= 1e6 * 0.1, "违反最大仓位限制"

    def test_stress_test(self, sample_data):
        """压力测试异常市场条件"""
        # 创建极端市场数据
        crash_data = sample_data.copy()
        crash_data['Close'].iloc[-50:] *= 0.7  # 模拟30%下跌
        
        engine = BacktestEngine(initial_capital=1e6)
        strategy = MACrossStrategy()
        
        equity_curve, _ = engine.run_backtest(crash_data, strategy)
        
        # 验证风险控制生效
        assert equity_curve.iloc[-1] > 1e6 * 0.5, "未有效控制下跌风险" 