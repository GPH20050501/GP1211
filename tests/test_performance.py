import pytest
import time
import pandas as pd
import numpy as np
from data.market_data import MarketDataLoader
from strategies.ma_cross_strategy import ma_cross_strategy
from backtest.backtest import BacktestEngine

class TestPerformance:
    @pytest.mark.benchmark
    def test_data_processing_speed(self):
        """測試數據處理速度"""
        loader = MarketDataLoader()
        
        start_time = time.time()
        data = loader.load_data('AAPL', '2023-01-01', '2024-01-01')
        processing_time = time.time() - start_time
        
        assert processing_time < 5.0, f"數據處理時間過長: {processing_time:.2f}秒"

    @pytest.mark.benchmark
    def test_backtest_performance(self):
        """測試回測性能"""
        # 創建大量測試數據
        dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
        data = pd.DataFrame({
            'Close': np.random.randn(len(dates)).cumsum() + 100
        }, index=dates)
        
        start_time = time.time()
        signals = ma_cross_strategy(data)
        strategy_time = time.time() - start_time
        
        assert strategy_time < 1.0, f"策略計算時間過長: {strategy_time:.2f}秒" 