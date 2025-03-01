import pytest
import pandas as pd
import numpy as np
from data.market_data import MarketDataLoader
from utils.performance import PerformanceAnalyzer

class TestDataFlow:
    def test_data_pipeline(self):
        """測試完整數據處理流程"""
        # 1. 數據加載
        loader = MarketDataLoader()
        data = loader.load_data('AAPL', '2024-01-01', '2024-02-01')
        
        # 2. 數據清洗
        cleaned_data = loader._clean_data(data)
        
        # 3. 性能分析
        analyzer = PerformanceAnalyzer(cleaned_data)
        metrics = analyzer.calculate_metrics()
        
        # 驗證數據流程的完整性
        assert cleaned_data is not None
        assert not cleaned_data.isnull().any().any()
        assert all(metric in metrics for metric in [
            'total_return',
            'sharpe_ratio',
            'max_drawdown',
            'win_rate'
        ])

    @pytest.fixture
    def analyzer(self):
        return PerformanceAnalyzer()
        
    @pytest.fixture
    def sample_equity_curve(self):
        """生成測試用的權益曲線"""
        dates = pd.date_range('2024-01-01', periods=100)
        return pd.Series(np.linspace(100000, 110000, 100), index=dates)
        
    def test_performance_calculation(self, analyzer, sample_equity_curve):
        """測試績效計算"""
        metrics = analyzer.calculate_metrics(sample_equity_curve)
        
        assert isinstance(metrics, dict)
        assert 'total_return' in metrics
        assert 'annual_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics 