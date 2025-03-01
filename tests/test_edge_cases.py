import pytest
import pandas as pd
import numpy as np
from data.market_data import MarketDataLoader
from strategies.ma_cross_strategy import MACrossStrategy
from backtest.backtest import BacktestEngine

class TestEdgeCases:
    @pytest.fixture
    def market_loader(self):
        return MarketDataLoader()
        
    @pytest.fixture
    def strategy(self):
        return MACrossStrategy()
        
    def test_empty_data(self, market_loader, strategy):
        """測試空數據處理"""
        empty_data = pd.DataFrame()
        
        with pytest.raises(ValueError):
            market_loader._clean_data(empty_data)
            
    def test_single_row_data(self, market_loader, strategy):
        """測試單行數據處理"""
        single_row = pd.DataFrame({
            'Open': [100],
            'High': [101],
            'Low': [99],
            'Close': [100],
            'Volume': [1000]
        }, index=[pd.Timestamp('2024-01-01')])
        
        with pytest.raises(ValueError):
            strategy.generate_signals(single_row)
            
    def test_extreme_prices(self, market_loader):
        """測試極端價格"""
        extreme_data = pd.DataFrame({
            'Open': [100, np.inf, -np.inf, 1e10, -1e10],
            'High': [101, np.inf, -np.inf, 1e10, -1e10],
            'Low': [99, np.inf, -np.inf, 1e10, -1e10],
            'Close': [100, np.inf, -np.inf, 1e10, -1e10],
            'Volume': [1000, 1000, 1000, 1000, 1000]
        }, index=pd.date_range('2024-01-01', periods=5))
        
        cleaned_data = market_loader._clean_data(extreme_data)
        assert len(cleaned_data) < len(extreme_data)
        assert not cleaned_data.isin([np.inf, -np.inf]).any().any()
        
    def test_zero_volume(self, market_loader):
        """測試零成交量"""
        zero_volume_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [101, 102, 103],
            'Low': [99, 100, 101],
            'Close': [100, 101, 102],
            'Volume': [1000, 0, 1200]
        }, index=pd.date_range('2024-01-01', periods=3))
        
        cleaned_data = market_loader._clean_data(zero_volume_data)
        assert len(cleaned_data) < len(zero_volume_data)
        assert all(cleaned_data['Volume'] > 0) 