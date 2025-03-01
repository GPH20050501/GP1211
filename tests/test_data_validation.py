import pytest
import pandas as pd
import numpy as np
from data.market_data import MarketDataLoader

def test_data_cleaning():
    """測試數據清洗功能"""
    # 創建測試數據
    test_data = pd.DataFrame({
        'Open': [10, 11, np.nan, 13],
        'High': [12, 13, 14, 15],
        'Low': [9, 10, 11, 12],
        'Close': [11, 12, 13, 14],
        'Volume': [1000, 1100, 1200, 1300]
    })
    
    loader = MarketDataLoader()
    cleaned_data = loader._clean_data(test_data)
    
    # 驗證清洗後的數據
    assert cleaned_data is not None
    assert not cleaned_data.isnull().any().any()
    assert len(cleaned_data) == 3  # 應該刪除了含有 NaN 的行
    
def test_data_quality():
    """測試數據質量報告"""
    test_data = pd.DataFrame({
        'Open': [10, 11, 12],
        'High': [12, 13, 14],
        'Low': [9, 10, 11],
        'Close': [11, 12, 13],
        'Volume': [1000, 1100, 1200]
    }, index=pd.date_range('2024-01-01', periods=3))
    
    loader = MarketDataLoader()
    report = loader.generate_quality_report(test_data)
    
    assert '數據起始日期' in report
    assert '數據結束日期' in report
    assert '交易天數' in report 