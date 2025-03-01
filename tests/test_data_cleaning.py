import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from data.market_data import MarketDataLoader, MarketDataCleaner
import pytz

class TestDataCleaning:
    @pytest.fixture
    def market_loader(self):
        """創建 MarketDataLoader 實例"""
        return MarketDataLoader()
    
    @pytest.fixture
    def sample_data(self):
        """生成測試數據"""
        dates = pd.date_range('2024-01-01', periods=5)
        return pd.DataFrame({
            'Open': [10, 11, np.nan, 13, 14],
            'High': [12, 13, 14, 15, 16],
            'Low': [9, 10, 11, 12, 13],
            'Close': [11, 12, 13, 14, 15],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        }, index=dates)

    @pytest.fixture
    def cleaner(self):
        return MarketDataCleaner()

    def test_missing_value_handling(self, market_loader, sample_data):
        """測試缺失值處理"""
        cleaned_data = market_loader._clean_data(sample_data)
        assert not cleaned_data.isnull().any().any()

    def test_outlier_detection(self, cleaner):
        """測試異常值檢測"""
        test_data = pd.DataFrame({
            'Close': [100, 100, 1000, 100],  # 1000 是明顯的異常值
            'Volume': [1000, 1000, 1000, 1000]
        }, index=pd.date_range('2024-01-01', periods=4))
        
        cleaned_data = cleaner.clean_data(test_data)
        assert len(cleaned_data) < len(test_data), "應該移除異常值"
        
    def test_price_logic_validation(self, cleaner):
        """測試價格邏輯驗證"""
        test_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [99, 102, 103],  # 99 < Open，違反邏輯
            'Low': [98, 100, 101],
            'Close': [101, 101, 102],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range('2024-01-01', periods=3))
        
        cleaned_data = cleaner.clean_data(test_data)
        assert len(cleaned_data) < len(test_data), "應該過濾掉違反價格邏輯的數據"
        
    def test_future_timestamp_prevention(self, market_loader):
        """測試未來時間戳過濾"""
        current_time = pd.Timestamp.now(tz='UTC')
        future_dates = pd.date_range(current_time, periods=3, freq='D')
        test_data = pd.DataFrame({
            'Close': [101, 102, 103],
            'Volume': [1100, 1200, 1300]
        }, index=future_dates)
        
        cleaned_data = market_loader._clean_data(test_data)
        assert all(cleaned_data.index <= current_time), "未過濾未來時間戳"
        
    @pytest.mark.parametrize("test_input,expected", [
        # 測試價格為零或負數的情況
        (pd.DataFrame({
            'Close': [100, 0, -100, 101],
            'Volume': [1000, 1100, 1200, 1300]
        }, index=pd.date_range('2024-01-01', periods=4)),
         2),  # 應該只保留2個有效數據
        
        # 測試成交量為零的情況
        (pd.DataFrame({
            'Close': [100, 101, 102, 103],
            'Volume': [1000, 0, -1, 1300]
        }, index=pd.date_range('2024-01-01', periods=4)),
         2),  # 應該只保留2個有效數據
        
        # 測試OHLC邏輯錯誤的情況
        (pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [99, 102, 103],  # High < Open
            'Low': [98, 100, 101],
            'Close': [101, 101, 102],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range('2024-01-01', periods=3)),
         2)  # 應該過濾掉邏輯錯誤的數據
    ])
    def test_edge_cases(self, cleaner, test_input, expected):
        """測試邊界情況"""
        cleaned_data = cleaner.clean_data(test_input)
        assert len(cleaned_data) == expected, \
            f"邊界情況處理錯誤，期望{expected}行數據，實際得到{len(cleaned_data)}行"

    def test_data_types(self, market_loader, sample_data):
        """測試數據類型一致性"""
        cleaned_data = market_loader._clean_data(sample_data)
        
        # 驗證數值列的類型
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            assert pd.api.types.is_numeric_dtype(cleaned_data[col]), f"{col}列不是數值類型"
        
        # 驗證索引類型
        assert isinstance(cleaned_data.index, pd.DatetimeIndex), "索引不是DatetimeIndex類型"
        assert cleaned_data.index.is_monotonic_increasing, "索引未按時間順序排序"

    def test_timezone_handling(self, market_loader):
        """測試時區處理"""
        # 創建不同時區的測試數據
        dates = pd.date_range('2024-01-01', periods=3, tz='Asia/Tokyo')
        test_data = pd.DataFrame({
            'Close': [100, 101, 102],
            'Volume': [1000, 1100, 1200]
        }, index=dates)
        
        cleaned_data = market_loader._clean_data(test_data)
        
        # 驗證時區轉換
        assert cleaned_data.index.tz == pytz.UTC, "時區未統一轉換為UTC"
        assert len(cleaned_data) == len(test_data), "時區轉換不應改變數據長度"

    def test_continuous_time_series(self, cleaner):
        """測試時間序列連續性"""
        # 創建不連續的時間序列數據
        dates = pd.date_range('2024-01-01', periods=5)
        dates = dates.append(pd.date_range('2024-01-07', periods=5))
        
        data = pd.DataFrame({
            'Close': range(10),
            'Volume': range(1000, 1100, 10)
        }, index=dates)
        
        cleaned_data = cleaner.clean_data(data)
        
        # 驗證時間序列
        date_diffs = cleaned_data.index.to_series().diff().dropna()
        assert all(date_diffs == pd.Timedelta(days=1)), "時間序列不連續" 