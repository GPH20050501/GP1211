import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from data.market_data import MarketDataLoader

class TestDataValidation:
    @pytest.fixture
    def market_loader(self):
        """創建 MarketDataLoader 實例"""
        return MarketDataLoader()
    
    @pytest.fixture
    def sample_data(self):
        """生成測試用的樣本數據"""
        dates = pd.date_range('2024-01-01', periods=5, tz='UTC')
        return pd.DataFrame({
            'Open': [100, 101, np.nan, 103, 104],
            'High': [102, 103, 104, 105, 106],
            'Low': [98, 99, 100, 101, 102],
            'Close': [101, 102, 103, 104, 105],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        }, index=dates)

    def test_data_completeness(self, market_loader, sample_data):
        """測試數據完整性"""
        # 驗證必要欄位
        required_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
        assert all(col in sample_data.columns for col in required_columns)
        
        # 驗證數據類型
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            assert pd.api.types.is_numeric_dtype(sample_data[col])

    def test_time_zone_consistency(self, market_loader, sample_data):
        """測試時區一致性"""
        # 確保索引是時間戳類型
        assert isinstance(sample_data.index, pd.DatetimeIndex)
        
        # 確保時區設置正確
        assert sample_data.index.tz == timezone.utc
        
        # 驗證時間序列的連續性
        time_diffs = sample_data.index.to_series().diff()[1:]
        assert (time_diffs == pd.Timedelta('1D')).all()

    def test_price_validity(self, market_loader, sample_data):
        """測試價格有效性"""
        valid_data = sample_data.dropna()
        
        # 驗證 OHLC 邏輯關係
        assert (valid_data['High'] >= valid_data['Open']).all()
        assert (valid_data['High'] >= valid_data['Close']).all()
        assert (valid_data['Low'] <= valid_data['Open']).all()
        assert (valid_data['Low'] <= valid_data['Close']).all()
        
        # 驗證價格為正
        assert (valid_data[['Open', 'High', 'Low', 'Close']] > 0).all().all()

    def test_volume_validity(self, market_loader, sample_data):
        """測試成交量有效性"""
        # 成交量應為非負整數
        assert (sample_data['Volume'] >= 0).all()
        assert sample_data['Volume'].dtype in [np.int64, np.float64]
        
        # 檢查異常成交量
        mean_volume = sample_data['Volume'].mean()
        std_volume = sample_data['Volume'].std()
        assert not ((sample_data['Volume'] > mean_volume + 3 * std_volume) |
                   (sample_data['Volume'] < mean_volume - 3 * std_volume)).any()

    def test_future_data_leakage(self, market_loader):
        """測試未來數據泄露"""
        current_time = datetime.now(timezone.utc)
        test_data = market_loader.load_data('AAPL', '2024-01-01', '2024-02-01')
        
        # 確保沒有未來的數據
        assert not (test_data.index > current_time).any()

    def test_data_cleaning(self, market_loader, sample_data):
        """測試數據清洗功能"""
        cleaned_data = market_loader._clean_data(sample_data)
        
        # 驗證清洗後的數據
        assert cleaned_data is not None
        assert not cleaned_data.isnull().any().any()
        assert len(cleaned_data) < len(sample_data)  # 應該刪除了含有 NaN 的行

    @pytest.mark.parametrize("market,expected_columns", [
        ('AAPL', {'Open', 'High', 'Low', 'Close', 'Volume'}),
        ('BTC-USD', {'Open', 'High', 'Low', 'Close', 'Volume'}),
    ])
    def test_market_specific_data(self, market_loader, market, expected_columns):
        """測試不同市場的數據格式"""
        data = market_loader.load_data(market, '2024-01-01', '2024-02-01')
        assert set(data.columns) >= expected_columns

    def test_data_snapshot(self, market_loader, sample_data, snapshot):
        """使用快照測試驗證數據處理的一致性"""
        processed_data = market_loader._process_data(sample_data)
        snapshot.assert_match(processed_data.to_dict()) 