import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.market_data import MarketDataLoader
import logging
import pytz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestDataLayerExtended:
    @pytest.fixture
    def market_loader(self):
        return MarketDataLoader()

    @pytest.fixture
    def multi_market_data(self):
        """生成多市場測試數據"""
        markets = ['AAPL', 'GOOGL', 'BTC-USD']
        data = {}
        for market in markets:
            # 生成基準價格
            base_price = 100
            n_samples = 100
            
            # 生成符合邏輯的價格數據
            opens = np.random.normal(base_price, 2, n_samples)
            highs = opens + abs(np.random.normal(2, 0.5, n_samples))  # 確保高於開盤價
            lows = opens - abs(np.random.normal(2, 0.5, n_samples))   # 確保低於開盤價
            closes = opens + np.random.normal(0, 1, n_samples)        # 圍繞開盤價波動
            
            # 確保價格邏輯關係
            for i in range(n_samples):
                # 確保 high 是最高的
                high = max(opens[i], highs[i], closes[i])
                # 確保 low 是最低的
                low = min(opens[i], lows[i], closes[i])
                # 更新值
                highs[i] = high
                lows[i] = low
            
            data[market] = pd.DataFrame({
                'Open': opens,
                'High': highs,
                'Low': lows,
                'Close': closes,
                'Volume': abs(np.random.normal(1000000, 100000, n_samples))  # 確保成交量為正
            }, index=pd.date_range('2024-01-01', periods=n_samples))
            
            # 最後驗證價格邏輯
            assert (data[market]['High'] >= data[market]['Open']).all()
            assert (data[market]['High'] >= data[market]['Close']).all()
            assert (data[market]['Low'] <= data[market]['Open']).all()
            assert (data[market]['Low'] <= data[market]['Close']).all()
        
        return data

    def test_future_data_leakage(self, market_loader, multi_market_data):
        """測試未來數據泄露問題"""
        try:
            for market, data in multi_market_data.items():
                # 模擬移動平均計算
                ma_window = 5
                close_prices = data['Close']
                
                # 正確的移動平均（只使用過去數據）
                correct_ma = close_prices.rolling(window=ma_window, min_periods=1).mean()
                
                # 驗證沒有使用未來數據
                for i in range(len(close_prices)):
                    current_ma = correct_ma.iloc[i]
                    past_data = close_prices.iloc[max(0, i-ma_window+1):i+1]
                    expected_ma = past_data.mean()
                    assert abs(current_ma - expected_ma) < 1e-10, f"在位置 {i} 檢測到未來數據泄露"
                
                logger.info(f"市場 {market} 未來數據泄露檢查通過")
                
        except Exception as e:
            logger.error(f"未來數據泄露測試失敗: {str(e)}")
            raise

    def test_timezone_consistency(self, market_loader, multi_market_data):
        """測試時區一致性"""
        try:
            # 測試不同時區的數據
            timezones = ['America/New_York', 'Asia/Tokyo', 'Europe/London']
            
            for tz in timezones:
                # 創建帶時區的測試數據
                test_data = pd.DataFrame({
                    'Close': range(10),
                    'Volume': range(1000, 1100, 10)
                }, index=pd.date_range('2024-01-01', periods=10, tz=tz))
                
                processed_data = market_loader._clean_data(test_data)
                
                # 修改驗證方式
                for idx in processed_data.index:
                    assert idx.tzinfo is not None, "時間戳缺少時區信息"
                    assert (idx.tzinfo.tzname(None) == 'UTC' or 
                           isinstance(idx.tzinfo, pytz.UTC.__class__)), \
                        f"時區不是UTC (當前時區: {idx.tzinfo})"
                
                # 額外的時區一致性檢查
                first_tz = processed_data.index[0].tzinfo
                assert all(idx.tzinfo == first_tz for idx in processed_data.index), \
                    "數據中存在不同的時區"
                
                logger.info(f"時區 {tz} 轉換檢查通過")
            
            logger.info("時區一致性檢查通過")
            
        except Exception as e:
            logger.error(f"時區一致性測試失敗: {str(e)}")
            raise

    def test_data_synchronization(self, market_loader, multi_market_data):
        """測試多市場數據同步性"""
        try:
            # 檢查所有市場的數據時間戳是否對齊
            all_timestamps = set()
            for market, data in multi_market_data.items():
                all_timestamps.update(data.index)
            
            # 確保每個市場都有完整的時間序列
            for market, data in multi_market_data.items():
                missing_timestamps = all_timestamps - set(data.index)
                assert len(missing_timestamps) == 0, \
                    f"市場 {market} 缺少時間戳: {missing_timestamps}"
            
            logger.info("多市場數據同步性檢查通過")
            
        except Exception as e:
            logger.error(f"數據同步性測試失敗: {str(e)}")
            raise

    def test_data_quality_metrics(self, market_loader, multi_market_data):
        """測試數據質量指標"""
        try:
            for market, data in multi_market_data.items():
                # 計算數據質量指標
                quality_metrics = {
                    '缺失值比例': data.isnull().mean(),
                    '異常值比例': (
                        (data > data.mean() + 3 * data.std()) |
                        (data < data.mean() - 3 * data.std())
                    ).mean(),
                    '價格有效性': (
                        (data['High'] >= data['Open']) &
                        (data['High'] >= data['Close']) &
                        (data['Low'] <= data['Open']) &
                        (data['Low'] <= data['Close'])
                    ).mean()
                }
                
                # 驗證數據質量
                assert quality_metrics['缺失值比例'].max() < 0.01, \
                    f"市場 {market} 缺失值比例過高"
                assert quality_metrics['異常值比例'].max() < 0.05, \
                    f"市場 {market} 異常值比例過高"
                assert quality_metrics['價格有效性'].mean() > 0.99, \
                    f"市場 {market} 價格有效性不足"
                
                logger.info(f"市場 {market} 數據質量指標檢查通過")
                
        except Exception as e:
            logger.error(f"數據質量指標測試失敗: {str(e)}")
            raise

if __name__ == "__main__":
    pytest.main(['-v', __file__]) 