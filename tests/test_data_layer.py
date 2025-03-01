import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.market_data import MarketDataLoader
import logging

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestDataLayer:
    @pytest.fixture
    def market_loader(self):
        """初始化市場數據加載器"""
        return MarketDataLoader()
    
    @pytest.fixture
    def sample_data(self):
        """生成測試數據"""
        dates = pd.date_range('2024-01-01', periods=10)
        return pd.DataFrame({
            'Open': [100, 101, np.nan, 103, 104, 105, 106, 107, 108, 109],
            'High': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'Low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
            'Close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        }, index=dates)

    def test_data_loading(self, market_loader):
        """測試數據加載功能"""
        try:
            data = market_loader.load_data('AAPL', '2024-01-01', '2024-01-10')
            logger.info(f"成功加載數據，共 {len(data)} 條記錄")
            
            # 驗證數據結構
            assert isinstance(data, pd.DataFrame)
            assert all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
            assert isinstance(data.index, pd.DatetimeIndex)
            
        except Exception as e:
            logger.error(f"數據加載失敗: {str(e)}")
            raise

    def test_data_cleaning(self, market_loader, sample_data):
        """測試數據清洗功能"""
        try:
            cleaned_data = market_loader._clean_data(sample_data)
            logger.info(f"數據清洗前: {len(sample_data)} 條，清洗後: {len(cleaned_data)} 條")
            
            # 驗證清洗結果
            assert not cleaned_data.isnull().any().any(), "清洗後數據仍存在空值"
            assert all(cleaned_data['High'] >= cleaned_data['Low']), "價格邏輯錯誤"
            assert all(cleaned_data['Volume'] > 0), "成交量非法"
            
        except Exception as e:
            logger.error(f"數據清洗失敗: {str(e)}")
            raise

    def test_price_validation(self, market_loader, sample_data):
        """測試價格數據驗證"""
        # 添加一些異常價格
        sample_data.loc[sample_data.index[5], 'High'] = 1000  # 異常高價
        sample_data.loc[sample_data.index[6], 'Low'] = 0     # 異常低價
        
        cleaned_data = market_loader._clean_data(sample_data)
        
        # 檢查價格邏輯
        price_valid = (
            (cleaned_data['High'] >= cleaned_data['Open']).all() and
            (cleaned_data['High'] >= cleaned_data['Close']).all() and
            (cleaned_data['Low'] <= cleaned_data['Open']).all() and
            (cleaned_data['Low'] <= cleaned_data['Close']).all()
        )
        
        assert price_valid, "價格數據邏輯關係錯誤"

    def test_volume_validation(self, market_loader, sample_data):
        """測試成交量數據驗證"""
        try:
            # 添加異常成交量
            sample_data.loc[sample_data.index[5], 'Volume'] = -1000  # 負成交量
            sample_data.loc[sample_data.index[6], 'Volume'] = 0      # 零成交量
            
            cleaned_data = market_loader._clean_data(sample_data)
            
            # 驗證成交量
            assert all(cleaned_data['Volume'] > 0), "存在非正成交量"
            assert not any(cleaned_data['Volume'] > sample_data['Volume'].mean() * 10), "未處理異常大成交量"
            
        except Exception as e:
            logger.error(f"成交量驗證失敗: {str(e)}")
            raise

    def test_time_series_continuity(self, market_loader):
        """測試時間序列連續性"""
        try:
            # 創建不連續的時間序列數據
            dates = pd.date_range('2024-01-01', periods=5)
            dates = dates.append(pd.date_range('2024-01-07', periods=5))
            
            data = pd.DataFrame({
                'Close': range(10),
                'Volume': range(1000, 1100, 10)
            }, index=dates)
            
            processed_data = market_loader._handle_time_series(data)
            
            # 驗證時間序列
            date_diffs = processed_data.index.to_series().diff().dropna()
            assert all(date_diffs == pd.Timedelta(days=1)), "時間序列不連續"
            
        except Exception as e:
            logger.error(f"時間序列驗證失敗: {str(e)}")
            raise

if __name__ == "__main__":
    # 執行測試
    pytest.main(['-v', __file__]) 