import os
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict

# 獲取專案根目錄
project_root = Path(__file__).parent.parent

# 將專案根目錄添加到 Python 路徑
sys.path.insert(0, str(project_root))

# 設置測試環境變量
os.environ['TESTING'] = 'true'

# 添加共用的 fixtures
@pytest.fixture(scope="session")
def sample_market_data() -> pd.DataFrame:
    """生成測試用的市場數據"""
    dates = pd.date_range('2024-01-01', periods=100)
    data = pd.DataFrame({
        'Open': np.random.normal(100, 10, 100),
        'High': np.random.normal(105, 10, 100),
        'Low': np.random.normal(95, 10, 100),
        'Close': np.random.normal(100, 10, 100),
        'Volume': np.random.normal(1000000, 100000, 100)
    }, index=dates)
    
    # 確保價格邏輯合理
    data['High'] = data[['Open', 'Close']].max(axis=1) + abs(np.random.normal(1, 0.1, 100))
    data['Low'] = data[['Open', 'Close']].min(axis=1) - abs(np.random.normal(1, 0.1, 100))
    
    return data

@pytest.fixture(scope="session")
def test_config() -> Dict:
    """測試配置"""
    return {
        'initial_capital': 100000,
        'commission_rate': 0.001,
        'slippage': 0.001,
        'risk_free_rate': 0.02,
        'max_position_size': 0.1
    }

@pytest.fixture
def mock_market_data_loader(monkeypatch):
    """Mock 市場數據加載器"""
    class MockMarketDataLoader:
        def load_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
            return sample_market_data()
            
    return MockMarketDataLoader()

# 添加日誌配置
import logging
@pytest.fixture(autouse=True)
def setup_logging():
    """設置測試日誌"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ) 