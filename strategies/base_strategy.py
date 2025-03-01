from typing import Dict, List
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BaseStrategy:
    """策略基礎類"""
    def __init__(self, parameters: Dict = None):
        self.position = 0
        self.signals = []
        self.parameters = parameters or {}
        self.name = "BaseStrategy"
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成交易信號"""
        raise NotImplementedError
        
    def calculate_position(self, signal: float) -> float:
        """計算倉位大小"""
        return signal * self.parameters.get('position_size', 1.0)
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """驗證輸入數據"""
        try:
            required_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
            if not all(col in data.columns for col in required_columns):
                raise ValueError("數據缺少必要的列")
            return True
        except Exception as e:
            logger.error(f"數據驗證失敗: {str(e)}")
            return False 