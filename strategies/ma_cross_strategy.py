import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Iterator
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SignalEvent:
    """交易信號事件"""
    timestamp: pd.Timestamp
    signal_type: str  # 'buy' 或 'sell'
    strength: float = 1.0

class MACrossStrategy:
    """移動平均線交叉策略"""
    
    def __init__(self, 
                 short_window: int = 5, 
                 long_window: int = 20,
                 parameters: Optional[Dict] = None):
        """
        初始化策略
        
        Args:
            short_window: 短期移動平均線週期
            long_window: 長期移動平均線週期
            parameters: 其他策略參數
        """
        self.short_window = short_window
        self.long_window = long_window
        self._parameters = parameters or {
            'max_position': 1.0,
            'stop_loss': 0.02,
            'take_profit': 0.05
        }
        self.position = 0
        self.signals = []
        self.name = f"MA Cross ({short_window}, {long_window})"
        
    @property
    def parameters(self) -> Dict:
        """獲取策略參數"""
        return self._parameters
        
    def items(self) -> Iterator[Tuple[str, float]]:
        """
        實現類似字典的 items() 方法
        
        Returns:
            Iterator: 參數名稱和值的迭代器
        """
        yield ('short_window', self.short_window)
        yield ('long_window', self.long_window)
        for key, value in self._parameters.items():
            yield (key, value)
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        計算技術指標
        
        Args:
            data: 市場數據
            
        Returns:
            pd.DataFrame: 包含技術指標的數據
        """
        df = data.copy()
        
        # 計算移動平均線
        df['MA_short'] = df['Close'].rolling(window=self.short_window).mean()
        df['MA_long'] = df['Close'].rolling(window=self.long_window).mean()
        
        # 計算交叉信號
        df['Signal'] = 0
        df.loc[df['MA_short'] > df['MA_long'], 'Signal'] = 1
        df.loc[df['MA_short'] < df['MA_long'], 'Signal'] = -1
        
        return df
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        生成交易信號
        
        Args:
            data: 市場數據
            
        Returns:
            pd.Series: 交易信號
        """
        df = self.calculate_indicators(data)
        
        # 檢測交叉點
        signals = pd.Series(0, index=df.index)
        signal_changes = df['Signal'].diff()
        
        # 買入信號
        signals[signal_changes > 0] = 1
        
        # 賣出信號
        signals[signal_changes < 0] = -1
        
        # 記錄信號統計
        long_signals = (signals == 1).sum()
        short_signals = (signals == -1).sum()
        logger.info(f"生成了 {len(signals[signals != 0])} 個交易信號")
        logger.info(f"做多信號: {long_signals}")
        logger.info(f"做空信號: {short_signals}")
        
        return signals
        
    def should_trade(self, 
                    current_price: float, 
                    current_position: float,
                    available_capital: float) -> Tuple[int, float]:
        """
        判斷是否應該交易
        
        Args:
            current_price: 當前價格
            current_position: 當前持倉
            available_capital: 可用資金
            
        Returns:
            Tuple[int, float]: (交易方向, 交易數量)
        """
        # 這裡可以添加更複雜的交易邏輯
        if current_position == 0:
            # 可以開新倉
            position_size = available_capital * self._parameters['max_position'] / current_price
            return (1, position_size)
        else:
            # 平倉
            return (-1, current_position)
            
    def update_position(self, signal: int, current_position: float) -> float:
        """
        更新持倉
        
        Args:
            signal: 交易信號
            current_position: 當前持倉
            
        Returns:
            float: 新的持倉
        """
        if signal == 0:
            return current_position
            
        if signal > 0 and current_position <= 0:
            return 1
        elif signal < 0 and current_position >= 0:
            return -1
        else:
            return current_position

    def calculate_position(self, signal: float) -> float:
        """計算倉位大小"""
        if signal == 0:
            return self.position
        return signal  # 1 代表全倉做多，-1 代表全倉做空
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """驗證輸入數據"""
        required_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
        return all(col in data.columns for col in required_columns)

def ma_cross_strategy(data: pd.DataFrame, 
                     short_window: int = 5, 
                     long_window: int = 20) -> pd.Series:
    """
    移動平均線交叉策略的函數版本
    
    Args:
        data: 市場數據
        short_window: 短期移動平均線週期
        long_window: 長期移動平均線週期
        
    Returns:
        pd.Series: 交易信號
    """
    strategy = MACrossStrategy(short_window, long_window)
    return strategy.generate_signals(data) 