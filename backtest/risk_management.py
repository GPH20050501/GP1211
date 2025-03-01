import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class RiskManager:
    """風險管理器"""
    
    def __init__(self,
                 max_position_size: float = 0.1,
                 max_drawdown: float = 0.2,
                 daily_var_limit: float = 0.05,
                 stop_loss: float = 0.02,
                 take_profit: float = 0.05):
        """
        初始化風險管理器
        
        Args:
            max_position_size: 最大持倉比例 (佔總資金)
            max_drawdown: 最大回撤限制
            daily_var_limit: 日VaR限制
            stop_loss: 止損比例
            take_profit: 止盈比例
        """
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.daily_var_limit = daily_var_limit
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        # 風險監控指標
        self.current_drawdown = 0.0
        self.peak_value = 0.0
        self.risk_metrics = {}
        
    def check_position_limit(self, 
                           position_value: float, 
                           total_capital: float) -> bool:
        """
        檢查持倉限制
        
        Args:
            position_value: 持倉市值
            total_capital: 總資金
            
        Returns:
            bool: 是否符合持倉限制
        """
        position_ratio = position_value / total_capital
        return position_ratio <= self.max_position_size
        
    def calculate_position_size(self,
                              price: float,
                              volatility: float,
                              available_capital: float) -> float:
        """
        計算建議持倉大小
        
        Args:
            price: 當前價格
            volatility: 波動率
            available_capital: 可用資金
            
        Returns:
            float: 建議持倉數量
        """
        # 基於波動率調整持倉
        vol_adjusted_ratio = self.max_position_size * (0.2 / volatility)  # 0.2為目標波動率
        position_ratio = min(vol_adjusted_ratio, self.max_position_size)
        
        # 計算持倉數量
        position_value = available_capital * position_ratio
        position_size = position_value / price
        
        return position_size
        
    def update_risk_metrics(self, 
                           equity_curve: pd.Series,
                           positions: pd.Series,
                           current_price: float) -> Dict:
        """
        更新風險指標
        
        Args:
            equity_curve: 權益曲線
            positions: 持倉序列
            current_price: 當前價格
            
        Returns:
            Dict: 風險指標字典
        """
        try:
            # 計算當前回撤
            self.peak_value = max(self.peak_value, equity_curve[-1])
            self.current_drawdown = (self.peak_value - equity_curve[-1]) / self.peak_value
            
            # 計算波動率
            returns = equity_curve.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            
            # 計算VaR
            var_95 = np.percentile(returns, 5) * np.sqrt(252)
            
            # 更新風險指標
            self.risk_metrics = {
                'current_drawdown': self.current_drawdown,
                'peak_value': self.peak_value,
                'volatility': volatility,
                'var_95': var_95,
                'position_exposure': (positions * current_price).sum()
            }
            
            return self.risk_metrics
            
        except Exception as e:
            logger.error(f"更新風險指標時發生錯誤: {str(e)}")
            return {}
            
    def check_risk_limits(self) -> bool:
        """
        檢查是否超過風險限制
        
        Returns:
            bool: 是否符合風險限制
        """
        if not self.risk_metrics:
            return True
            
        # 檢查回撤限制
        if self.current_drawdown > self.max_drawdown:
            logger.warning(f"超過最大回撤限制: {self.current_drawdown:.2%}")
            return False
            
        # 檢查VaR限制
        if abs(self.risk_metrics.get('var_95', 0)) > self.daily_var_limit:
            logger.warning(f"超過日VaR限制: {self.risk_metrics['var_95']:.2%}")
            return False
            
        return True
        
    def should_close_position(self,
                            entry_price: float,
                            current_price: float,
                            position_type: str) -> bool:
        """
        檢查是否應該平倉
        
        Args:
            entry_price: 進場價格
            current_price: 當前價格
            position_type: 持倉方向 ('long' 或 'short')
            
        Returns:
            bool: 是否應該平倉
        """
        if position_type == 'long':
            profit_ratio = (current_price - entry_price) / entry_price
            # 檢查止盈止損
            if profit_ratio <= -self.stop_loss or profit_ratio >= self.take_profit:
                return True
                
        elif position_type == 'short':
            profit_ratio = (entry_price - current_price) / entry_price
            # 檢查止盈止損
            if profit_ratio <= -self.stop_loss or profit_ratio >= self.take_profit:
                return True
                
        return False 