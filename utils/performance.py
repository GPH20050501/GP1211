import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """績效分析器"""
    
    def __init__(self):
        self.metrics = {}
        
    def calculate_returns(self, equity_curve: pd.Series) -> pd.Series:
        """計算收益率"""
        return equity_curve.pct_change().dropna()
        
    def calculate_metrics(self, equity_curve: pd.Series) -> Dict:
        """計算績效指標"""
        try:
            returns = self.calculate_returns(equity_curve)
            
            # 計算基本指標
            total_return = (equity_curve[-1] / equity_curve[0]) - 1
            annual_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
            
            # 計算最大回撤
            cummax = equity_curve.cummax()
            drawdown = (equity_curve - cummax) / cummax
            max_drawdown = drawdown.min()
            
            self.metrics = {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"計算績效指標時發生錯誤: {str(e)}")
            raise 