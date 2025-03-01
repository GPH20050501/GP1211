import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import itertools
import logging
from utils.performance_metrics import calculate_advanced_metrics

@dataclass
class Trade:
    """交易記錄類"""
    entry_date: pd.Timestamp
    entry_price: float
    position_size: float
    commission: float
    slippage: float
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    profit_loss: float = 0.0
    holding_period: int = 0

def calculate_metrics(history: pd.Series, trades: List[Trade]) -> Dict:
    """計算完整的績效指標"""
    try:
        # 1. 數據修復和驗證
        if not isinstance(history, pd.Series):
            history = pd.Series(history)
        
        # 確保數據不為空
        if len(history) == 0:
            logging.warning("資金曲線為空，返回預設值")
            return _get_empty_metrics()
            
        # 確保有日期索引
        if not isinstance(history.index, pd.DatetimeIndex):
            try:
                # 嘗試將現有索引轉換為日期索引
                history.index = pd.to_datetime(history.index)
            except Exception as e:
                logging.warning(f"無法將現有索引轉換為日期索引: {str(e)}")
                # 創建新的日期索引
                history.index = pd.date_range(
                    start=pd.Timestamp.now().normalize() - pd.Timedelta(days=len(history)-1),
                    periods=len(history),
                    freq='D'
                )
            logging.info("已修復資金曲線日期索引")
        
        # 2. 計算基礎指標
        try:
            basic_metrics = _calculate_basic_metrics(history, trades)
        except Exception as e:
            logging.warning(f"基礎指標計算失敗: {str(e)}")
            basic_metrics = _get_empty_metrics()
            
        # 3. 計算進階指標
        try:
            advanced_metrics = calculate_advanced_metrics(history, trades)
        except Exception as e:
            logging.warning(f"進階指標計算失敗: {str(e)}")
            advanced_metrics = {
                '年化複合成長率': 0,
                '期望值E[R]': 0,
                '破產風險(%)': 0,
                '回報/風險比': 0,
                '最大回撤(%)': 0
            }
            
        # 4. 合併所有指標
        metrics = {**basic_metrics, **advanced_metrics}
        
        return metrics
        
    except Exception as e:
        logging.error(f"績效指標計算過程發生錯誤: {str(e)}")
        return _get_empty_metrics()

def _calculate_basic_metrics(history: pd.Series, trades: List[Trade]) -> Dict:
    """計算基礎績效指標"""
    try:
        # 1. 基礎計算
        initial_capital = float(history.iloc[0])
        final_capital = float(history.iloc[-1])
        
        # 2. 收益計算
        total_return = final_capital - initial_capital
        total_return_pct = (total_return / initial_capital * 100) if initial_capital != 0 else 0
        
        # 3. 交易統計
        metrics = {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return_pct': round(total_return_pct, 2)
        }
        
        # 4. 交易相關指標
        if trades:
            winning_trades = [t for t in trades if t.profit_loss > 0]
            losing_trades = [t for t in trades if t.profit_loss <= 0]
            
            metrics.update({
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': round(len(winning_trades) / len(trades) * 100, 2) if trades else 0,
                'avg_profit_per_trade': round(np.mean([t.profit_loss for t in winning_trades]), 2) if winning_trades else 0,
                'avg_loss_per_trade': round(np.mean([t.profit_loss for t in losing_trades]), 2) if losing_trades else 0,
                'profit_loss_ratio': round(abs(np.mean([t.profit_loss for t in winning_trades]) / 
                                             np.mean([t.profit_loss for t in losing_trades])), 2) 
                                    if losing_trades and np.mean([t.profit_loss for t in losing_trades]) != 0 else 0,
                'avg_holding_period': round(np.mean([t.holding_period for t in trades]), 1) if trades else 0
            })
        
        return metrics
        
    except Exception as e:
        logging.error(f"基礎指標計算失敗: {str(e)}")
        return _get_empty_metrics()

def _get_empty_metrics():
    """返回空的績效指標"""
    return {
        'initial_capital': 0,
        'final_capital': 0,
        'total_return_pct': 0,
        'annual_return': 0,
        'max_drawdown': 0,
        'win_rate': 0,
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'avg_profit_per_trade': 0,
        'avg_loss_per_trade': 0,
        'profit_loss_ratio': 0,
        'avg_holding_period': 0,
        'sharpe_ratio': 0,
        'sortino_ratio': 0,
        'calmar_ratio': 0
    } 