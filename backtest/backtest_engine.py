import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """交易記錄"""
    entry_date: pd.Timestamp
    entry_price: float
    position_size: float
    direction: int  # 1: 做多, -1: 做空
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    profit_loss: Optional[float] = None
    commission: float = 0.0
    slippage: float = 0.0
    holding_period: Optional[int] = None

class BacktestEngine:
    """回測引擎"""
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = pd.Series()
        self.trades: List[Trade] = []
        self.equity_curve = pd.Series()
        self.data = pd.DataFrame()
        
    def run_backtest(self, data: pd.DataFrame, strategy: BaseStrategy) -> Tuple[pd.Series, float]:
        """執行回測
        
        Args:
            data: 市場數據
            strategy: 交易策略
            
        Returns:
            equity_curve: 資金曲線
            final_capital: 最終資金
        """
        try:
            self.data = data.copy()
            self.positions = pd.Series(0, index=data.index)
            self.equity_curve = pd.Series(self.initial_capital, index=data.index, dtype=np.float64)
            
            # 生成交易信號
            signals = strategy.generate_signals(data)
            current_position = 0
            
            # 遍歷每個交易日
            for i in range(len(data)):
                current_price = data['Close'].iloc[i]
                
                # 更新資金曲線
                if current_position != 0:
                    pnl = current_position * (current_price - data['Close'].iloc[i-1])
                    self.current_capital += pnl
                
                # 處理交易信號
                if signals.iloc[i] != 0 and signals.iloc[i] != current_position:
                    # 計算新倉位大小
                    position_size = strategy.calculate_position(signals.iloc[i])
                    
                    # 計算交易成本
                    commission = abs(position_size * current_price * 0.001)  # 0.1% 手續費
                    slippage = abs(position_size * current_price * 0.001)   # 0.1% 滑點
                    
                    # 更新資金
                    self.current_capital -= (commission + slippage)
                    
                    # 記錄交易
                    trade = Trade(
                        entry_date=data.index[i],
                        entry_price=current_price,
                        position_size=position_size,
                        direction=signals.iloc[i],
                        commission=commission,
                        slippage=slippage
                    )
                    self.trades.append(trade)
                    current_position = position_size
                    
                # 更新持倉和資金曲線
                self.positions.iloc[i] = current_position
                self.equity_curve.iloc[i] = float(self.current_capital)
            
            # 平倉最後的持倉
            if current_position != 0:
                final_price = data['Close'].iloc[-1]
                commission = abs(current_position * final_price * 0.001)
                slippage = abs(current_position * final_price * 0.001)
                self.current_capital -= (commission + slippage)
                
                if self.trades:
                    self.trades[-1].exit_date = data.index[-1]
                    self.trades[-1].exit_price = final_price
                    self.trades[-1].profit_loss = (final_price - self.trades[-1].entry_price) * \
                                                self.trades[-1].position_size - \
                                                (self.trades[-1].commission + self.trades[-1].slippage)
            
            logger.info(f"回測完成，最終資金: {self.current_capital:.2f}")
            return self.equity_curve, self.current_capital
            
        except Exception as e:
            logger.error(f"回測執行失敗: {str(e)}")
            raise
            
    def calculate_metrics(self) -> Dict:
        """計算回測指標"""
        try:
            returns = self.equity_curve.pct_change().dropna()
            
            # 計算基本指標
            total_return = (self.current_capital / self.initial_capital) - 1
            annual_return = (1 + total_return) ** (252 / len(returns)) - 1
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
            
            # 計算最大回撤
            cummax = self.equity_curve.cummax()
            drawdown = (self.equity_curve - cummax) / cummax
            max_drawdown = drawdown.min()
            
            # 計算交易統計
            winning_trades = [t for t in self.trades if t.profit_loss and t.profit_loss > 0]
            win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'trade_count': len(self.trades),
                'final_capital': self.current_capital
            }
            
        except Exception as e:
            logger.error(f"計算績效指標失敗: {str(e)}")
            raise