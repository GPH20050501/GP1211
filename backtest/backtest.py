import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from utils.performance_metrics import calculate_metrics
from utils.decorators import performance_monitor
import logging
from .capital_management import CapitalManager, PositionConfig
from .risk_management import RiskManager

@dataclass
class Trade:
    """交易記錄類"""
    entry_date: pd.Timestamp  # 進場時間
    entry_price: float       # 進場價格
    position_size: float     # 持倉數量
    commission: float        # 手續費
    slippage: float         # 滑價
    exit_date: Optional[pd.Timestamp] = None  # 出場時間
    exit_price: Optional[float] = None        # 出場價格
    profit_loss: float = 0.0                  # 獲利/虧損
    holding_period: int = 0                   # 持倉天數

    def __post_init__(self):
        """確保時間格式的一致性"""
        if isinstance(self.entry_date, str):
            self.entry_date = pd.to_datetime(self.entry_date)
        if isinstance(self.exit_date, str) and self.exit_date is not None:
            self.exit_date = pd.to_datetime(self.exit_date)

    def to_dict(self) -> dict:
        """轉換為字典格式，用於創建DataFrame"""
        return {
            '進場時間': self.entry_date,
            '進場價格': self.entry_price,
            '出場時間': self.exit_date if self.exit_date is not None else pd.NaT,
            '出場價格': self.exit_price if self.exit_price is not None else np.nan,
            '持倉數量': self.position_size,
            '獲利/虧損': self.profit_loss,
            '手續費': self.commission,
            '滑價成本': self.slippage,
            '持倉天數': self.holding_period
        }

class BacktestEngine:
    """回測引擎"""
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trades = []
        self.commission = 0.001  # 手續費率
        self.slippage = 0.001   # 滑價率
        
        self.risk_manager = RiskManager(
            max_position_size=0.1,      # 最大持倉比例
            max_drawdown=0.2,           # 最大回撤限制
            daily_var_limit=0.05        # 日VaR限制
        )
        
    def _calculate_position(self, 
                          signal: float, 
                          current_price: float,
                          available_capital: float,
                          current_position: float = 0.0) -> float:
        """
        計算目標倉位
        
        Args:
            signal: 交易信號
            current_price: 當前價格
            available_capital: 可用資金
            current_position: 當前持倉
            
        Returns:
            float: 目標倉位數量
        """
        try:
            if signal == 0:
                return current_position
                
            # 計算基礎倉位大小
            position_size = self._calculate_position_size(available_capital, current_price)
            
            # 根據信號調整倉位方向
            target_position = position_size if signal > 0 else -position_size
            
            # 檢查風險限制
            position_value = abs(target_position * current_price)
            if not self.risk_manager.check_position_limit(position_value, self.current_capital):
                logger.warning("超過倉位限制，調整倉位大小")
                target_position *= (self.risk_manager.max_position_size * self.current_capital) / position_value
                
            return target_position
            
        except Exception as e:
            logger.error(f"計算倉位時發生錯誤: {str(e)}")
            return current_position
            
    def _calculate_position_size(self, available_cash: float, price: float) -> float:
        """計算倉位大小"""
        position = available_cash * 0.95 / price  # 使用95%的可用資金
        return float(position)

    def run_backtest(self, data: pd.DataFrame, signals: pd.Series) -> tuple:
        """優化回測過程的記憶體使用"""
        # 使用生成器處理大量數據
        def calculate_positions():
            for date, signal in signals.items():
                if signal != 0:
                    yield date, self._calculate_position(signal, float(data.loc[date]['Close']), self.current_capital)
        
        positions = pd.Series(dict(calculate_positions()))
        return self._calculate_equity_curve(positions)
    
    def _record_trade(self, 
                     market: str, 
                     direction: str, 
                     price: float, 
                     size: float):
        """記錄交易"""
        self.trade_history.append({
            'market': market,
            'direction': direction,
            'price': price,
            'size': size,
            'value': price * size
        })

class Backtest:
    def __init__(self, data: pd.DataFrame, initial_capital: float = 1000000, 
                 commission: float = 0.001, slippage: float = 0.001):
        """
        初始化回測系統
        """
        self.data = data.copy()
        # 將所有資金相關的列初始化為 float64 類型
        self.positions = pd.Series(0.0, index=data.index, dtype='float64')
        self.holdings = pd.Series(0.0, index=data.index, dtype='float64')
        self.cash = pd.Series(float(initial_capital), index=data.index, dtype='float64')
        self.total_value = pd.Series(float(initial_capital), index=data.index, dtype='float64')
        
        self.commission = commission
        self.slippage = slippage
        self.trades = []
        
        # 交易記錄
        self.current_trade: Optional[Trade] = None
        
        # 績效指標
        self.metrics: Dict = {}
        
    def calculate_metrics(self) -> Dict:
        """計算回測指標"""
        metrics = {
            '初始資金': self.cash.iloc[0],
            '最終資金': self.total_value.iloc[-1],
            '總報酬率': 0.0,
            '年化收益率': 0.0,
            '最大回撤': 0.0,
            '夏普比率': 0.0,
            '交易次數': 0,
            '獲利交易次數': 0,
            '虧損交易次數': 0,
            '勝率': 0.0,
            '平均獲利': 0.0,
            '平均虧損': 0.0,
            '獲利因子': 0.0,
            '波動率': 0.0,
            '交易成本比率': 0.0,
            '平均持倉天數': 0.0
        }
        
        if len(self.trades) == 0:
            return metrics
        
        final_capital = self.total_value.iloc[-1]
        total_return = ((final_capital - self.cash.iloc[0]) / self.cash.iloc[0]) * 100
        
        # 計算年化收益率
        days = (self.data.index[-1] - self.data.index[0]).days
        annual_return = (((final_capital / self.cash.iloc[0]) ** (365/days)) - 1) * 100
        
        # 計算交易統計
        profitable_trades = [t for t in self.trades if t.profit_loss > 0]
        losing_trades = [t for t in self.trades if t.profit_loss <= 0]
        
        metrics.update({
            '總報酬率': total_return,
            '年化收益率': annual_return,
            '交易次數': len(self.trades),
            '獲利交易次數': len(profitable_trades),
            '虧損交易次數': len(losing_trades),
            '勝率': (len(profitable_trades) / len(self.trades) * 100) if self.trades else 0.0,
            '平均獲利': np.mean([t.profit_loss for t in profitable_trades]) if profitable_trades else 0.0,
            '平均虧損': np.mean([t.profit_loss for t in losing_trades]) if losing_trades else 0.0,
            '獲利因子': (sum(t.profit_loss for t in profitable_trades) / abs(sum(t.profit_loss for t in losing_trades))) 
                      if losing_trades and sum(t.profit_loss for t in losing_trades) != 0 else 0.0,
            '平均持倉天數': np.mean([t.holding_period for t in self.trades]) if self.trades else 0.0
        })
        
        # 計算風險指標
        returns = self.total_value.pct_change().dropna()
        metrics['波動率'] = returns.std() * np.sqrt(252) * 100
        
        # 計算最大回撤
        cummax = self.total_value.cummax()
        drawdown = (self.total_value - cummax) / cummax
        metrics['最大回撤'] = abs(drawdown.min()) * 100
        
        # 計算夏普比率
        risk_free_rate = 0.02
        excess_returns = returns - risk_free_rate/252
        metrics['夏普比率'] = np.sqrt(252) * excess_returns.mean() / returns.std() if len(returns) > 0 and returns.std() != 0 else 0
        
        return metrics

    def run_strategy(self, signals: pd.Series) -> Dict:
        """執行回測策略"""
        try:
            current_position = 0
            
            for i in range(len(self.data)):
                current_price = float(self.data['Close'].iloc[i])
                
                # 處理買入信號
                if signals.iloc[i] == 1 and current_position == 0:
                    # 計算可買入的數量
                    available_cash = float(self.cash.iloc[i])
                    position_size = self._calculate_position_size(available_cash, current_price)
                    
                    # 記錄交易
                    trade = Trade(
                        entry_date=self.data.index[i],
                        entry_price=current_price,
                        position_size=position_size,
                        commission=position_size * current_price * self.commission,
                        slippage=position_size * current_price * self.slippage
                    )
                    self.trades.append(trade)
                    current_position = position_size
                    
                    # 更新持倉
                    self.positions.iloc[i:] = position_size
                    
                # 處理賣出信號
                elif signals.iloc[i] == -1 and current_position > 0:
                    # 更新最後一筆交易的出場資訊
                    if self.trades:
                        trade = self.trades[-1]
                        trade.exit_date = self.data.index[i]
                        trade.exit_price = current_price
                        trade.profit_loss = (current_price - trade.entry_price) * trade.position_size - \
                                          (trade.commission + trade.slippage)
                        trade.holding_period = (trade.exit_date - trade.entry_date).days
                    
                    current_position = 0
                    self.positions.iloc[i:] = 0
                
                # 更新持倉價值和現金
                self.holdings.iloc[i] = current_position * current_price
                
                if i > 0:
                    position_change = self.positions.iloc[i] - self.positions.iloc[i-1]
                    if position_change != 0:
                        transaction_cost = self._calculate_transaction_cost(
                            abs(position_change), current_price
                        )
                        self.cash.iloc[i:] = self.cash.iloc[i] - \
                                           (position_change * current_price + transaction_cost)
                
                # 更新總價值
                self.total_value.iloc[i] = self.cash.iloc[i] + self.holdings.iloc[i]
            
            # 計算績效指標
            self.metrics = self.calculate_metrics()
            
            return self.get_results()
            
        except Exception as e:
            logging.error(f"回測執行過程中發生錯誤: {str(e)}")
            raise

    def _calculate_transaction_cost(self, shares: float, price: float) -> float:
        """計算交易成本"""
        commission_cost = shares * price * self.commission
        slippage_cost = shares * price * self.slippage
        return float(commission_cost + slippage_cost)
    
    def get_trade_history(self) -> pd.DataFrame:
        """獲取詳細的交易歷史"""
        if not self.trades:
            return pd.DataFrame(columns=[
                '進場時間', '進場價格', '出場時間', '出場價格',
                '持倉數量', '獲利/虧損', '手續費', '滑價成本', '持倉天數'
            ])
        
        trade_records = [trade.to_dict() for trade in self.trades]
        return pd.DataFrame(trade_records)
    
    def get_results(self) -> Dict:
        """獲取回測結果"""
        return {
            'Positions': self.positions,
            'Holdings': self.holdings,
            'Cash': self.cash,
            'Total Value': self.total_value,
            'Returns': self.total_value.pct_change(),
            'Metrics': self.metrics,
            'Trade History': self.get_trade_history()
        } 