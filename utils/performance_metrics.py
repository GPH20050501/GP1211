import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
import math
import logging
from .decorators import performance_monitor

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    position_size: float
    profit_loss: float
    commission: float
    slippage: float

@performance_monitor
def calculate_metrics(equity_curve: pd.Series,
                     trades: list,
                     risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    計算回測績效指標
    
    Args:
        equity_curve: 權益曲線
        trades: 交易記錄列表
        risk_free_rate: 無風險利率
        
    Returns:
        Dict[str, float]: 績效指標字典
    """
    try:
        # 1. 基礎收益指標
        total_return = (equity_curve[-1] / equity_curve[0]) - 1
        
        # 2. 年化收益率
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1
        
        # 3. 波動率
        daily_returns = equity_curve.pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)
        
        # 4. 夏普比率
        excess_returns = daily_returns - risk_free_rate/252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / daily_returns.std()
        
        # 5. 最大回撤
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax
        max_drawdown = drawdown.min()
        
        # 6. 交易統計
        if trades:
            profitable_trades = sum(1 for t in trades if t.profit_loss > 0)
            win_rate = profitable_trades / len(trades)
            
            profit_trades = [t.profit_loss for t in trades if t.profit_loss > 0]
            loss_trades = [t.profit_loss for t in trades if t.profit_loss <= 0]
            
            avg_profit = np.mean(profit_trades) if profit_trades else 0
            avg_loss = np.mean(loss_trades) if loss_trades else 0
            profit_factor = abs(sum(profit_trades) / sum(loss_trades)) if loss_trades else float('inf')
        else:
            win_rate = 0
            avg_profit = 0
            avg_loss = 0
            profit_factor = 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'trade_count': len(trades)
        }
        
    except Exception as e:
        logger.error(f"計算績效指標時發生錯誤: {str(e)}")
        raise

def save_metrics(metrics, strategy_name, file_path='performance_results.csv'):
    """
    保存績效指標到CSV文件
    """
    metrics['Strategy'] = strategy_name
    df = pd.DataFrame([metrics])
    
    try:
        # 如果文件存在，追加數據
        existing_df = pd.read_csv(file_path)
        df = pd.concat([existing_df, df], ignore_index=True)
    except FileNotFoundError:
        pass
    
    df.to_csv(file_path, index=False)
    return df

def format_metrics(metrics: Dict) -> Dict[str, Dict]:
    """
    將績效指標格式化並分類
    """
    formatted = {
        '資金績效': {
            '初始資金': f"{metrics['MMGT起始資本']:,.0f}",
            '最終資金': f"{metrics.get('final_capital', 0):,.0f}",
            '總報酬率': f"{metrics['獲利百分比']:.2f}%",
            '年化複合成長率': f"{metrics['年化複合成長率']:.2f}%",
            '資金使用效率': f"{metrics['資金使用效率']:.2f}%"
        },
        '風險指標': {
            '最大回撤': f"{metrics['最大回撤(%)']:.2f}%",
            '回報/風險比': f"{metrics['回報/風險比']:.2f}",
            '破產風險': f"{metrics['破產風險(%)']:.2f}%",
            '平均交易風險': f"{metrics['平均交易風險(%)']:.2f}%",
            '期望值E[R]': f"{metrics['期望值E[R]']:.2f}%"
        },
        '交易統計': {
            '總交易次數': f"{metrics.get('total_trades', 0)}次",
            '獲利次數': f"{metrics.get('winning_trades', 0)}次",
            '虧損次數': f"{metrics.get('losing_trades', 0)}次",
            '勝率': f"{metrics.get('win_rate', 0):.2f}%",
            '平均盈虧比': f"{metrics['平均盈虧比']:.2f}",
            '平均持倉天數': f"{metrics.get('avg_holding_period', 0)}天"
        },
        '資金管理': {
            'MMGT固定百分比': f"{metrics['MMGT固定百分比']:.2f}%",
            '平均淨收益': f"{metrics['平均淨收益']:,.2f}",
            '平均交易成本': f"{metrics['平均交易成本']:,.2f}",
            '平均淨收益(USD/M)': f"{metrics['平均淨收益(USD/M)']:,.2f}"
        }
    }
    return formatted

def _get_empty_metrics() -> Dict:
    """返回空的績效指標"""
    return {
        'MMGT起始資本': 0,
        'final_capital': 0,
        '獲利百分比': 0,
        '年化複合成長率': 0,
        '資金使用效率': 0,
        '最大回撤(%)': 0,
        '回報/風險比': 0,
        '破產風險(%)': 0,
        '平均交易風險(%)': 0,
        '期望值E[R]': 0,
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'win_rate': 0,
        '平均盈虧比': 0,
        'avg_holding_period': 0,
        'MMGT固定百分比': 0,
        '平均淨收益': 0,
        '平均交易成本': 0,
        '平均淨收益(USD/M)': 0
    }

def calculate_advanced_metrics(equity_curve: pd.Series, trades: List[Trade]) -> Dict:
    """計算進階績效指標"""
    try:
        # 1. 數據準備
        metrics = _get_empty_metrics()  # 使用空指標字典初始化
        returns = equity_curve.pct_change().fillna(0)
        initial_capital = float(equity_curve.iloc[0])
        final_capital = float(equity_curve.iloc[-1])
        
        # 設置基本資金指標
        metrics['MMGT起始資本'] = initial_capital
        metrics['final_capital'] = final_capital
        
        # 2. 交易統計指標
        if trades:
            # 計算每筆交易的收益
            trade_returns = [t.profit_loss for t in trades]
            trade_costs = [(t.commission + t.slippage) for t in trades]
            
            # 基礎交易統計
            metrics.update({
                'total_trades': len(trades),
                'winning_trades': len([t for t in trades if t.profit_loss > 0]),
                'losing_trades': len([t for t in trades if t.profit_loss <= 0]),
                'win_rate': len([t for t in trades if t.profit_loss > 0]) / len(trades) * 100 if trades else 0
            })
            
            # 平均淨收益
            avg_net_profit = np.mean(trade_returns) - np.mean(trade_costs)
            metrics['平均淨收益'] = round(avg_net_profit, 2)
            
            # 每筆交易平均手續費與滑價
            metrics['平均交易成本'] = round(np.mean(trade_costs), 2)
            
            # 獲利百分比
            total_profit = sum(trade_returns)
            metrics['獲利百分比'] = round((total_profit / initial_capital) * 100, 2)
            
            # 平均盈虧比
            avg_profit = np.mean([r for r in trade_returns if r > 0]) if any(r > 0 for r in trade_returns) else 0
            avg_loss = abs(np.mean([r for r in trade_returns if r < 0])) if any(r < 0 for r in trade_returns) else 1
            metrics['平均盈虧比'] = round(avg_profit / avg_loss, 2) if avg_loss != 0 else float('inf')
            
            # 資金使用效率
            avg_position_size = np.mean([t.position_size * t.entry_price for t in trades])
            metrics['資金使用效率'] = round((avg_position_size / initial_capital) * 100, 2)
            
            # MMGT策略固定百分比
            position_ratios = [(t.position_size * t.entry_price) / initial_capital * 100 for t in trades]
            metrics['MMGT固定百分比'] = round(np.mean(position_ratios), 2)
            
            # 每筆交易平均風險(%)
            max_drawdowns = []
            for t in trades:
                entry_value = t.position_size * t.entry_price
                exit_value = t.position_size * t.exit_price if t.exit_price else entry_value
                trade_drawdown = (min(entry_value, exit_value) - max(entry_value, exit_value)) / entry_value * 100
                max_drawdowns.append(abs(trade_drawdown))
            metrics['平均交易風險(%)'] = round(np.mean(max_drawdowns), 2) if max_drawdowns else 0
            
            # 平均淨收益$/USD(M)
            total_net_profit = sum(trade_returns) - sum(trade_costs)
            metrics['平均淨收益(USD/M)'] = round(total_net_profit / (initial_capital / 1_000_000), 2)
        
        # 3. 年化複合成長率
        try:
            start_date = pd.to_datetime(equity_curve.index[0])
            end_date = pd.to_datetime(equity_curve.index[-1])
            days = (end_date - start_date).days
            years = max(days / 365, 0.01)  # 避免除以零
            cagr = ((final_capital / initial_capital) ** (1/years) - 1) if initial_capital > 0 else 0
            metrics['年化複合成長率'] = round(cagr * 100, 2)
        except Exception as e:
            logging.warning(f"年化複合成長率計算失敗: {str(e)}")
            metrics['年化複合成長率'] = 0
            
        # 4. 其他風險指標
        try:
            # 期望值
            metrics['期望值E[R]'] = round(returns.mean() * 100, 2)
            
            # 波動率
            volatility = returns.std() * np.sqrt(252)
            
            # 回報風險比
            metrics['回報/風險比'] = round(metrics['年化複合成長率'] / (volatility * 100), 2) if volatility != 0 else 0
            
            # 最大回撤
            rolling_max = equity_curve.expanding().max()
            drawdown = (equity_curve - rolling_max) / rolling_max * 100
            metrics['最大回撤(%)'] = round(abs(drawdown.min()), 2) if not drawdown.empty else 0
            
            # 破產風險
            metrics['破產風險(%)'] = round(simulate_bankruptcy_risk(returns.values, initial_capital), 2)
            
        except Exception as e:
            logging.warning(f"風險指標計算失敗: {str(e)}")
            metrics.update({
                '期望值E[R]': 0,
                '回報/風險比': 0,
                '最大回撤(%)': 0,
                '破產風險(%)': 0
            })
        
        return format_metrics(metrics)
        
    except Exception as e:
        logging.error(f"進階指標計算失敗: {str(e)}")
        return format_metrics(_get_empty_metrics())

def simulate_bankruptcy_risk(returns, initial_capital, simulations=1000):
    """計算破產風險"""
    bankruptcy_count = 0
    for _ in range(simulations):
        capital = initial_capital
        sampled_returns = np.random.choice(returns, size=252)
        for r in sampled_returns:
            capital *= (1 + r)
            if capital <= initial_capital * 0.1:
                bankruptcy_count += 1
                break
    return (bankruptcy_count / simulations) * 100 