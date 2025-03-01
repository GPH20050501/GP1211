import pytest
from data.market_data import MarketDataLoader
from strategies.ma_cross_strategy import ma_cross_strategy
from backtest.backtest import BacktestEngine

def test_full_workflow():
    """測試完整工作流程"""
    # 1. 數據加載和清洗
    loader = MarketDataLoader()
    data = loader.load_data('AAPL', '2024-01-01', '2024-02-01')
    
    # 2. 生成策略信號
    signals = ma_cross_strategy(data)
    
    # 3. 執行回測
    engine = BacktestEngine(initial_capital=100000)
    equity_curve, final_capital = engine.run_backtest(data, signals)
    
    # 驗證整體結果
    assert data is not None
    assert signals is not None
    assert equity_curve is not None
    assert final_capital > 0 