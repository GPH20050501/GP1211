import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategies.ma_cross_strategy import MACrossStrategy
from strategies.base_strategy import BaseStrategy
import logging
import sys
import os
import time
from backtest.backtest_engine import BacktestEngine

logger = logging.getLogger(__name__)

class TestStrategy:
    @pytest.fixture
    def base_strategy(self):
        return BaseStrategy()
        
    @pytest.fixture
    def ma_strategy(self):
        return MACrossStrategy(short_window=5, long_window=20)
        
    @pytest.fixture
    def sample_data(self):
        """生成測試用的價格數據"""
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
        
    def test_base_strategy_initialization(self, base_strategy):
        """測試基礎策略初始化"""
        assert base_strategy.position == 0
        assert len(base_strategy.signals) == 0
        assert isinstance(base_strategy.parameters, dict)
        
    def test_ma_strategy_signals(self, ma_strategy, sample_data):
        """測試均線交叉策略信號生成"""
        try:
            signals = ma_strategy.generate_signals(sample_data)
            
            # 基本驗證
            assert isinstance(signals, pd.Series)
            assert len(signals) == len(sample_data)
            assert not signals.isna().any(), "信號中不應該有 NaN"
            assert signals.isin([0, 1, -1]).all(), "信號值應該只包含 -1, 0, 1"
            
            # 信號邏輯驗證
            signal_points = signals[signals != 0]
            if len(signal_points) > 0:
                # 驗證沒有連續信號
                for i in range(1, len(signals)):
                    if signals.iloc[i] != 0:
                        assert signals.iloc[i-1] == 0, \
                            f"在位置 {i} 存在連續信號"
            
            logger.info(f"生成了 {len(signal_points)} 個交易信號")
            logger.info("均線交叉策略信號測試通過")
            
        except Exception as e:
            logger.error(f"策略信號測試失敗: {str(e)}")
            raise
            
    def test_strategy_position_calculation(self, ma_strategy):
        """測試倉位計算"""
        test_signals = pd.Series([0, 1, -1, 0, 1])
        positions = []
        
        for signal in test_signals:
            pos = ma_strategy.calculate_position(signal)
            positions.append(float(pos))  # 確保轉換為 float
        
        # 驗證倉位計算
        assert all(isinstance(pos, float) for pos in positions)
        assert all(-1 <= pos <= 1 for pos in positions)
        
    def test_data_validation(self, ma_strategy, sample_data):
        """測試數據驗證"""
        # 正常數據
        assert ma_strategy.validate_data(sample_data)
        
        # 缺少必要列的數據
        invalid_data = sample_data.drop('Close', axis=1)
        assert not ma_strategy.validate_data(invalid_data)

    def test_ma_strategy_edge_cases(self, ma_strategy):
        """測試策略邊界情況"""
    # 創建測試數據
        dates = pd.date_range('2024-01-01', periods=10)
        data = pd.DataFrame({
            'Open': [100] * 10,
            'High': [110] * 10,
            'Low': [90] * 10,
            'Close': [100, 101, 102, 103, 104, 110, 115, 116, 117, 118],  # 大幅上漲
            'Volume': range(1000, 2000, 100)
        }, index=dates)
    
    # 生成信號
        signals = ma_strategy.generate_signals(data)
        
        # 驗證大幅上漲時的做多信號
        assert signals.iloc[5] == 1, "價格大幅上漲應產生做多信號"

    def test_strategy_risk_control(self, ma_strategy):
        """測試策略風險控制"""
        # 設置最大倉位限制
        ma_strategy.parameters['max_position'] = 1.0
        
        # 測試大信號
        large_signal = 2.0
        position = ma_strategy.calculate_position(large_signal)
        
        # 驗證倉位限制
        assert position <= ma_strategy.parameters['max_position'], \
            "倉位超過最大限制"
        assert position > 0, "倉位應該為正"

    def test_strategy_performance(self, ma_strategy, sample_data):
        """測試策略績效計算"""
        try:
            signals = ma_strategy.generate_signals(sample_data)
            
            # 計算策略收益
            returns = sample_data['Close'].pct_change()
            strategy_returns = signals.shift(1) * returns
            
            # 基本績效指標
            total_return = (1 + strategy_returns).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(returns)) - 1
            sharpe_ratio = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
            
            # 驗證績效指標的合理性
            assert not np.isnan(total_return), "總收益不應為 NaN"
            assert not np.isnan(sharpe_ratio), "夏普比率不應為 NaN"
            assert -1 <= total_return <= 10, "總收益應在合理範圍內"
            assert -5 <= sharpe_ratio <= 5, "夏普比率應在合理範圍內"
            
            logger.info(f"策略績效測試通過，總收益: {total_return:.2%}, 夏普比率: {sharpe_ratio:.2f}")
            
        except Exception as e:
            logger.error(f"策略績效測試失敗: {str(e)}")
            raise

    @staticmethod
    def test_critical_changes():
        """核心功能變更後的測試"""
        engine = BacktestEngine(initial_capital=100000)
        strategy = MACrossStrategy()
        
        # 測試倉位計算
        position = strategy.calculate_position(1)
        assert position > 0
        assert position <= 1.0  # 確保不超過最大倉位

def main():
    try:
        # 1. 加載數據
        logger.info("開始加載數據...")
        loader = MultiMarketDataLoader()
        market_data = loader.load_data(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2024-01-01'
        )
        
        if not market_data:
            raise ValueError("無法加載市場數據")
        
        # 2. 創建策略
        logger.info("初始化策略...")
        strategy = MACrossStrategy(short_window=5, long_window=20)
        
        # 3. 執行回測
        logger.info("開始回測...")
        engine = BacktestEngine(initial_capital=100000)
        equity_curve, final_capital = engine.run_backtest(
            data=market_data['AAPL'],
            strategy=strategy
        )
        
        # 4. 生成圖表
        logger.info("生成圖表...")
        plotter = BacktestPlotter(
            data=market_data['AAPL'],
            equity_curve=equity_curve,
            trades=engine.trades
        )
        fig = plotter.plot_results()
        fig.write_html("backtest_plot.html")
        
        # 5. 生成報告
        logger.info("生成報告...")
        report_generator = ReportGenerator()
        backtest_report = {
            'strategy_name': strategy.name,
            'start_date': datetime.strptime('2023-01-01', '%Y-%m-%d'),
            'end_date': datetime.strptime('2024-01-01', '%Y-%m-%d'),
            'initial_capital': engine.initial_capital,
            'final_capital': final_capital,
            'trades': engine.trades,
            **engine.calculate_metrics()
        }
        report_path = report_generator.generate_report(backtest_report)
        
        logger.info(f"回測完成！報告已保存至: {report_path}")
        logger.info(f"最終資金: {final_capital:,.2f}")
        
    except Exception as e:
        logger.error(f"回測過程中發生錯誤: {str(e)}")
        raise

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pytest.main(['-v', __file__])
    pytest.main(['-v', 'tests/test_backtest.py'])
    main()