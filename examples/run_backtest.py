import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.market_data import MultiMarketDataLoader
from strategies.ma_cross_strategy import MACrossStrategy
from backtest.backtest_engine import BacktestEngine
from visualization.plotter import BacktestPlotter
from reporting.report_generator import ReportGenerator, BacktestReport
from datetime import datetime
import logging

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        metrics = engine.calculate_metrics()
        backtest_report = BacktestReport(
            strategy_name=strategy.__class__.__name__,
            start_date=datetime.strptime('2023-01-01', '%Y-%m-%d'),
            end_date=datetime.strptime('2024-01-01', '%Y-%m-%d'),
            initial_capital=engine.initial_capital,
            final_capital=final_capital,
            total_return=metrics['total_return'],
            annual_return=metrics['annual_return'],
            sharpe_ratio=metrics['sharpe_ratio'],
            max_drawdown=metrics['max_drawdown'],
            win_rate=metrics['win_rate'],
            trade_count=len(engine.trades),
            trades=engine.trades,
            metrics=metrics
        )
        
        report_generator = ReportGenerator()
        report_path = report_generator.generate_report(backtest_report)
        
        logger.info(f"回測完成！報告已保存至: {report_path}")
        logger.info(f"最終資金: {final_capital:,.2f}")
        
        # 6. 顯示主要績效指標
        print("\n=== 回測結果摘要 ===")
        print(f"總收益率: {metrics['total_return']:.2%}")
        print(f"年化收益率: {metrics['annual_return']:.2%}")
        print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
        print(f"最大回撤: {metrics['max_drawdown']:.2%}")
        print(f"勝率: {metrics['win_rate']:.2%}")
        print(f"交易次數: {len(engine.trades)}")
        
    except Exception as e:
        logger.error(f"回測過程中發生錯誤: {str(e)}")
        raise

if __name__ == "__main__":
    main() 