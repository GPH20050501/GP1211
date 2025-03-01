import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class BacktestReport:
    """回測報告數據結構"""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    trade_count: int
    trades: List
    metrics: Dict

class ReportGenerator:
    """回測報告生成器"""
    def __init__(self, report_dir: str = "reports"):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(exist_ok=True)
        
    def generate_report(self, backtest_report: BacktestReport) -> str:
        """生成HTML格式的回測報告"""
        try:
            # 創建報告內容
            html_content = f"""
            <html>
            <head>
                <title>回測報告 - {backtest_report.strategy_name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #f8f9fa; padding: 20px; }}
                    .metrics {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }}
                    .metric-card {{ background-color: #fff; padding: 15px; border: 1px solid #ddd; }}
                    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                    th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>{backtest_report.strategy_name} - 回測報告</h1>
                    <p>回測期間: {backtest_report.start_date.date()} 至 {backtest_report.end_date.date()}</p>
                </div>
                
                <h2>績效指標</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <h3>收益指標</h3>
                        <p>初始資金: {backtest_report.initial_capital:,.2f}</p>
                        <p>最終資金: {backtest_report.final_capital:,.2f}</p>
                        <p>總收益率: {backtest_report.total_return:.2%}</p>
                        <p>年化收益率: {backtest_report.annual_return:.2%}</p>
                    </div>
                    <div class="metric-card">
                        <h3>風險指標</h3>
                        <p>夏普比率: {backtest_report.sharpe_ratio:.2f}</p>
                        <p>最大回撤: {backtest_report.max_drawdown:.2%}</p>
                        <p>勝率: {backtest_report.win_rate:.2%}</p>
                        <p>交易次數: {backtest_report.trade_count}</p>
                    </div>
                </div>
                
                <h2>交易記錄</h2>
                <table>
                    <tr>
                        <th>入場時間</th>
                        <th>入場價格</th>
                        <th>方向</th>
                        <th>出場時間</th>
                        <th>出場價格</th>
                        <th>收益</th>
                    </tr>
            """
            
            # 添加交易記錄
            for trade in backtest_report.trades:
                direction = "做多" if trade.direction > 0 else "做空"
                profit_loss = trade.profit_loss if trade.profit_loss else "未平倉"
                
                # 先計算格式化的值
                exit_price_str = f"{trade.exit_price:.2f}" if trade.exit_price else "-"
                profit_loss_str = f"{profit_loss:.2f}" if isinstance(profit_loss, (int, float)) else profit_loss
                
                html_content += f"""
                    <tr>
                        <td>{trade.entry_date.date()}</td>
                        <td>{trade.entry_price:.2f}</td>
                        <td>{direction}</td>
                        <td>{trade.exit_date.date() if trade.exit_date else '-'}</td>
                        <td>{exit_price_str}</td>
                        <td>{profit_loss_str}</td>
                    </tr>
                """
            
            html_content += """
                </table>
            </body>
            </html>
            """
            
            # 保存報告
            report_path = self.report_dir / f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"報告已生成: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"生成報告失敗: {str(e)}")
            raise 