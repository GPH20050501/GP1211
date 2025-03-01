import pytest
from datetime import datetime
from pathlib import Path
from reporting.report_generator import ReportGenerator, BacktestReport
from backtest.backtest_engine import Trade

class TestReporting:
    @pytest.fixture
    def sample_report(self):
        """生成測試用的回測報告數據"""
        return BacktestReport(
            strategy_name="測試策略",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            initial_capital=100000,
            final_capital=110000,
            total_return=0.1,
            annual_return=0.3,
            sharpe_ratio=1.5,
            max_drawdown=-0.05,
            win_rate=0.6,
            trade_count=10,
            trades=[
                Trade(
                    entry_date=datetime(2024, 1, 5),
                    entry_price=100.0,
                    position_size=1.0,
                    direction=1,
                    exit_date=datetime(2024, 1, 10),
                    exit_price=105.0,
                    profit_loss=5.0
                )
            ],
            metrics={}
        )
        
    def test_report_generation(self, sample_report, tmp_path):
        """測試報告生成"""
        generator = ReportGenerator(report_dir=str(tmp_path))
        report_path = generator.generate_report(sample_report)
        
        # 驗證報告文件
        assert Path(report_path).exists()
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert sample_report.strategy_name in content
            assert "回測報告" in content
            assert "績效指標" in content
            assert "交易記錄" in content
            
    def test_report_content(self, sample_report, tmp_path):
        """測試報告內容"""
        generator = ReportGenerator(report_dir=str(tmp_path))
        report_path = generator.generate_report(sample_report)
        
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 驗證關鍵指標
            assert f"{sample_report.total_return:.2%}" in content
            assert f"{sample_report.sharpe_ratio:.2f}" in content
            assert f"{sample_report.max_drawdown:.2%}" in content
            assert str(sample_report.trade_count) in content
            
            # 驗證交易記錄
            for trade in sample_report.trades:
                assert str(trade.entry_price) in content
                assert str(trade.exit_price) in content

if __name__ == "__main__":
    pytest.main(['-v', __file__]) 