import pytest
import pandas as pd
import numpy as np
from backtest.risk_management import RiskManager

class TestRiskManagement:
    @pytest.fixture
    def risk_manager(self):
        return RiskManager(
            max_position_size=0.1,
            max_drawdown=0.2,
            daily_var_limit=0.05,
            stop_loss=0.05,
            take_profit=0.1
        )

    def test_dynamic_position_sizing(self, risk_manager):
        """测试波动率调整的仓位计算"""
        # 高波动率市场
        high_vol = risk_manager.calculate_position_size(
            price=100,
            volatility=0.3,
            available_capital=100000
        )
        # 低波动率市场
        low_vol = risk_manager.calculate_position_size(
            price=100,
            volatility=0.1,
            available_capital=100000
        )
        assert high_vol < low_vol, "高波动率时应减少仓位"

    def test_drawdown_monitoring(self, risk_manager):
        """测试回撤监控机制"""
        equity_curve = pd.Series([1e6, 1.2e6, 9e5, 8e5])
        positions = pd.Series([0, 1e5, 1e5, 0])
        
        # 更新风险指标
        metrics = risk_manager.update_risk_metrics(
            equity_curve=equity_curve,
            positions=positions,
            current_price=100
        )
        
        assert metrics['current_drawdown'] == pytest.approx(0.2, 0.1)
        assert not risk_manager.check_risk_limits(), "应触发最大回撤限制"

    def test_stop_loss_mechanism(self, risk_manager):
        """测试止损止盈逻辑"""
        # 测试多头止损
        assert risk_manager.should_close_position(
            entry_price=100,
            current_price=95,
            position_type='long'
        ), "应触发多头止损"
        
        # 测试空头止盈
        assert risk_manager.should_close_position(
            entry_price=100,
            current_price=90,
            position_type='short'
        ), "应触发空头止盈" 