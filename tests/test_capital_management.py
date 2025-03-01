import pytest
from backtest.capital_management import CapitalManager, PositionConfig

class TestCapitalManagement:
    def test_leverage_control(self):
        """测试杠杆控制机制"""
        config = {
            'US_Stock': PositionConfig(max_position_ratio=0.3),
            'Crypto': PositionConfig(max_position_ratio=0.1)
        }
        manager = CapitalManager(1e6, config)
        
        us_stock_size = manager.calculate_position_size(100, 'US_Stock')
        crypto_size = manager.calculate_position_size(100, 'Crypto')
        
        assert us_stock_size == 3000  # 1e6 * 0.3 / 100
        assert crypto_size == 1000    # 1e6 * 0.1 / 100

    def test_invalid_configuration(self):
        """测试无效配置检测"""
        with pytest.raises(ValueError):
            # 无效的最大仓位比例
            PositionConfig(max_position_ratio=1.5)
            
        with pytest.raises(ValueError):
            # 无效的仓位调整步长
            PositionConfig(max_position_ratio=0.5, position_step=0)

    def test_position_config_validation(self):
        """測試倉位配置驗證"""
        with pytest.raises(ValueError):
            PositionConfig(
                max_position_ratio=1.5,  # 超過最大限制
                min_position_ratio=0.05  # 低於最小限制
            ) 