import pytest
import os
import importlib
from pathlib import Path

class TestSystemIntegrity:
    @pytest.fixture
    def project_root(self):
        return Path(__file__).parent.parent

    def test_required_files_exist(self, project_root):
        """檢查必要文件是否存在"""
        required_files = [
            'README.md',
            'SPEC.md',
            'requirements.txt',
            'data/market_data.py',
            'strategies/ma_cross_strategy.py',
            'backtest/backtest.py',
            'utils/performance.py'
        ]
        
        for file_path in required_files:
            assert (project_root / file_path).exists(), f"缺少必要文件: {file_path}"

    def test_module_imports(self):
        """檢查所有模組是否可以正確導入"""
        modules = [
            'data.market_data',
            'strategies.ma_cross_strategy',
            'backtest.backtest',
            'utils.performance'
        ]
        
        for module in modules:
            try:
                importlib.import_module(module)
            except ImportError as e:
                pytest.fail(f"模組 {module} 導入失敗: {str(e)}") 