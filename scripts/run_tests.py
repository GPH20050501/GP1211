import pytest
import sys
import os
import logging
from pathlib import Path
from typing import Dict, List
import time
import psutil
import argparse

logger = logging.getLogger(__name__)

class TestRunner:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_results: Dict = {}
        self.performance_metrics: Dict = {}
        
    def setup_logging(self):
        """配置日誌系統"""
        log_dir = self.project_root / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'test.log'),
                logging.StreamHandler()
            ]
        )

    def check_dependencies(self) -> bool:
        """檢查必要的套件"""
        required_packages = {
            'pytest': ('8.0.0', '9.0.0'),  # 當前版本 8.3.4
            'pytest-cov': ('4.1.0', '7.0.0'),  # 當前版本 6.0.0
            'pytest-benchmark': ('4.0.0', '6.0.0'),  # 當前版本 5.1.0
            'pandas': ('2.2.0', '2.3.0'),  # 當前版本 2.2.3
            'numpy': ('1.24.0', '3.0.0'),  # 當前版本 2.2.2
            'plotly': ('5.18.0', '7.0.0'),  # 當前版本 6.0.0
            'psutil': ('5.9.0', '7.0.0'),  # 當前版本 6.1.1
            'python-dotenv': ('1.0.0', '2.0.0')  # 當前版本 1.0.1
        }
        
        missing_packages = []
        version_mismatch = []
        
        for package, (min_version, max_version) in required_packages.items():
            try:
                module = __import__(package.replace('-', '_'))
                if hasattr(module, '__version__'):
                    current_version = module.__version__
                    # 使用版本範圍檢查
                    if not (min_version <= current_version < max_version):
                        version_mismatch.append(
                            f"{package} (當前: {current_version}, 需要: >={min_version}, <{max_version})"
                        )
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"缺少必要的套件: {', '.join(missing_packages)}")
            return False
        
        if version_mismatch:
            # 只顯示警告，不阻止測試執行
            logger.warning(f"版本不完全匹配: {', '.join(version_mismatch)}")
            logger.warning("版本不完全匹配可能會影響某些功能，但不會阻止測試執行")
        
        return True

    def run_tests(self, args: argparse.Namespace) -> bool:
        """執行測試"""
        try:
            start_time = time.time()
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024

            # 準備測試參數
            pytest_args = ['-v']
            
            if args.collect_only:
                pytest_args.append('--collect-only')
            
            if args.verbose:
                pytest_args.append('-vv')
                
            if not args.no_coverage:
                pytest_args.extend(['--cov=.', '--cov-report=html'])
                
            pytest_args.extend(['--tb=short', '--durations=10', 'tests/'])

            # 運行測試
            test_result = pytest.main(pytest_args)

            # 記錄性能指標
            end_time = time.time()
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024
            
            self.performance_metrics = {
                'execution_time': end_time - start_time,
                'memory_usage': memory_after - memory_before,
                'peak_memory': psutil.Process().memory_info().peak_wset / 1024 / 1024
            }

            return test_result == 0

        except Exception as e:
            logger.error(f"測試執行失敗: {str(e)}")
            return False

    def generate_report(self):
        """生成測試報告"""
        report_path = self.project_root / 'reports' / 'test_report.html'
        
        # 基本測試信息
        report_content = f"""
        <html>
        <head>
            <title>測試報告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ margin: 10px 0; padding: 10px; background-color: #f5f5f5; }}
                .success {{ color: green; }}
                .failure {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>測試執行報告</h1>
            <div class="metric">
                <h2>性能指標</h2>
                <p>執行時間: {self.performance_metrics['execution_time']:.2f} 秒</p>
                <p>記憶體使用: {self.performance_metrics['memory_usage']:.2f} MB</p>
                <p>峰值記憶體: {self.performance_metrics['peak_memory']:.2f} MB</p>
            </div>
        </body>
        </html>
        """
        
        report_path.parent.mkdir(exist_ok=True)
        report_path.write_text(report_content, encoding='utf-8')
        logger.info(f"測試報告已生成: {report_path}")

    def print_environment_info(self):
        """輸出環境信息"""
        logger.info("環境信息:")
        logger.info(f"Python 版本: {sys.version}")
        logger.info(f"作業系統: {os.name}")
        logger.info("\n已安裝的套件:")
        
        import pkg_resources
        installed_packages = [f"{dist.key} {dist.version}" 
                             for dist in pkg_resources.working_set]
        for package in sorted(installed_packages):
            logger.info(f"  {package}")

def main():
    # 解析命令行參數
    parser = argparse.ArgumentParser(description='運行測試套件')
    parser.add_argument('--collect-only', action='store_true', help='只收集測試用例')
    parser.add_argument('--verbose', '-v', action='store_true', help='顯示詳細輸出')
    parser.add_argument('--no-coverage', action='store_true', help='不生成覆蓋率報告')
    args = parser.parse_args()

    runner = TestRunner()
    
    # 1. 設置日誌
    runner.setup_logging()
    logger.info("開始執行測試...")
    
    # 2. 檢查依賴
    if not runner.check_dependencies():
        sys.exit(1)
    
    # 3. 運行測試
    success = runner.run_tests(args)
    
    # 4. 生成報告
    runner.generate_report()
    
    # 5. 輸出結果
    if success:
        logger.info("所有測試通過！")
    else:
        logger.error("測試失敗！")
        sys.exit(1)

if __name__ == "__main__":
    main() 