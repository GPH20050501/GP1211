import pytest
import sys
from pathlib import Path

def main():
    """在 PyCharm 中運行測試"""
    # 設置專案根目錄
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # 運行測試
    pytest.main([
        '-v',                     # 詳細輸出
        '--tb=short',            # 簡短的錯誤追踪
        'tests',                 # 測試目錄
        '--cov=.',              # 覆蓋率報告
        '--cov-report=html'      # HTML 格式的覆蓋率報告
    ])

if __name__ == "__main__":
    main() 