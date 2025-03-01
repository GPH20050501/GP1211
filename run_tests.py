import pytest
import logging

def run_validation():
    """執行分層驗證"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # 運行所有測試
        pytest.main(['-v', 'tests/'])
        
    except Exception as e:
        logger.error(f"驗證過程中發生錯誤: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    success = run_validation()
    print(f"驗證{'成功' if success else '失敗'}") 