import functools
import time
import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)

def performance_monitor(func: Callable) -> Callable:
    """
    監控函數執行時間和性能的裝飾器
    
    Args:
        func: 要監控的函數
        
    Returns:
        wrapped_func: 包裝後的函數
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            logger.info(f"函數 {func.__name__} 執行時間: {execution_time:.4f} 秒")
            
            return result
            
        except Exception as e:
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            logger.error(f"函數 {func.__name__} 執行失敗，耗時 {execution_time:.4f} 秒")
            logger.error(f"錯誤信息: {str(e)}")
            raise
            
    return wrapper 