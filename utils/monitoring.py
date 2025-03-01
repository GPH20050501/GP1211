import time
from functools import wraps
import logging
import psutil
import os

def performance_monitor(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        logging.info(f"{func.__name__} 執行時間: {execution_time:.2f}秒")
        return result
    return wrapper

def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logging.info(f"記憶體使用: {memory_info.rss / 1024 / 1024:.2f} MB") 

class SystemMonitor:
    def __init__(self):
        self.metrics_history = []
        self.alerts = []
    
    @performance_monitor
    def monitor_system_health(self):
        # 監控系統資源
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
        cpu_usage = psutil.Process().cpu_percent()
        
        # 監控數據質量
        data_quality = self.check_data_quality()
        
        # 監控策略表現
        strategy_metrics = self.check_strategy_performance()
        
        # 生成警報
        if memory_usage > 1000:  # 1GB
            self.alerts.append("記憶體使用過高警告")
        
        return {
            'memory_usage': memory_usage,
            'cpu_usage': cpu_usage,
            'data_quality': data_quality,
            'strategy_metrics': strategy_metrics,
            'alerts': self.alerts
        } 