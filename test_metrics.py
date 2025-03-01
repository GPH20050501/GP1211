from utils.performance_metrics import calculate_advanced_metrics, Trade
import pandas as pd
import numpy as np
import logging

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_metrics():
    logging.info("開始測試績效指標計算")
    
    # 創建測試數據
    equity_curve = pd.Series(
        [1000000, 1100000, 1050000],
        index=pd.date_range(start='2024-01-01', periods=3)
    )
    logging.info(f"創建資金曲線: {equity_curve.values}")
    
    trades = [
        Trade(
            entry_time=pd.Timestamp('2024-01-01'),
            exit_time=pd.Timestamp('2024-01-02'),
            entry_price=100,
            exit_price=110,
            position_size=1000,
            profit_loss=10000,
            commission=10,
            slippage=5
        )
    ]
    logging.info(f"創建交易記錄: {len(trades)} 筆")
    
    # 計算績效指標
    try:
        logging.info("開始計算績效指標")
        result = calculate_advanced_metrics(equity_curve, trades)
        logging.info("績效指標計算完成")
        
        print("\n=== 績效指標摘要 ===")
        for category, metrics in result.items():
            print(f"\n{category}:")
            for name, value in metrics.items():
                print(f"  {name}: {value}")
                
        logging.info("測試成功完成")
    except Exception as e:
        logging.error(f"測試失敗: {str(e)}")
        raise

if __name__ == "__main__":
    test_metrics() 