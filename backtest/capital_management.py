from dataclasses import dataclass
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class PositionConfig:
    """倉位配置類"""
    max_position_ratio: float  # 最大持倉比例 (0.1 ~ 1.0)
    min_position_ratio: float = 0.1  # 最小持倉比例
    position_step: float = 0.1  # 倉位調整步長
    
    def __post_init__(self):
        self._validate_config()
    
    def _validate_config(self):
        """驗證倉位配置"""
        if not 0.1 <= self.min_position_ratio <= self.max_position_ratio <= 1.0:
            raise ValueError("倉位比例設置無效，應在 10% ~ 100% 之間")
        if self.position_step <= 0 or self.position_step > 0.5:
            raise ValueError("倉位調整步長設置無效")

class CapitalManager:
    """資金管理器"""
    def __init__(self, 
                 initial_capital: float,
                 position_config: Optional[Dict[str, PositionConfig]] = None):
        """
        Args:
            initial_capital: 初始資金
            position_config: 市場倉位配置，格式為 {market: PositionConfig}
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # 確保至少有一個默認配置
        if position_config is None:
            position_config = {}
        if not position_config:  # 如果是空字典
            position_config['default'] = PositionConfig(max_position_ratio=0.1)
            
        self.position_config = position_config
        
    def calculate_position_size(self, 
                              price: float, 
                              market: str = 'default') -> float:
        """
        計算倉位大小
        
        Args:
            price: 當前價格
            market: 市場類型
            
        Returns:
            可開倉位大小
        """
        # 如果找不到指定市場的配置，使用該市場本身作為配置
        config = self.position_config.get(market)
        if config is None:
            # 如果市場配置不存在，使用默認配置
            config = self.position_config['default']
            logger.info(f"使用默認倉位配置於市場 {market}")
            
        max_position_value = self.current_capital * config.max_position_ratio
        position_size = max_position_value / price
        
        logger.info(f"市場: {market}, 當前資金: {self.current_capital:.2f}, "
                   f"最大倉位比例: {config.max_position_ratio:.2%}, "
                   f"計算倉位: {position_size:.2f}")
        
        return position_size
    
    def update_capital(self, new_capital: float):
        """更新當前資金"""
        self.current_capital = new_capital
        logger.info(f"更新資金: {self.current_capital:.2f}") 