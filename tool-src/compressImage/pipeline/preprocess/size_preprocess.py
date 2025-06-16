import numpy as np
from typing import Dict

from defines import config
from pipeline.abstract_comps import PipelinePreprocess


class SizePreProcess(PipelinePreprocess):
    """图片分辨率预处理器 - 低于指定分辨率，就不做处理"""
    
    def do(self, image: np.ndarray, last_result: Dict = None) -> Dict:
        """
        分析是否是icon图

        判断贴图是否在方块、物品的映射json中
        
        Args:
            image: 要分析的图像(numpy数组)
            last_result: 上一个分析器的处理结果
            params: 额外参数
        
        Returns:
            包含处理结果的字典
        """
        # 分辨率太低的图不处理
        original_width = last_result.get("original_width", 0)
        original_height = last_result.get("original_height", 0)
        if original_width <= config.MIN_RESOLUTION and original_height <= config.MIN_RESOLUTION:
            last_result["error"] = f"分辨率低于{config.MIN_RESOLUTION}，不压缩"
        return last_result

