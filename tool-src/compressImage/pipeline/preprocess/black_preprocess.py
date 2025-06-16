import numpy as np
from typing import Dict

from defines import config
from pipeline.abstract_comps import PipelinePreprocess


class BlackPreProcess(PipelinePreprocess):
    """黑名单图预处理器"""
    
    def do(self, image: np.ndarray, last_result: Dict = None) -> Dict:
        """
        分析图像是否是黑名单目录
        
        Args:
            image: 要分析的图像(numpy数组)
            last_result: 上一个分析器的处理结果
            params: 额外参数
        
        Returns:
            包含处理结果的字典
        """
        # 如果是黑名单目录，则不处理
        file_path = last_result.get("file_path")
        if config.path_in_blacks(file_path):
            last_result["error"] = "黑名单目录"
        
        return last_result

