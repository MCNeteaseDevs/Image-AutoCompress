import numpy as np
from typing import Dict

from pipeline.abstract_comps import PipelinePreprocess
from utils.utils import process_file_rel_path


class PathPreProcess(PipelinePreprocess):
    """路径预处理器 - 预处理路径"""
    
    def do(self, image: np.ndarray, last_result: Dict = None) -> Dict:
        """
        返回image_path相对res_path的路径，并使用斜杠

        Args:
            image: 要分析的图像(numpy数组)
            last_result: 上一个分析器的处理结果
            params: 额外参数
        
        Returns:
            包含处理结果的字典
        """
        file_path = last_result.get("file_path")
        res_path = last_result.get("res_path")
        last_result["file_rel_path"] = process_file_rel_path(file_path, res_path)
        return last_result

