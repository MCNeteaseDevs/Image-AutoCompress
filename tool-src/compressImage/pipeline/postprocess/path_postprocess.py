from PIL import Image
from typing import Dict

from defines.compression_result import CompressionResult
from pipeline.abstract_comps import PipelinePreprocess
from utils.utils import process_file_rel_path


class PathPostProcess(PipelinePreprocess):
    """路径后处理器"""
    
    def do(self, image: Image.Image, last_result: Dict, params: Dict = None) -> Dict:
        """
        返回image_path相对res_path的路径，并使用斜杠

        Args:
            image: 要分析的图像(numpy数组)
            last_result: 上一个分析器的处理结果
            params: 额外参数
        
        Returns:
            包含处理结果的字典
        """
        result: CompressionResult = last_result["result"]
        file_path = result.original_path
        res_path = last_result["res_path"]
        last_result["file_rel_path"] = process_file_rel_path(file_path, res_path)
        return last_result

