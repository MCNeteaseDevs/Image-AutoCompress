import os
from typing import Dict
from PIL import Image
from typing import Dict

from defines.compression_result import CompressionResult
from pipeline.abstract_comps import PipelinePostProcess


class SavePostProcess(PipelinePostProcess):
    """存储图片 后处理器"""
    
    def do(self, image: Image.Image, last_result: Dict, params: Dict = None) -> Dict:
        """
        存储图片到本地

        Args:
            image: PIL图像对象
            last_result: 上一个的处理结果
            params: 额外参数
            
        Returns:
            综合处理结果的字典
        """
        result: CompressionResult = last_result["result"]
        # 处理成功
        if result.is_compressed:
            # 保存文件
            if result.image_data is not None:
                output_dir = result.compressed_path
                os.makedirs(os.path.dirname(output_dir), exist_ok=True)
                
                # 确定保存参数
                output_path = result.compressed_path
                _, ext = os.path.splitext(output_path)
                # 写入文件
                if ext.lower() == ".png":
                    result.image_data.save(output_path, format="PNG", optimize=True, compression_level=9)
                elif ext.lower() == ".jpg" or ext.lower() == ".jpeg":
                    result.image_data.save(output_path, format="JPEG", quality=95, optimize=True)
                else:
                    result.image_data.save(output_path)
                
                last_result["save_to_disk"] = True
        return last_result

