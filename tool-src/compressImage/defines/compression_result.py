from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from PIL import Image
from defines.image_type import ImageType


@dataclass
class CompressionResult:
    """
    压缩结果
    """
    # 原图
    original_path: str
    original_size: int = 0
    original_width: int = 0
    original_height: int = 0
    # 压缩图
    compressed_path: str = None
    compressed_size: int = 0
    compressed_width: int = 0
    compressed_height: int = 0
    # 评估结果
    ssim_score: float = 0.0
    psnr_score: float = 0.0
    perceptual_score: float = 0.0  # 感知质量分数
    feature_preservation: float = 0.0  # 特征保留率
    quality_metrics: Dict[str, float] = None  # 完整质量指标
    # 图片其他数据
    image_type: str = ImageType.PixelArt  # 图片类型
    image_data: Image.Image = None  # 图像数据字段，用于保存图片
    is_text_image: bool = False  # 是否为带文本图
    is_sfx_image: bool = False  # 是否为序列帧图
    # 压缩数据
    processing_time: float = 0.0  # 处理时长
    is_compressed: bool = True # 是否已压缩
    info: str = "" # 文字信息，如失败log
    update_json_record: List[str] = None # 修改文件的记录，用于保存修改信息


    def to_dict(self):
        """
        JSON序列化时排除图像数据
        """
        result = {k: v for k, v in self.__dict__.items() if k != "image_data"}
        return result
    
    def get_quality_score(self) -> float:
        """
        获取综合质量评分，使用感知分数
        """
        return self.perceptual_score

