from abc import ABC, abstractmethod
from typing import Dict, Tuple
from PIL import Image
import numpy as np


"""
组件抽象类
"""


class PipelineComponent(ABC):
    """管道组件基类"""
    def __init__(self, name: str = None, config: Dict = None):
        self.name = name
        if self.name is None:
            self.name = self.__class__.__name__
        self.config = config or {}
        
    def get_name(self) -> str:
        return self.name


class PipelinePreprocess(PipelineComponent):
    """管道预处理器 -负责预处理，即处理图片之前的准备工作"""
    
    @abstractmethod
    def do(self, image: np.ndarray, last_result: Dict, params: Dict = None) -> Dict:
        """
        执行预处理
        
        Args:
            image: 要分析的图像(numpy数组)
            last_result: 上一个预处理的处理结果
            params: 额外参数
        
        Returns:
            综合处理结果的字典
        """
        pass


class PipelineProcess(PipelineComponent):
    """管道处理器 -负责对图片的处理"""
    
    @abstractmethod
    def do(self, image: Image.Image, params: Dict = None) -> Tuple[Image.Image, Dict]:
        """
        执行处理逻辑
        
        Args:
            image: 要处理的PIL图像对象
            result: 处理结果
            params: 额外参数
            
        Returns:
            处理后的图像和处理元数据
        """
        pass
        
    def is_applicable(self, preprocess_results: Dict) -> bool:
        """
        判断处理器是否适用于当前图像
        
        Args:
            preprocess_results: 图像分析结果(来源于预处理)
            
        Returns:
            是否应用此处理器
        """
        return True


class PipelinePostProcess(PipelineComponent):
    """管道后处理器 -负责执行图片处理完成后的工作"""
    
    @abstractmethod
    def do(self, image: Image.Image, last_result: Dict, params: Dict = None) -> Dict:
        """
        评估处理前后图像的质量变化
        
        Args:
            image: PIL图像对象
            last_result: 上一个的处理结果
            params: 额外参数
            
        Returns:
            综合处理结果的字典
        """
        pass


class PipelineEndProcess(PipelineComponent):
    """管道最终处理器 -负责执行**所有**图片处理完成后的工作"""
    
    @abstractmethod
    def do(self, last_results: Dict, params: Dict = None) -> Dict:
        """
        评估处理前后图像的质量变化
        
        Args:
            last_results: 上一个的处理结果
            params: 额外参数
            
        Returns:
            综合处理结果的字典
        """
        pass

