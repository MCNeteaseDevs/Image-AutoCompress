import os
import numpy as np
from typing import Dict

from defines import config
from utils.ripgrep import JsonModifier
from pipeline.abstract_comps import PipelinePreprocess


class ParticlePreProcess(PipelinePreprocess):
    """粒子特效图预处理器"""
    
    def do(self, image: np.ndarray, last_result: Dict = None) -> Dict:
        """
        分析是否是粒子特效图

        判断贴图是否在粒子的json的引用中
        
        Args:
            image: 要分析的图像(numpy数组)
            last_result: 上一个分析器的处理结果
            params: 额外参数
        
        Returns:
            包含处理结果的字典
        """
        file_rel_path = last_result.get("file_rel_path")
        if not file_rel_path:
            return last_result
        
        # 判断是否是粒子特效图
        res_path = last_result.get("res_path")
        is_particle = self.image_is_particle(file_rel_path, res_path)
        last_result["is_particle"] = is_particle

        if is_particle:
            # 特效图分辨率阈值更高
            original_width = last_result.get("original_width", 0)
            original_height = last_result.get("original_height", 0)
            if original_width <= config.PARTICLE_MIN_RESOLUTION and original_height <= config.PARTICLE_MIN_RESOLUTION:
                last_result["error"] = f"特效图 分辨率低于{config.PARTICLE_MIN_RESOLUTION}，不压缩"
        return last_result

    def _is_sfx(self, image_path: str) -> bool:
        """
        判断贴图是否是序列帧特效（带json）
        """
        # 判断同级目录下，是否有json引用了该图片
        
        # 路径预处理
        image_path_text = os.path.basename(image_path)
        # 去掉格式后缀
        image_path_text = os.path.splitext(image_path_text)[0]
        res_path = os.path.dirname(image_path)
        
        # 搜索字符串
        modifier = JsonModifier.Instance()
        results = modifier.find_files_with_text(image_path_text, res_path, search_depth=1, whole_word=False)
        return True if len(results) > 0 else False

    def image_is_particle(self, image_path_text: str, res_path: str) -> bool:
        """
        判断贴图是否是特效的图（包括序列帧）

        Args:
            image_path_text: 贴图相对路径的字符串
            res_path: 资源包根目录
        """
        # 先判断是否是序列帧
        if self._is_sfx(image_path_text):
            return True
        
        # 查询res/particles、res/effects、res/models/effect下，是否有json引用了该图片
        
        # 拼接路径
        path_list = (
            # 微软版粒子
            os.path.join(res_path, "particles"),
            # 中国版粒子
            os.path.join(res_path, "effects"),
            # 中国版模型粒子
            os.path.join(res_path, "models", "effect"),
        )
        for _path in path_list:
            # 搜索字符串
            modifier = JsonModifier.Instance()
            results = modifier.find_files_with_text(image_path_text, _path, whole_word=True)
            if len(results) > 0:
                return True
        return False

