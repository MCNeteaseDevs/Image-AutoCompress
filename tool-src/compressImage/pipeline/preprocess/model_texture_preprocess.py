import os
import numpy as np
from typing import Dict

from utils.ripgrep import JsonModifier
from pipeline.abstract_comps import PipelinePreprocess


class ModelTexturePreProcess(PipelinePreprocess):
    """模型贴图预处理器 - 检测是否是模型贴图"""
    
    def do(self, image: np.ndarray, last_result: Dict = None) -> Dict:
        """
        分析是否是模型贴图
        
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
        
        # 如果是模型图，则跳过
        res_path = last_result.get("res_path")
        if self.image_is_model_texture(file_rel_path, res_path):
            # 跳过后续处理
            last_result["error"] = "模型纹理，不压缩"
        return last_result

    def _is_netease_model(self, terrain_key: str, res_path: str) -> bool:
        """
        判断贴图的key是否是netease_block里的模型贴图的引用
        """
        # 处理terrain_texture.json目录
        model_path = os.path.join(res_path, "models", "netease_block")
        modifier = JsonModifier.Instance()
        
        is_model_texture = False

        def _check(json_obj, path, value, file_path):
            if isinstance(value, dict):
                textures = value.get("textures", [])
                if textures and terrain_key in textures:
                    nonlocal is_model_texture
                    is_model_texture = True
            return value
        
        modifier.batch_modify_json_files(terrain_key, _check, model_path, whole_word=True, context_range=1)
        return is_model_texture

    def image_is_model_texture(self, image_path_text: str, res_path: str, terrain_key: str = None) -> bool:
        """
        判断贴图是否是模型贴图
        """
        # 资源路径限制
        path_list = (
            os.path.join(res_path, "attachables"),
            os.path.join(res_path, "entity"),
        )
        
        # 搜索字符串
        modifier = JsonModifier.Instance()
        for _path in path_list:
            results = modifier.find_files_with_text(image_path_text, _path, whole_word=True)
            if len(results) > 0:
                return True
        
        # 再判断netease_block：terrain_texture.json目录
        if terrain_key:
            return self._is_netease_model(terrain_key, res_path)
        else:
            terrain_path = os.path.join(res_path, "textures", "terrain_texture.json")
            modifier = JsonModifier.Instance()
            
            is_model_texture = False

            def _check(json_obj, path, value, file_path):
                if isinstance(value, dict):
                    if value.get("textures") == image_path_text:
                        key = path[-1]
                        nonlocal is_model_texture
                        # 查询models/netease_block中，是否有引用该贴图
                        is_model_texture = self._is_netease_model(key, res_path)
                return value
            
            modifier.batch_modify_json_files(image_path_text, _check, terrain_path, whole_word=True, context_range=1)
            return is_model_texture

