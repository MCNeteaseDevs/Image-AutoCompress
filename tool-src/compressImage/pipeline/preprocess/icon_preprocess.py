import os
import numpy as np
from typing import Dict

from defines import config
from utils.ripgrep import JsonModifier
from pipeline.abstract_comps import PipelinePreprocess


class IconPreProcess(PipelinePreprocess):
    """方块、物品的icon图预处理器 - 分析是否是icon图"""
    
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
        file_rel_path = last_result.get("file_rel_path")
        if not file_rel_path:
            return last_result
        
        res_path = last_result.get("res_path")
        is_icon = self.image_is_icon(file_rel_path, res_path)
        last_result["is_icon"] = is_icon
        
        if is_icon:
            # 分辨率太低的图不处理
            original_width = last_result.get("original_width", 0)
            original_height = last_result.get("original_height", 0)
            if original_width <= config.ICON_MIN_RESOLUTION and original_height <= config.ICON_MIN_RESOLUTION:
                last_result["error"] = f"icon图 分辨率低于{config.ICON_MIN_RESOLUTION}，不压缩"
        return last_result

    def image_is_icon(self, image_path_text: str, res_path: str) -> bool:
        """
        判断贴图是否是方块icon、物品icon
        """
        # 资源路径限制
        path_list = (
            os.path.join(res_path, "textures", "item_texture.json"),
            os.path.join(res_path, "textures", "terrain_texture.json"),
            os.path.join(res_path, "textures", "flipbook_textures_items.json"),
            os.path.join(res_path, "textures", "flipbook_textures.json"),
        )

        is_icon = False

        # 搜索字符串
        modifier = JsonModifier.Instance()
        for _path in path_list:
            results = modifier.find_files_with_text(image_path_text, _path, search_depth=1, whole_word=True)
            if len(results) > 0:
                is_icon = True
                break
        # 不做模型纹理判断，因为在此之前就已经做了拦截了
        return is_icon

