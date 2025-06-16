import os
import numpy as np
from typing import Dict

from utils.ripgrep import JsonModifier
from pipeline.abstract_comps import PipelinePreprocess


class UISFXPreProcess(PipelinePreprocess):
    """UI序列帧图预处理器"""
    
    def __init__(self):
        super().__init__("UISFXAnalyzer")
    
    def do(self, image: np.ndarray, last_result: Dict = None) -> Dict:
        """
        分析是否是UI序列帧图

        判断贴图是否在ui的json的引用中、且应用了uv动画
        
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
        is_ui_sfx = self.image_is_ui_sfx(file_rel_path, res_path)
        last_result["is_ui_sfx"] = is_ui_sfx
        return last_result
    
    def image_is_ui_sfx(self, image_path_text: str, res_path: str) -> bool:
        """
        判断贴图是否是UI序列帧图片

        Args:
            image_path: 贴图路径(相对资源包根目录、或者完整路径)
            res_path: 资源包根目录
        """
        # TODO: 优化项，这里只需要在找到后，就可停止后续的逻辑，而不是遍历完
        # 需处理继承控件

        # 判断依据：在res_path/ui下的所有json文件中，如果有使用该贴图、且指定了uv_size
        # uv、uv_size都有可能是继承来的属性，需要查对应的uv，才能得知是否是序列帧、且得知序列帧的帧数是多少
        # uv_size=单帧大小，uv=动画配置，frame_count=序列帧的帧数
        # 需要根据帧数，才能准确算出单帧的大小

        # 资源路径需限制在ui下
        if not res_path.endswith("ui"):
            res_path = os.path.join(res_path, "ui")

        is_sfx = False
        
        # 检查是否使用uv动画
        def _check(json_obj, path, value, file_path):
            if isinstance(value, dict):
                if value.get("texture") == image_path_text:
                    # 如果uv的值是@***，则表示是使用动画，则可以认为是序列帧
                    uvVal = value.get("uv")
                    if uvVal and isinstance(uvVal, str) and uvVal.startswith("@"):
                        nonlocal is_sfx
                        is_sfx = True
            return value  # 默认不修改
        
        # 搜索字符串
        modifier = JsonModifier.Instance()
        modifier.batch_modify_json_files(image_path_text, _check, res_path, whole_word=True)
        return is_sfx

