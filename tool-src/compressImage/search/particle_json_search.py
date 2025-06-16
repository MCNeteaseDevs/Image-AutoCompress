import os
from typing import Dict
from utils.ripgrep import JsonModifier
from utils.utils import process_file_rel_path


"""
粒子特效/序列帧 json文件的搜索与修改处理
- 搜索图片所在的同级目录
- 搜索res/particles
- 搜索res/effects
- 搜索res/models/effect
"""


# region 封装方法
def _is_sfx(image_path: str) -> bool:
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

def image_is_particle(image_path: str, res_path: str) -> bool:
    """
    判断贴图是否是特效的图（包括序列帧）

    Args:
        image_path: 贴图完整路径
        res_path: 资源包根目录
    """
    # 先判断是否是序列帧
    if _is_sfx(image_path):
        return True
    
    # 查询res/particles、res/effects、res/models/effect下，是否有json引用了该图片
    
    # 路径预处理
    image_path_text = process_file_rel_path(image_path, res_path)

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

def update_particle_sfx_json(image_path: str, uv_size: tuple) -> Dict[str, bool]:
    """
    更新特效序列帧的json配置

    Args:
        image_path: 贴图完整路径
        uv_size: 贴图总分辨率
    """
    # 路径预处理
    image_path_text = os.path.basename(image_path)
    res_path = os.path.dirname(image_path)
    
    # TODO: 该版本存在问题：如果重复执行逻辑，这些数值会越来越小（因为是拿json本身的数据来重复计算，而不是根据图的数据计算，会存在重复计算的问题）
    # 修改内容
    def _update_context(json_obj, path, value, file_path):
        if isinstance(value, dict):
            # 需判断是否是整个json的数据
            if "frames" in value and "meta" in value:
                # 贴图原本大小
                size = (value["meat"]["size"]["w"], value["meat"]["size"]["h"])
                
                # 修改贴图大小
                value["meat"]["size"]["h"] = uv_size[1]
                value["meat"]["size"]["w"] = uv_size[0]

                if len(value["frames"]) > 0:
                    # 计算压缩率
                    ratio = round(size[0] / uv_size[0], 1)
                    # 修改帧数据的贴图大小
                    for frame in value["frames"]:
                        frame["frame"]["h"] *= ratio
                        frame["frame"]["w"] *= ratio
                        frame["frame"]["x"] *= ratio
                        frame["frame"]["y"] *= ratio
                        frame["sourceSize"]["h"] *= ratio
                        frame["sourceSize"]["w"] *= ratio
                        frame["spriteSourceSize"]["h"] *= ratio
                        frame["spriteSourceSize"]["w"] *= ratio
                pass
        return value  # 默认不修改
    
    # 搜索字符串
    modifier = JsonModifier.Instance()
    results = modifier.batch_modify_json_files(image_path_text, _update_context, res_path, search_depth=1, whole_word=True, context_range=2)
    return results

# endregion

