import os
import sys
from PIL import Image
import cv2

from defines.image_type import ImageType


"""
工具代码
"""


def process_file_rel_path(file_path, res_path):
    """
    返回file_path相对res_path的路径，并使用斜杠
    
    Args:
        file_path (str): 图像文件的路径
        res_path (str): 资源包根路径
        
    Returns:
        str: 处理后的路径，如果不是完整路径则返回原路径
    """
    rel_path = None

    try:
        # 检查是否是完整路径
        if not os.path.isabs(file_path):
            rel_path = file_path
        if not rel_path:
            # 获取相对路径
            rel_path = os.path.relpath(file_path, res_path)
    except ValueError:  # 处理路径不在同一个驱动器的情况
        rel_path = file_path
    
    # 将反斜杠转换为斜杠
    rel_path = rel_path.replace("\\", "/")
    # 去掉格式后缀
    rel_path_no_ext = os.path.splitext(rel_path)[0]
    return rel_path_no_ext


# 获取插值算法
def get_image_resample(image_type: str, is_pil=True) -> str:
    """
    根据图片类型，返回插值算法
    """
    if is_pil:
        resample = Image.Resampling.NEAREST if image_type == ImageType.PixelArt else Image.Resampling.BICUBIC
    else:
        resample = cv2.INTER_NEAREST if image_type == ImageType.PixelArt else cv2.INTER_AREA
    return resample

# 获取插值算法名字字符串
def get_resample_name(resample: str) -> str:
    """
    获取插值算法名字的字符串
    """
    if resample in (Image.Resampling.NEAREST, cv2.INTER_NEAREST):
        return "NEAREST"
    else:
        return "CUBIC"


def get_app_root():
    """获取应用程序的根目录"""
    if getattr(sys, 'frozen', False):
        # 如果是打包后的可执行文件 - 使用可执行文件所在目录
        # 在多文件模式下，这将正确指向分发目录
        return os.path.dirname(sys.executable)
    else:
        # 如果是开发环境
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

