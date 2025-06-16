import logging


DEBUG = False
"""开启/关闭debug"""
if DEBUG:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.WARNING)


MIN_RESOLUTION = 32
"""低于该分辨率的贴图，不进行压缩"""

PARTICLE_MIN_RESOLUTION = 128
"""低于该分辨率的特效图，不进行压缩"""

ICON_MIN_RESOLUTION = 16
"""低于该分辨率的icon图，不进行压缩"""

NORMAL_QUALITY = 0.9
"""评估质量阈值-普通贴图"""
TEXT_QUALITY = 0.95
"""评估质量阈值-文本贴图"""

BLACK_PATHS = (
    "textures\\environment",                # 环境贴图
    "textures\\colormap",                   # 颜色映射贴图
    "textures\\gui",                        # 界面贴图
    "textures\\map",                        # 地图贴图
    "textures\\misc",                       # 杂项贴图
    "textures\\painting",                   # 画贴图
    "textures\\particle\\particles.png",    # 粒子贴图
    "textures\\models",                     # 模型贴图（有可能是通过代码引用，无法区分，故统一不处理）
    "textures\\entity",                     # 实体模型贴图（有可能是通过代码引用，无法区分，故统一不处理）
)
"""黑名单目录(相对路径)"""

def path_in_blacks(path: str):
    """
    判断路径是否在黑名单中
    """
    # 判断黑名单目录，是否是path的相对路径
    for black in BLACK_PATHS:
        if black in path:
            return True
    return False

if __name__ == "__main__":
    patht = r"D:\yeh\dev\mcmod_dev\dianli_industrial_bak\resource_pack_x4x17pf9\textures\environment\sky_clouds.png"
    print(path_in_blacks(patht))
