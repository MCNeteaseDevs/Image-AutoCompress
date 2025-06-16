import json
import os
import sys
import logging

from defines import config
from utils.utils import get_app_root


# region 常量
# 设置ripgrep路径 - 处理多文件模式下的_internal目录结构
base_dir = get_app_root()
possible_paths = [
    # 1. 直接在程序目录下查找
    os.path.join(base_dir, "compressImageConfig", "settings.json"),
    # 2. 兼容在同级目录下找
    os.path.join(base_dir, "settings.json"),
    # 3. 在_internal目录下查找
    os.path.join(base_dir, "_internal", "config", "settings.json"),
]

# 查找第一个存在的路径
settings_path = "compressImageConfig/settings.json"
for path in possible_paths:
    if os.path.exists(path):
        settings_path = path
        break
# endregion


def load_config_from_json(json_path=None):
    """
    从JSON文件加载配置并更新config.py中的变量
    
    Args:
        json_path: JSON配置文件路径
    """
    try:
        if json_path is None:
            json_path = settings_path
        
        # 检查JSON文件是否存在
        if not os.path.exists(json_path):
            logging.info(f"配置文件不存在: {json_path}")
            return False
        
        # 读取JSON文件
        with open(json_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
        
        # 映射JSON中的值到config.py变量
        config_updated = False
        
        # 处理普通配置项
        for key in ["MIN_RESOLUTION", "PARTICLE_MIN_RESOLUTION", 
                    "ICON_MIN_RESOLUTION", "NORMAL_QUALITY", "TEXT_QUALITY"]:
            if key in settings and "value" in settings[key]:
                if hasattr(config, key):
                    setattr(config, key, settings[key]["value"])
                    logging.info(f"已更新配置: {key} = {settings[key]['value']}")
                    config_updated = True
        
        # 处理BLACK_PATHS特殊情况（因为它是元组）
        if "BLACK_PATHS" in settings and "value" in settings["BLACK_PATHS"]:
            if hasattr(config, "BLACK_PATHS"):
                # 确保转换为元组类型，因为在config.py中定义为元组
                setattr(config, "BLACK_PATHS", tuple(settings["BLACK_PATHS"]["value"]))
                logging.info(f"已更新配置: BLACK_PATHS (共 {len(settings['BLACK_PATHS']['value'])} 项)")
                config_updated = True
        
        if config_updated:
            logging.info("配置已从JSON文件成功加载")
            return True
        else:
            logging.warning("没有找到可更新的配置项")
            return False
            
    except json.JSONDecodeError as e:
        logging.error(f"JSON解析错误: {str(e)}")
        return False
    except Exception as e:
        logging.error(f"加载配置时出错: {str(e)}")
        return False

if __name__ == "__main__":
    # 如果直接运行该脚本，则尝试加载配置
    json_path = "config/settings.json"
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    
    success = load_config_from_json(json_path)
    print(f"配置加载{'成功' if success else '失败'}")