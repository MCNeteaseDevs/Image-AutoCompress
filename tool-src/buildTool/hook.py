# hook.py
import os
import sys


"""
期望的目录结构
- apps/
    - _internal/
        - python依赖+第三方库
    - 工具.exe
    - 工具config/
    - libs/
        - 第三方exe库
"""

PYTHON_RUNTIME = "_internal"
LIBS = "libs"



# 获取应用根目录和工具目录
def get_app_paths():
    # 获取可执行文件路径
    exe_path = sys.executable
    exe_dir = os.path.dirname(exe_path)
    
    tool_dir = exe_dir
    # 上一级目录
    last_path = os.path.dirname(exe_dir)

    # 确定上级目录下是否有python环境
    if os.path.exists(os.path.join(last_path, PYTHON_RUNTIME)):
        python_runtime = os.path.join(last_path, PYTHON_RUNTIME)
    else:
        """
        原始结构: 
        - 工具.exe
        - _internal/
            - python虚拟机+第三方库
            - libs/
                - 各种工具库
            - config/
        """
        internal_dir = os.path.join(tool_dir, "_internal")
        if os.path.exists(internal_dir):
            app_root = tool_dir
            python_runtime = internal_dir
        else:
            # 可能是开发环境
            app_root = last_path
            python_runtime = None

    # 上级是否有libs目录
    if os.path.exists(os.path.join(last_path, LIBS)):
        libs_dir = os.path.join(last_path, LIBS)
    else:
        internal_dir = os.path.join(tool_dir, "_internal")
        if os.path.exists(internal_dir):
            libs_dir = os.path.join(internal_dir, LIBS)
        else:
            # 可能是开发环境
            libs_dir = os.path.join(app_root, LIBS)
    
    tool_name = os.path.basename(tool_dir)
    if tool_name.endswith(".exe"):
        tool_name = tool_name[:-4]
    
    return {
        "app_root": app_root,
        "tool_dir": tool_dir,
        "tool_name": tool_name,
        "python_runtime": python_runtime,
        "libs_dir": libs_dir,
    }

# 获取路径信息
paths = get_app_paths()

# 设置Python路径
if os.path.exists(paths["python_runtime"]):
    sys.path.insert(0, paths["python_runtime"])
    # 添加site-packages目录(如果存在)
    site_packages = os.path.join(paths["python_runtime"], "site-packages")
    if os.path.exists(site_packages):
        sys.path.insert(0, site_packages)

# 设置工具路径（具体的exe文件，在该路径下查找）
if os.path.exists(paths["libs_dir"]):
    os.environ["LIBS_PATH"] = paths["libs_dir"]

# 打印调试信息
print(f"应用根目录: {paths['app_root']}")
print(f"工具目录: {paths['tool_dir']}")
print(f"工具名称: {paths['tool_name']}")
print(f"Python运行时: {paths['python_runtime']}")
print(f"工具库目录: {paths['libs_dir']}")
