# -*- mode: python ; coding: utf-8 -*-


# region 项目配置

# 打包文件名，该名字必须固定，在mcs中需要用到该名字，所以不能随意修改
exe_name = "compressImage"

# icon路径
icon = "../assets/icon.ico"

# 定义共享和工具特有的资源
shared_tools_files = [
    ('../libs/ripgrep/rg.exe', 'libs/ripgrep'),
    # 以下许可证文件出于法律合规性考虑建议包含
    ('../libs/ripgrep/LICENSE-MIT', 'libs/ripgrep/licenses'),
    ('../libs/ripgrep/UNLICENSE', 'libs/ripgrep/licenses'),
    ('../libs/ripgrep/COPYING', 'libs/ripgrep/licenses'),
]
tool_specific_files = [
    ('../compressImage/compressImageConfig/settings.json', 'compressImageConfig'),
    # 如果有工具特有的数据文件，可以添加在这里
]

datas = shared_tools_files + tool_specific_files

# 运行时钩子
# runtime_hooks = ['hook.py']
runtime_hooks = []

# 入口py文件
py_files = [
    "../compressImage/main.py"
]

# 需显式排除的库，pyinstall不能很好的处理
excludes = [
    'matplotlib', 'PyQt5', 'PySide2', 'tk', 'tcl', 'pandas', 'contourpy',
    'notebooks', 'IPython', 'ipykernel', 'sphinx', 'flake8', 'pycodestyle',
    'pyflakes', 'alabaster', 'babel', 'docutils', 'jinja2', 'markupsafe', 
    'pypandoc', 'torch', 'tensorboard', 'tensorflow', 'huggingface-hub', 
    'tiktoken', 'openai', 'aider-chat', 'GitPython', 'litellm', 'rich',
    'altgraph', 'aiohttp', 'httpcore', 'httpx', 'beautifulsoup4', 'soupsieve',
    'pydub', 'sounddevice', 'soundfile', 'gitdb', 'smmap', 'pyinstaller',
    'watchfiles', 'posthog', 'mixpanel', 'pywt', '_tkinter', 'tkinter', 'Tkinter',
    'yaml', 
]

# endregion


# region 打包配置

# 变量 block_cipher 用于存储加密的密码，默认为 None
block_cipher = None


# ========= 收集你的脚本需要的所有模块和文件
# 变量 a 是一个 Analysis 对象
# 把要打包的脚本传给他，初始化的过程中，他会分析依赖情况
# 最后会生成 a.pure  a.scripts  a.binaries  a.datas 这样4个关键列表，以及 a.zipped_data (不重要)
# 其中：
#     a.pure 是依赖的纯 py 文件，
#     a.scripts 是要依次执行的脚本文件，
#     a.binaries 是依赖的二进制文件，
#     a.datas 是要复制的普通文件
a = Analysis(
    py_files,     # 指定要打包的 Python 脚本的入口
    pathex=[],          # 用来指定模块搜索路径
    binaries=[],        # 包含了动态链接库或共享对象文件，会在运行之后自动更新，加入依赖的二进制文件
    datas=datas,   # 列表，用于指定需要包含的额外文件。每个元素都是一个元组：（文件的源路径, 在打包文件中的路径)
    hiddenimports=[],   # 用于指定一些 PyInstaller 无法自动检测到的模块
    hookspath=[],       # 指定查找 PyInstaller 钩子的路径
    hooksconfig={},     # 自定义 hook 配置
    runtime_hooks=runtime_hooks,        # 指定运行时 hook，本质是一个 Python 脚本，hook 会在你的脚本运行前运行，可用于准备环境
    # 显式排除的库(可打包后看有多什么库，就单独针对该库排除)
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
# 除此之外，a 还有一些没有列出的属性：
#   pure 是一个列表，包含了所有纯 Python 模块的信息，这些模块会被打包到一个 .pyz 文件中。
#   scripts 是一个列表，包含了你的 Python 脚本的信息。这些脚本会被打包到一个 exe 文件中。

# 优化：删除重复文件和PDB文件
a.binaries = sorted(set(a.binaries))
a.binaries = [x for x in a.binaries if not x[0].endswith('.pdb')]
a.datas = sorted(set(a.datas))

# 将复制的资源文件，从_internal中挪出来
for data in a.datas:
    destfile = data[1]
    for d in datas:
        if d[1] in destfile:
            destfile


# # 给依赖的binaryes挪到python_runtime目录下
# import re
# import os
# # 用一个函数选择性对依赖文件目标路径改名
# def new_dest(package: str):
#     if package == 'base_library.zip' or re.match(r'python\d+.dll', package):
#         return package
#     return 'python_runtime' + os.sep + package
# a.binaries = [(new_dest(x[0]), x[1], x[2]) for x in a.binaries]


# ========= 创建 pyz 文件，它在运行时会被解压缩到临时目录中，然后被加载和执行。它会被打包进 exe 文件
# 变量 pyz 是一个 PYZ 对象，默认给他传入 a.pure 和 a.zipped_data 
# 初始化过程中，它会把 a.pure  a.zipped_data 打包成一个 pyz 文件
# 如果有密码，还会加密打包
pyz = PYZ(
    a.pure, 
    a.zipped_data,
    cipher=block_cipher,
    compress=True
)


# ========== 生成exe文件
# 变量 exe 是一个 EXE 对象，
# 给它传入打包好的 pyz 文件、a.scripts、程序名、图标、是否显示Console、是否debug
# 最后他会打包生成一个 exe 文件
exe = EXE(
    pyz,            # 包含了所有纯 Python 模块
    a.scripts,      # 包含了主脚本及其依赖
    [],             # 所有需要打包到 exe 文件内的二进制文件
    exclude_binaries=True,          # 若为 True，所有的二进制文件将被排除在 exe 之外，转而被 COLLECT 函数收集
    name=exe_name,     # exe 文件名
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,    # 是否移除所有的符号信息，使打包出的 exe 文件更小（Windows没有这个工具）
    upx=True,       # 是否用 upx 压缩 exe 文件
    runtime_tmpdir=None,
    console=True,   # 使用控制台模式便于查看日志
    icon=icon,      # exe 图标
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)


# ========= 收集文件，创建分发目录
# 变量 coll 是一个 COLLECT 对象，
# 给它传入：
# 	exe  
# 	a.binaries  二进制文件
# 	a.dattas	普通文件
# 	name        输出文件夹名字
# 在实例化的过程中，会把传入的这些项目，都复制到 name 文件夹中
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_args=['--best', '--lzma'],
    name=exe_name,     # 生成的文件夹名字
)
# endregion


# region 将复制的资源重新更换位置
# 查找_internal下的目录，如果发现datas中的文件，则往上挪一个目录级别
import os
import shutil
from pathlib import Path

def relocate_data_files(dist_dir, datas_list):
    """
    将_internal目录下的特定文件移动到上级目录
    
    Args:
        dist_dir: 打包输出目录路径
        datas_list: 数据文件列表，格式为[(source, dest), ...]
    """
    dist_path = Path(dist_dir)
    internal_path = dist_path / '_internal'
    
    if not internal_path.exists():
        print("未找到_internal目录，跳过文件重定位")
        return
    
    # 提取datas中的目标路径
    target_paths = [dest for _, dest in datas_list]
    
    for target_path in target_paths:
        # 在_internal中查找对应路径
        internal_target = internal_path / target_path
        
        if internal_target.exists():
            # 目标位置（上移一级）
            new_location = dist_path / target_path
            
            print(f"移动文件: {internal_target} -> {new_location}")
            
            # 确保目标目录存在
            new_location.parent.mkdir(parents=True, exist_ok=True)
            
            # 移动文件或目录
            if internal_target.is_file():
                shutil.move(str(internal_target), str(new_location))
            else:
                # 如果是目录，需要处理可能的目标目录已存在情况
                if new_location.exists():
                    shutil.rmtree(str(new_location))
                shutil.move(str(internal_target), str(new_location))
            
            # 清理空的父目录
            try:
                parent_dir = internal_target.parent
                while parent_dir != internal_path and not any(parent_dir.iterdir()):
                    parent_dir.rmdir()
                    parent_dir = parent_dir.parent
            except OSError:
                pass  # 目录不为空或其他错误，忽略

# 在COLLECT完成后执行文件重定位
def post_build_relocate():
    """构建后处理函数"""
    # 获取构建输出目录
    if hasattr(coll, 'name'):
        dist_dir = os.path.join('dist', coll.name)
    else:
        dist_dir = 'dist'
    
    if os.path.exists(dist_dir):
        relocate_data_files(dist_dir, datas)
        print("文件重定位完成")
    else:
        print(f"构建目录不存在: {dist_dir}")

post_build_relocate()
# endregion