import json
import re
import shutil
import subprocess
import os
import sys
from typing import Dict, List, Any, Tuple, Optional, Union
import copy
import logging

from utils.utils import get_app_root


# region 常量
# from libutils.libs_path_manager import LibsManager
# rgPath = LibsManager.get_instance().get_ripgrep_path()

base_dir = get_app_root()
possible_paths = [
    # 1. 直接在程序目录下查找
    os.path.join(base_dir, "libs", "ripgrep", "rg.exe"),
    # 2. 在_internal目录下查找
    os.path.join(base_dir, "_internal", "libs", "ripgrep", "rg.exe"),
]

# 添加PyInstaller打包环境中的路径
if getattr(sys, 'frozen', False):
    # PyInstaller创建临时文件夹后将所有文件解压到其中
    # 对于单文件模式，使用sys._MEIPASS获取临时文件夹路径
    meipass_dir = getattr(sys, '_MEIPASS', None)
    if meipass_dir:
        possible_paths.append(os.path.join(meipass_dir, "libs", "ripgrep", "rg.exe"))

# 最后尝试系统PATH中的rg
possible_paths.append("rg.exe")

# 查找第一个存在的路径
rgPath = None
for path in possible_paths:
    if os.path.exists(path) or (path == "rg.exe" and shutil.which("rg.exe")):
        rgPath = path
        break

# 如果没有找到，使用默认值并记录警告
if not rgPath:
    # 开发模式
    rgPath = os.path.join(base_dir, "..", "libs", "ripgrep", "rg.exe")
# endregion

# region JsonModifier
class JsonModifier:
    """
    搜索JSON文件中的特定内容并修改其上下文
    """

    _singleton = None
    _findFilesCache = {}

    def __init__(self, rg_path: str = "rg"):
        """初始化JSON修改器
        
        Args:
            rg_path: ripgrep可执行文件的绝对路径
        """
        self.rg_path = rg_path
        # 验证rg命令是否可用
        try:
            subprocess.run([self.rg_path, "--version"], 
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE, 
                           check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            raise RuntimeError("ripgrep不可用，请确保它存储在./libs/ripgrep/rg.exe下。如没有，请尝试重装MCS。")
    
    @classmethod
    def Instance(cls):
        """
        单例模式
        """
        if cls._singleton is None:
            cls._singleton = cls(rgPath)
        return cls._singleton
    
    @classmethod
    def _add_results_to_cache(cls, search_key: str, search_path: str, value: List[str]) -> None:
        """
        将搜索结果添加到缓存中
        """
        key = (search_key, search_path)
        if key not in cls._findFilesCache:
            cls._findFilesCache[key] = value
        pass
    
    @classmethod
    def _get_results_from_cache(cls, search_key: str, search_path: str) -> Optional[List[str]]:
        """
        从缓存中获取搜索结果
        """
        key = (search_key, search_path)
        if key in cls._findFilesCache:
            return cls._findFilesCache[key]
        return None

    def find_files_with_text(self, search_text: str, 
                                 path: str = ".", 
                                 pattern: str = "json",
                                 search_depth: int = None,
                                 case_sensitive: bool = True,
                                 whole_word: bool = False) -> List[str]:
        """
        查找包含特定文本的所有JSON文件
        
        Args:
            search_text: 要搜索的文本
            path: 要搜索的路径
            pattern: 要搜索的文件模式，默认为json
            search_depth: 搜索深度，默认为无限制；1=当前目录
            case_sensitive: 是否区分大小写
            whole_word: 是否全词匹配
            
        Returns:
            包含匹配文本的JSON文件路径列表
        """
        # 优先从缓存中获取
        cache = self._get_results_from_cache(search_text, path)
        if cache is not None:
            return cache
        
        cmd = [self.rg_path, "--files-with-matches"]
        
        # 添加区分大小写选项
        if not case_sensitive:
            cmd.append("-i")
            
        # 添加全词匹配选项
        if whole_word:
            cmd.append("-w")
            
        # 添加递归深度控制
        if search_depth is not None:
            cmd.extend(["--max-depth", str(search_depth)])
            
        # 限制只搜索JSON文件
        if pattern:
            cmd.extend(["-t", pattern])
        
        # 添加搜索文本和路径
        cmd.extend([search_text, path])
        
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',  # 明确指定 UTF-8 编码
                check=False
            )
            
            # 返回匹配的文件列表
            if result.stdout:
                results = result.stdout.strip().split("\n")
                # 将结果添加到缓存
                self._add_results_to_cache(search_text, path, results)
                return results
            return []
        except subprocess.SubprocessError as e:
            logging.error(f"Failed to execute search command: {e}", stack_info=True)
            return []
    
    def find_matches_in_file(self, file_path: str, 
                            search_text: str, 
                            case_sensitive: bool = True,
                            whole_word: bool = False) -> List[Tuple[int, str]]:
        """
        在指定文件中查找包含特定文本的行
        
        Args:
            file_path: JSON文件路径
            search_text: 要搜索的文本
            case_sensitive: 是否区分大小写
            whole_word: 是否全词匹配
            
        Returns:
            包含匹配的行号和行内容的列表
        """
        cmd = [self.rg_path, "--line-number"]
        
        # 添加区分大小写选项
        if not case_sensitive:
            cmd.append("-i")
            
        # 添加全词匹配选项
        if whole_word:
            cmd.append("-w")
            
        cmd.extend([search_text, file_path])
        
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            matches = []
            if result.stdout:
                for line in result.stdout.strip().split("\n"):
                    parts = line.split(":", 2)
                    if len(parts) >= 2:
                        line_num = int(parts[1])
                        line_content = parts[2] if len(parts) > 2 else ""
                        matches.append((line_num, line_content))
            
            return matches
        except (subprocess.SubprocessError, ValueError) as e:
            logging.error(f"Error finding matches in {file_path}: {e}", stack_info=True)
            return []
    
    def _validate_json_value(self, value: Any) -> bool:
        """验证值是否可以被JSON序列化"""
        try:
            json.dumps(value)
            return True
        except (TypeError, OverflowError):
            return False
    
    def modify_json_context(self, file_path: str, search_text: str, 
                           modifier_func: callable, 
                           case_sensitive: bool = True,
                           whole_word: bool = False,
                           context_range: int = 1,
                           backup: bool = False,
                           create_path: bool = False,
                           max_depth: int = 100) -> bool:
        """
        修改JSON文件中包含指定文本的部分的上下文
        
        Args:
            file_path: JSON文件路径
            search_text: 要搜索的文本
            modifier_func(json_obj, path, value, file_path): 修改函数，返回修改后的值
            case_sensitive: 是否区分大小写
            whole_word: 是否全词匹配
            context_range: 上下文范围（在多少层结构内查找上下文）
            backup: 是否备份原文件
            create_path: 是否创建不存在的json路径
            max_depth: 最大深度（搜索的上下文范围）
            
        Returns:
            是否成功修改文件
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    json_str = f.read()
                    # 移除单行注释
                    json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
                    # 移除多行注释
                    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
                    json_obj = json.loads(json_str)
                except json.JSONDecodeError as e:
                    logging.error(f"[Error] parsing JSON in file {file_path}: {e}", stack_info=True)
                    return False
        except IOError as e:
            logging.error(f"Error reading file {file_path}: {e}", stack_info=True)
            return False
        
        # 处理逻辑
        if json_obj:
            # 查找并修改
            modified = False
            modified_obj = json_obj
            # modified_obj = copy.deepcopy(json_obj)
            # 查找字符串所在的json层级关系的path
            paths_with_matches = self._find_paths_in_json(
                modified_obj, search_text, case_sensitive=case_sensitive, whole_word=whole_word, max_depth=max_depth
            )
            
            for json_path in paths_with_matches:
                # 获取需要修改的上下文路径
                context_paths = self._get_context_paths(json_path, context_range)
                
                # 对每个上下文路径应用修改
                for ctx_path in context_paths:
                    # 提取当前路径的值
                    current_value = self._get_value_by_path(modified_obj, ctx_path)
                    if current_value is not None:
                        # 创建当前值的深拷贝，避免修改函数直接更改原始数据
                        current_value_copy = copy.deepcopy(current_value)
                        
                        # 应用修改函数
                        try:
                            new_value = modifier_func(modified_obj, ctx_path, current_value_copy, file_path)
                            
                            # 验证修改后的值是否合法
                            if not self._validate_json_value(new_value):
                                logging.warning(f"Modified value at path {ctx_path} is not JSON serializable. Skipping.")
                                continue
                            
                            # 设置修改后的值
                            modified |= self._set_value_by_path(
                                modified_obj, ctx_path, new_value, create_path=create_path
                            )
                        except Exception as e:
                            logging.error(f"Error applying modifier to path {ctx_path}: {e}", stack_info=True)
                            continue

            # 如果有修改，创建备份并写回文件
            if modified:
                if backup:
                    try:
                        shutil.copy2(file_path, f"{file_path}.bak")
                    except IOError as e:
                        logging.error(f"Error creating backup for {file_path}: {e}", stack_info=True)
                        return False
                
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(modified_obj, f, indent=4, ensure_ascii=False)
                except IOError as e:
                    logging.error(f"Error writing to file {file_path}: {e}", stack_info=True)
                    return False
            
        return modified
    
    def batch_modify_json_files(self, search_text: str, modifier_func: callable,
                               path: str = ".", pattern: str = "json",
                               search_depth: int = None,
                               case_sensitive: bool = True,
                               whole_word: bool = False,
                               context_range: int = 1, 
                               backup: bool = False,
                               create_path: bool = False,
                               json_max_depth: int = 100) -> Dict[str, bool]:
        """
        批量修改包含指定文本的所有JSON文件
        
        Args:
            search_text: 要搜索的文本
            modifier_func(json_obj, path, value, file_path): 修改函数
            path: 要搜索的路径
            pattern: 文件匹配模式
            search_depth: 搜索深度，默认搜索所有子目录
            case_sensitive: 是否区分大小写
            whole_word: 是否全词匹配
            context_range: 上下文范围
            backup: 是否备份原文件
            create_path: 是否自动创建不存在的json路径
            json_max_depth: json结构的递归查找的最大深度
            
        Returns:
            每个文件的修改结果
        """
        # 查找所有匹配的文件
        results = {}
        errors = {}
        
        # 使用ripgrep查找
        matching_files = self.find_files_with_text(search_text, path, pattern=pattern, search_depth=search_depth, case_sensitive=case_sensitive, whole_word=whole_word)
        
        # 处理每个文件
        for file_path in matching_files:
            try:
                results[file_path] = self.modify_json_context(
                    file_path, search_text, modifier_func,
                    case_sensitive=case_sensitive, whole_word=whole_word, context_range=context_range, backup=backup,
                    create_path=create_path, max_depth=json_max_depth
                )
            except Exception as e:
                errors[file_path] = str(e)
                results[file_path] = False
                logging.error(f"Error processing {file_path}: {e}", stack_info=True)
        
        if errors:
            logging.warning(f"Encountered errors in {len(errors)} files")
        
        return results
    
    def _find_paths_in_json(self, 
                          json_obj: Any, search_text: str, 
                          case_sensitive: bool = True,
                          whole_word: bool = False,
                          max_depth: int = 100) -> List[List[Union[str, int]]]:
        """
        在JSON对象中查找包含特定文本的所有路径
        
        Args:
            json_obj: JSON对象
            search_text: 要搜索的文本
            case_sensitive: 是否区分大小写
            whole_word: 是否全词匹配
            max_depth: 递归查找的最大深度
            
        Returns:
            包含路径的列表，每个路径是一个键或索引的列表
        """
        paths = []
        
        def _compare_text(text, search):
            # 转换为字符串进行比较
            text_str = str(text)
            search_str = str(search)
            
            if not case_sensitive:
                text_str = text_str.lower()
                search_str = search_str.lower()
            
            if whole_word:
                # 全词匹配实现：使用正则表达式匹配整词
                import re
                pattern = r'\b' + re.escape(search_str) + r'\b'
                return re.search(pattern, text_str) is not None
            else:
                # 普通子字符串匹配
                return text_str.find(search_str) >= 0
        
        def _search_in_obj(obj, current_path=None, depth=0):
            if current_path is None:
                current_path = []
            
            # 检查递归深度
            if depth > max_depth:
                logging.warning(f"Maximum recursion depth ({max_depth}) reached at path: {'.'.join(str(p) for p in current_path)}")
                return
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    # 检查键是否包含搜索文本
                    if _compare_text(key, search_text):
                        paths.append(current_path + [key])
                    
                    # 检查值是否是基本类型
                    if isinstance(value, (str, int, float, bool)) and _compare_text(value, search_text):
                        paths.append(current_path + [key])
                    
                    # 递归搜索嵌套结构
                    elif isinstance(value, (dict, list)):
                        _search_in_obj(value, current_path + [key], depth + 1)
            
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    # 检查项是否是基本类型
                    if isinstance(item, (str, int, float, bool)) and _compare_text(item, search_text):
                        paths.append(current_path + [i])
                    
                    # 递归搜索嵌套结构
                    elif isinstance(item, (dict, list)):
                        _search_in_obj(item, current_path + [i], depth + 1)
        
        _search_in_obj(json_obj)
        return paths
    
    def _get_context_paths(self, path: List[Union[str, int]], context_range: int = 1) -> List[List[Union[str, int]]]:
        """
        获取路径周围的上下文路径
        
        Args:
            path: 当前路径
            context_range: 上下文范围
            
        Returns:
            上下文路径列表
        """
        # 生成包含当前路径及其父路径的上下文
        context_paths = []
        for i in range(max(0, len(path) - context_range), len(path) + 1):
            # 取消判断i>0，因为如果i=0，则当前路径就是根节点，即返回整个json内容
            # if i > 0:
            context_paths.append(path[:i])
        
        return context_paths
    
    def _get_value_by_path(self, json_obj: Any, path: List[Union[str, int]]) -> Any:
        """
        根据路径获取JSON对象中的值
        
        Args:
            json_obj: JSON对象
            path: 路径
            
        Returns:
            路径指向的值，如果路径无效则返回None
        """
        current = json_obj
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            elif isinstance(current, list) and isinstance(key, int) and 0 <= key < len(current):
                current = current[key]
            else:
                return None
        return current
    
    def _set_value_by_path(self, json_obj: Any, path: List[Union[str, int]], value: Any, 
                          create_path: bool = False) -> bool:
        """
        根据路径设置JSON对象中的值
        
        Args:
            json_obj: JSON对象
            path: 路径
            value: 新值
            create_path: 是否自动创建不存在的json路径
            
        Returns:
            是否成功修改
        """
        if not path:
            return False
        
        current = json_obj
        for i, key in enumerate(path[:-1]):
            # 处理字典
            if isinstance(current, dict):
                if create_path and key not in current:
                    # 预判下一级需要的类型
                    next_key = path[i+1]
                    if isinstance(next_key, int):
                        current[key] = []  # 创建列表
                    else:
                        current[key] = {}  # 创建字典
                
                if key in current:
                    current = current[key]
                else:
                    return False
                    
            # 处理列表
            elif isinstance(current, list) and isinstance(key, int):
                if 0 <= key < len(current):
                    current = current[key]
                elif create_path and key == len(current):
                    # 预判下一级需要的类型
                    next_key = path[i+1]
                    if isinstance(next_key, int):
                        current.append([])  # 添加列表
                    else:
                        current.append({})  # 添加字典
                    current = current[-1]
                else:
                    return False
            else:
                return False
        
        # 改进比较逻辑
        def values_equal(v1, v2):
            if isinstance(v1, dict) and isinstance(v2, dict):
                return v1 == v2
            elif isinstance(v1, list) and isinstance(v2, list):
                return v1 == v2
            else:
                try:
                    return v1 == v2
                except:
                    # 处理无法直接比较的情况
                    return str(v1) == str(v2)

        last_key = path[-1]
        
        # 处理字典类型
        if isinstance(current, dict):
            if create_path or last_key in current:
                if last_key not in current or not values_equal(current[last_key], value):
                    current[last_key] = value
                    return True
        # 处理列表类型
        elif isinstance(current, list):
            # 如果索引存在
            if isinstance(last_key, int) and 0 <= last_key < len(current):
                if not values_equal(current[last_key], value):
                    current[last_key] = value
                    return True
            # 如果需要创建新索引
            elif isinstance(last_key, int) and last_key == len(current) and create_path:
                current.append(value)
                return True
        
        return False
    
    def modify_large_json_file(self, file_path: str, search_text: str,
                            modifier_func: callable, max_memory: int = 100*1024*1024,
                            backup: bool = False, case_sensitive: bool = True, whole_word: bool = False,
                            chunk_size: int = 8192) -> bool:
        """
        处理大型JSON文件的专用方法，通过流式处理减少内存占用
        
        Args:
            file_path: JSON文件路径
            search_text: 要搜索的文本
            modifier_func: 修改函数
            max_memory: 最大内存使用量（字节）
            backup: 是否备份原文件
            case_sensitive: 是否区分大小写
            whole_word: 是否全词匹配
            chunk_size: 处理时的块大小
            
        Returns:
            是否成功修改文件
        """
        # 首先检查文件大小
        file_size = os.path.getsize(file_path)
        
        if file_size < max_memory:
            # 如果文件不是很大，使用标准方法
            return self.modify_json_context(file_path, search_text, modifier_func, case_sensitive=case_sensitive, whole_word=whole_word, backup=backup)
            
        # 对于大文件，使用流式解析
        try:
            import ijson  # 确保安装了ijson库
            
            logging.info(f"Processing large file ({file_size} bytes) using streaming parser")
            
            # 创建备份
            if backup:
                try:
                    shutil.copy2(file_path, f"{file_path}.bak")
                except IOError as e:
                    logging.error(f"Error creating backup for {file_path}: {e}", stack_info=True)
                    return False
            
            # 临时文件路径
            temp_output = f"{file_path}.temp"
            
            # 第一阶段：扫描文件找出匹配位置
            matches = self._find_matches_in_large_json(file_path, search_text, case_sensitive=case_sensitive, whole_word=whole_word)
            
            if not matches:
                logging.info(f"No matches found in {file_path}")
                return False
                
            # 第二阶段：应用修改并写入临时文件
            modified = self._apply_changes_to_large_json(
                file_path, temp_output, matches, modifier_func, chunk_size
            )
            
            # 如果成功修改，替换原文件
            if modified:
                try:
                    os.replace(temp_output, file_path)
                    logging.info(f"Successfully modified {file_path}")
                    return True
                except OSError as e:
                    logging.error(f"Error replacing original file: {e}", stack_info=True)
                    return False
            else:
                # 清理临时文件
                try:
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
                except OSError:
                    pass
                return False
                
        except ImportError:
            logging.warning("ijson library not available. Falling back to standard processing.")
            return self.modify_json_context(file_path, search_text, modifier_func, case_sensitive=case_sensitive, whole_word=whole_word, backup=backup)

    def _find_matches_in_large_json(self, file_path: str, search_text: str, 
                                case_sensitive: bool = True, whole_word: bool = False) -> List[Dict]:
        """
        在大型JSON文件中查找匹配的位置
        
        Args:
            file_path: JSON文件路径
            search_text: 要搜索的文本
            case_sensitive: 是否区分大小写
            whole_word: 是否全词匹配
            
        Returns:
            包含匹配位置信息的列表
        """
        import ijson
        
        matches = []
        current_path = []
        current_positions = {}
        
        def _compare_text(text, search):
            # 转换为字符串进行比较
            text_str = str(text)
            search_str = str(search)
            
            if not case_sensitive:
                text_str = text_str.lower()
                search_str = search_str.lower()
            
            if whole_word:
                # 全词匹配实现：使用正则表达式匹配整词
                import re
                pattern = r'\b' + re.escape(search_str) + r'\b'
                return re.search(pattern, text_str) is not None
            else:
                # 普通子字符串匹配
                return text_str.find(search_str) >= 0
        
        with open(file_path, 'rb') as f:
            # 使用ijson的基于事件的解析器
            parser = ijson.parse(f)
            
            for prefix, event, value in parser:
                # 更新当前路径
                if event == 'start_map' or event == 'start_array':
                    current_positions[len(current_path)] = f.tell()
                    if isinstance(value, str):  # 某些ijson后端可能会提供映射键
                        current_path.append(value)
                    else:
                        current_path.append(None)
                
                elif event == 'end_map' or event == 'end_array':
                    if current_path:
                        current_path.pop()
                
                elif event == 'map_key':
                    if current_path and current_path[-1] is None:
                        current_path[-1] = value
                        
                    # 检查键是否包含搜索文本
                    if _compare_text(value, search_text):
                        matches.append({
                            'path': list(current_path),
                            'position': current_positions.get(len(current_path) - 1, 0),
                            'type': 'key',
                            'value': value
                        })
                
                elif event in ('string', 'number', 'boolean', 'null'):
                    # 检查值是否包含搜索文本
                    if _compare_text(str(value), search_text):
                        matches.append({
                            'path': list(current_path),
                            'position': current_positions.get(len(current_path), 0),
                            'type': 'value',
                            'value': value
                        })
        
        # 去重并按位置排序
        unique_matches = []
        seen_positions = set()
        
        for match in matches:
            pos_key = (match['position'], tuple(match['path']))
            if pos_key not in seen_positions:
                seen_positions.add(pos_key)
                unique_matches.append(match)
        
        return sorted(unique_matches, key=lambda x: x['position'])

    def _apply_changes_to_large_json(self, input_path: str, output_path: str, 
                                    matches: List[Dict], modifier_func: callable,
                                    chunk_size: int = 8192) -> bool:
        """
        应用修改到大型JSON文件
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            matches: 匹配位置列表
            modifier_func: 修改函数
            chunk_size: 块大小
            
        Returns:
            是否成功修改
        """
        import ijson
        import tempfile
        
        if not matches:
            return False
        
        # 使用多个临时文件进行分段处理
        temp_files = []
        modified = False
        
        try:
            with open(input_path, 'rb') as infile:
                # 为每个匹配创建一个处理段
                last_pos = 0
                
                for match in matches:
                    pos = match['position']
                    path = match['path']
                    
                    # 跳过已处理的部分
                    if pos < last_pos:
                        continue
                    
                    # 创建一个临时文件用于此段
                    temp_fd, temp_path = tempfile.mkstemp(suffix='.json')
                    os.close(temp_fd)
                    temp_files.append(temp_path)
                    
                    # 从当前位置读取到匹配位置之前
                    infile.seek(last_pos)
                    bytes_to_copy = pos - last_pos
                    
                    with open(temp_path, 'wb') as outfile:
                        # 复制前半部分
                        while bytes_to_copy > 0:
                            chunk = infile.read(min(chunk_size, bytes_to_copy))
                            if not chunk:
                                break
                            outfile.write(chunk)
                            bytes_to_copy -= len(chunk)
                        
                        # 提取当前上下文进行修改
                        infile.seek(pos)
                        
                        # 解析当前位置的JSON对象
                        parser = ijson.parse(infile)
                        builder = ijson.ObjectBuilder()
                        
                        # 找到当前路径的结束位置
                        current_depth = len(path)
                        for prefix, event, value in parser:
                            builder.event(event, value)
                            
                            if event == 'end_map' or event == 'end_array':
                                current_depth -= 1
                                if current_depth < 0:
                                    break
                        
                        # 构建原始对象
                        orig_obj = builder.value
                        
                        # 应用修改函数
                        try:
                            # 创建当前值的深拷贝
                            orig_obj_copy = copy.deepcopy(orig_obj)
                            
                            # 应用修改
                            modified_obj = modifier_func(None, path, orig_obj_copy)
                            
                            # 验证修改后的值
                            if self._validate_json_value(modified_obj):
                                if modified_obj != orig_obj:
                                    modified = True
                                    # 写入修改后的对象
                                    outfile.write(json.dumps(modified_obj, ensure_ascii=False).encode('utf-8'))
                                else:
                                    # 无变化，写入原始对象
                                    outfile.write(json.dumps(orig_obj, ensure_ascii=False).encode('utf-8'))
                            else:
                                logging.warning(f"Modified value at path {path} is not JSON serializable")
                                # 写入原始对象
                                outfile.write(json.dumps(orig_obj, ensure_ascii=False).encode('utf-8'))
                        except Exception as e:
                            logging.error(f"Error applying modifier to path {path}: {e}", stack_info=True)
                            # 发生错误时写回原始对象
                            outfile.write(json.dumps(orig_obj, ensure_ascii=False).encode('utf-8'))
                        
                        # 记录处理结束位置
                        last_pos = infile.tell()
                
                # 处理文件剩余部分
                if last_pos < os.path.getsize(input_path):
                    temp_fd, temp_path = tempfile.mkstemp(suffix='.json')
                    os.close(temp_fd)
                    temp_files.append(temp_path)
                    
                    infile.seek(last_pos)
                    with open(temp_path, 'wb') as outfile:
                        while True:
                            chunk = infile.read(chunk_size)
                            if not chunk:
                                break
                            outfile.write(chunk)
            
            # 合并所有临时文件
            with open(output_path, 'wb') as outfile:
                for temp_path in temp_files:
                    with open(temp_path, 'rb') as infile:
                        while True:
                            chunk = infile.read(chunk_size)
                            if not chunk:
                                break
                            outfile.write(chunk)
            
            return modified
            
        except Exception as e:
            logging.error(f"Error processing large JSON file: {e}", stack_info=True)
            return False
        
        finally:
            # 清理临时文件
            for temp_path in temp_files:
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except OSError:
                    pass
# endregion


# region 测试
# 使用示例
if __name__ == "__main__":
    # 创建修改器
    modifier = JsonModifier.Instance()
    
    # filePath = "./assets/json/bore_machine_ui.json"
    # rgStr = "textures/ui/bore_machine_ui/bore_machine_sfx"

    # 使用示例一
    # # 示例修改函数
    # def example_modifier(json_obj, path, value):
    #     """示例修改函数，根据值类型进行不同的修改"""
    #     if isinstance(value, dict):
    #         if value.get("texture") == rgStr:
    #             # 将uv对应的值改掉
    #             value["uv_size"] = [1, 3]
    #         return value
    #     return value  # 默认不修改
    
    # # 修改特定文件
    # print("修改单个文件:" + filePath)
    # success = modifier.modify_json_context(
    #     file_path=filePath,
    #     search_text=rgStr,
    #     modifier_func=example_modifier,
    #     context_range=1,
    #     backup=False
    # )
    # print(f"修改结果: {'成功' if success else '失败或无修改'}")
    pass

# endregion