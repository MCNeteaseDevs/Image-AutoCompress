import logging
import os
import shutil
import time
from typing import Dict, List
from PIL import Image
from typing import Dict

from defines.compression_result import CompressionResult
from utils.ripgrep import JsonModifier
from pipeline.abstract_comps import PipelinePostProcess


class UpdateUIJsonProcess(PipelinePostProcess):
    """修改ui json序列帧数据 后处理器"""
    
    def do(self, image: Image.Image, last_result: Dict, params: Dict = None) -> Dict:
        """
        修改ui的json文件中的uv_size(适用于ui序列帧)

        Args:
            image: PIL图像对象
            last_result: 上一个的处理结果
            params: 额外参数
            
        Returns:
            综合处理结果的字典
        """
        result: CompressionResult = last_result["result"]
        # 处理成功 且 已将图片存储到本地
        if last_result.get("save_to_disk"):
            # 如果是ui序列帧图，则自动修改json
            if result.is_sfx_image:
                res_path = last_result["res_path"]
                file_rel_path = last_result.get("file_rel_path")
                output_dir = last_result["output_dir"]
                # 先在输出目录上搜索
                #   如果搜索不到，再到资源目录上搜索，并把搜索到的文件克隆到输出目录
                # 在输出目录上修改
                file_list = self.find_ui_sfx_files(file_rel_path, output_dir)
                if not file_list:
                    # 从资源包目录中查找
                    file_list = self.find_ui_sfx_files(file_rel_path, res_path)
                    if not file_list:
                        # 找不到使用改贴图的json
                        return last_result
                    # 封装输出目录的路径
                    output_file_path = []
                    for file_path in file_list:
                        output_file_path.append((file_path, file_path.replace(res_path, output_dir)))
                    # 先判断是否在复制中：输出目录下，是否有同名的temp文件
                    self.wait_to_copy_end(output_file_path)
                    # 此时没有在进行复制操作，才将文件复制到输出目录
                    for files in output_file_path:
                        os.makedirs(os.path.dirname(files[1]), exist_ok=True)
                        self.safe_copy_with_marker(files[0], files[1])

                update_files = self.update_ui_sfx_json(file_rel_path, output_dir, (result.compressed_width, result.compressed_height))
                # 封装修改记录，方便后续可视化
                if update_files:
                    # 截断为相对资源包的相对路径
                    # 无论改不改成功，都提示这些文件需要检查
                    file_list = []
                    for file in update_files.keys():
                        file_path = os.path.relpath(file, output_dir)
                        file_list.append(file_path)
                    if file_list:
                        result.update_json_record = file_list
                last_result["update_json_record"] = update_files
        return last_result

    def safe_copy_with_marker(self, source_path: str, target_path: str):
        """安全复制文件，使用临时文件和重命名来确保原子性"""
        # 创建临时路径
        temp_path = f"{target_path}.temp"
            
        try:
            # 移除可能存在的临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # 复制到临时文件
            shutil.copy2(source_path, temp_path)
            
            # 原子重命名确保完整性
            os.replace(temp_path, target_path)
        except Exception as e:
            logging.info(f"safe_copy_with_marker error {e}", exc_info=True)
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
    def wait_to_copy_end(self, file_path: str):
        # 判断同级目录下，是否有temp文件，如果有，则进行等待
        wait_count = 10
        wait_time = 1
        while wait_count > 0:
            wait_count -= 1
            for file in file_path:
                if os.path.exists(f"{file}.temp"):
                    time.sleep(wait_time)
                    break
        pass

    def find_ui_sfx_files(self, image_path_text: str, res_path: str) -> List[str]:
        """
        查找使用了指定序列帧贴图的文件路径

        Args:
            image_path: 贴图路径字符串
            res_path: 资源包根目录
        """
        # 资源路径需限制在ui下
        if not res_path.endswith("ui"):
            res_path = os.path.join(res_path, "ui")

        file_list:List[str] = []

        # 检查是否使用uv动画
        def _check(json_obj, path, value, file_path):
            if isinstance(value, dict):
                if value.get("texture") == image_path_text:
                    # 如果uv的值是@***，则表示是使用动画，则可以认为是序列帧
                    uvVal = value.get("uv")
                    if uvVal and isinstance(uvVal, str) and uvVal.startswith("@"):
                        if file_path not in file_list:
                            file_list.append(file_path)
            return value  # 默认不修改
        
        # 搜索字符串
        modifier = JsonModifier.Instance()
        modifier.batch_modify_json_files(image_path_text, _check, res_path, whole_word=True)
        return file_list

    def update_ui_sfx_json(self, image_path_text: str, res_path: str, uv_size: tuple) -> Dict[str, bool]:
        """
        更新UI序列帧对应的json配置

        Args:
            image_path_text: 贴图相对路径(资源包根目录)
            res_path: 资源包根目录
            uv_size: 贴图总分辨率
        """
        # 资源路径需限制在ui下
        if not res_path.endswith("ui"):
            res_path = os.path.join(res_path, "ui")

        # 动画里定义的帧数
        anim_frame_count = None
        set_uv_size = list(uv_size)

        # 修改内容
        def _update_context(json_obj, path, value, file_path):
            if isinstance(value, dict):
                if value.get("texture") == image_path_text:
                    # 如果uv的值是@***，则表示是使用动画，则可以认为是序列帧
                    uvVal = value.get("uv")
                    if uvVal and isinstance(uvVal, str) and uvVal.startswith("@"):

                        # 获取帧数
                        nonlocal anim_frame_count
                        if anim_frame_count is None:
                            # anim的命名空间
                            ui_namespace = json_obj.get("namespace")
                            uvValList = uvVal.split(".")
                            if len(uvValList) > 0:
                                # 使用了其他json文件的命名空间
                                uvNamespace = uvVal[0].replace("@", "")
                                animKey = uvValList[-1]
                                if uvNamespace and uvNamespace != ui_namespace:
                                    ui_namespace = uvNamespace
                                    anim_frame_count = self._find_uv_anim_frame_count(ui_namespace, animKey, res_path)
                            else:
                                animKey = uvValList[0].replace("@", "")
                            if anim_frame_count is None:
                                # 使用本json的命名空间
                                anim_frame_count = self._find_uv_anim_frame_count_by_jsonobj(json_obj, animKey)
                            # 根据帧数重新计算uv_size（固定横向排列）
                            if anim_frame_count > 1:
                                nonlocal set_uv_size
                                set_uv_size = [
                                    int(uv_size[0] / anim_frame_count),
                                    uv_size[1]
                                ]
                        
                        # 修改uv_size
                        value["uv_size"] = set_uv_size
                        pass
            return value  # 默认不修改
        
        # 搜索字符串
        modifier = JsonModifier.Instance()
        results = modifier.batch_modify_json_files(image_path_text, _update_context, res_path, whole_word=True)
        return results

    def _find_uv_anim_frame_count(self, ui_namespace:str, anim_name: str, res_path: str) -> int:
        """
        查询uv序列帧动画的帧数值

        Args:
            ui_namespace: ui命名空间
            anim_name: 动画名称
            res_path: 资源包根目录
        """
        # TODO: 优化项，这里只需要在找到后，就可停止后续的逻辑，而不是遍历完

        # 资源路径需限制在ui下
        if not res_path.endswith("ui"):
            res_path = os.path.join(res_path, "ui")
        
        # 动画里定义的帧数
        anim_frame_count = 0

        def _find(json_obj, path, value, file_path):
            if isinstance(value, dict):
                if "frame_count" in value:
                    if json_obj.get("namespace") == ui_namespace:
                        nonlocal anim_frame_count
                        anim_frame_count = value["frame_count"]
            return value

        modifier = JsonModifier.Instance()
        modifier.batch_modify_json_files(anim_name, _find, res_path, context_range=1, whole_word=True)
        return anim_frame_count

    def _find_uv_anim_frame_count_by_jsonobj(self, json_obj: dict, anim_name: str) -> int:
        """
        查询uv序列帧动画的帧数值
        
        Args:
            json_obj: json文件的字典数据
            anim_name: 动画名称
        """
        # 动画里定义的帧数
        anim_frame_count = 0

        anim_dict = json_obj.get(anim_name)
        if anim_dict:
            anim_frame_count = anim_dict.get("frame_count", 0)
        return anim_frame_count

