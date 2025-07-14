import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from tqdm import tqdm
import logging

from defines import config
from defines.compression_result import CompressionResult
from pipeline.endprocess.html_process import HTMLEndProcess
from pipeline.postprocess.path_postprocess import PathPostProcess
from pipeline.postprocess.save_postprocess import SavePostProcess
from pipeline.postprocess.update_ui_json_postprocess import UpdateUIJsonProcess
from pipeline.preprocess.black_preprocess import BlackPreProcess
from pipeline.preprocess.sfx_preprocess import SFXPreProcess
from pipeline.preprocess.icon_preprocess import IconPreProcess
from pipeline.preprocess.image_type_preprocess import ImageTypePreProcess
from pipeline.preprocess.model_texture_preprocess import ModelTexturePreProcess
from pipeline.preprocess.particle_preprocess import ParticlePreProcess
from pipeline.preprocess.path_preprocess import PathPreProcess
from pipeline.preprocess.size_preprocess import SizePreProcess
from pipeline.preprocess.text_preprocess import TextPreProcess
from pipeline.preprocess.ui_sfx_preprocess import UISFXPreProcess
from pipeline.image_processing_pipeline import ImageProcessingPipeline


"""
游戏资产压缩器类
"""


class GameAssetsCompressor:
    """游戏资产压缩器 - 管道模式实现"""

    def __init__(self, workers=None, report_html=False):
        self.workers = workers or multiprocessing.cpu_count()
        self.pipeline = None
        self._report_html = report_html
        self._init_pipeline()

    def _init_pipeline(self):
        """初始化处理管道"""
        # 创建并配置管道
        self.pipeline = ImageProcessingPipeline(config.NORMAL_QUALITY)
        
        # 添加图片预处理器，用于压缩图片之前的分析、筛选、预处理，需严格按照处理顺序来添加
        self.pipeline.add_preprocess(BlackPreProcess())
        self.pipeline.add_preprocess(PathPreProcess())
        self.pipeline.add_preprocess(ModelTexturePreProcess())
        self.pipeline.add_preprocess(IconPreProcess())  # 判断icon之前，先判断模型纹理
        self.pipeline.add_preprocess(ParticlePreProcess())
        self.pipeline.add_preprocess(SizePreProcess())  # 判断大小，需在icon和特效的判断之后
        # 后面这些需要做图像识别，需放在后面执行
        self.pipeline.add_preprocess(UISFXPreProcess())
        self.pipeline.add_preprocess(TextPreProcess())
        self.pipeline.add_preprocess(SFXPreProcess())
        self.pipeline.add_preprocess(ImageTypePreProcess())
        
        # 添加后处理器，需严格按照处理顺序来添加
        self.pipeline.add_postprocess(PathPostProcess())
        self.pipeline.add_postprocess(SavePostProcess())
        self.pipeline.add_postprocess(UpdateUIJsonProcess())

        # 最终处理，批量处理完成后才执行的逻辑
        if self._report_html:
            self.pipeline.add_endprocess(HTMLEndProcess())
        pass
        
    def batch_process(self, input_path, output_path, res_path, recursive=True) -> List[CompressionResult]:
        """
        批量处理图像文件
        
        Args:
            input_path: 输入文件路径，可以是单个文件或目录
            output_path: 输出文件目录
            res_path: 资源包根目录
            max_workers: 最大工作线程数，默认为CPU数量
            recursive: 是否递归处理子目录中的文件，默认为True
        """
        if os.path.isfile(output_path):
            output_path = os.path.dirname(output_path)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # 收集所有需要处理的文件
        image_files = []
        if isinstance(input_path, list):
            image_files = input_path
        elif os.path.isdir(input_path):
            if recursive:
                for root, _, files in os.walk(input_path):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tga', '.webp', '.bmp', '.tiff')):
                            image_files.append(os.path.join(root, file))
            else:
                for file in os.listdir(input_path):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tga', '.webp', '.bmp', '.tiff')):
                        image_files.append(os.path.join(input_path, file))
        else:
            # 单个文件
            if input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tga', '.webp', '.bmp', '.tiff')):
                image_files.append(input_path)
        
        if not image_files:
            return []
        
        # 准备报告数据
        total_files = len(image_files)
        compression_reports: List[CompressionResult] = []
        
        # 创建进度条
        pbar = tqdm(total=total_files, desc="图片处理进度: ")
        
        # 使用线程池进行并行处理
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            # 为每个文件创建处理任务
            future_to_file = {}
            for img_file in image_files:
                # 确定输出文件路径
                rel_path = os.path.relpath(img_file, input_path) if os.path.isdir(input_path) else os.path.basename(img_file)
                out_file = os.path.join(output_path, rel_path)
                
                # 确保输出目录存在
                os.makedirs(os.path.dirname(out_file), exist_ok=True)
                
                # 提交任务
                future = executor.submit(self._process_single_image, img_file, out_file, res_path, output_path)
                future_to_file[future] = (img_file, out_file)
            
            # 收集结果
            for future in as_completed(future_to_file):
                img_file, out_file = future_to_file[future]
                try:
                    report = future.result()
                    compression_reports.append(report)
                except Exception as e:
                    logging.error(f"{e}", exc_info=True)
                    compression_reports.append(
                        CompressionResult(
                            original_path=img_file,
                            is_compressed=False,
                            info=str(e)
                        )
                    )
                finally:
                    pbar.update(1)
        
        pbar.close()
        
        # 最终处理逻辑
        self.pipeline.do_endprocess(compression_reports, input_path, output_path)

        return compression_reports
    
    def _process_single_image(self, input_file, output_file, res_path, output_dir) -> CompressionResult:
        """
        处理单个图像的辅助方法

        Args:
            input_file: 输入文件路径
            output_file: 输出目录
            res_path: 资源包根目录
            output_dir: 项目输出目录
        """
        try:
            output_path = os.path.dirname(output_file)
            # 使用管道处理图像
            result = self.pipeline.compress_image(
                input_file, 
                output_path,
                res_path
            )
            
            # 处理完成，执行后处理逻辑（后续加确认逻辑，需要确认后，才调用该逻辑）
            self.pipeline.do_postprocess(result, output_dir, res_path)

            return result
        except Exception as e:
            logging.error(f"{e}", exc_info=True)
            return CompressionResult(
                original_path=input_file,
                is_compressed=False,
                info=str(e)
            )

