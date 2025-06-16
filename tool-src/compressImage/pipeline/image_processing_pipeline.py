from io import BytesIO
import logging
import os
import time
from typing import Dict, List
from PIL import Image
import numpy as np

from defines import config
from core.image_quality import evaluate_compression_quality
from defines.compression_result import CompressionResult
from defines.image_type import ImageType
from pipeline.abstract_comps import PipelineEndProcess, PipelinePreprocess, PipelineProcess, PipelinePostProcess
from utils.utils import get_image_resample


"""
主管道类 - 协调整个处理流程
"""


class ImageProcessingPipeline:
    """图像处理管道主类"""
    
    def __init__(self, quality_threshold):
        self.quality_threshold = quality_threshold
        """评估质量阈值"""
        self.preprocess: List[PipelinePreprocess] = []
        """预处理"""
        self.process: List[PipelineProcess] = []
        """处理"""
        self.postprocess: List[PipelinePostProcess] = []
        """后处理"""
        self.endprocess: List[PipelineEndProcess] = []
        """最终处理"""
        
    def add_preprocess(self, process: PipelinePreprocess) -> 'ImageProcessingPipeline':
        """添加预处理组件"""
        self.preprocess.append(process)
        return self  # 允许链式调用
        
    def add_process(self, process: PipelineProcess) -> 'ImageProcessingPipeline':
        """添加处理组件"""
        self.process.append(process)
        return self
        
    def add_postprocess(self, process: PipelinePostProcess) -> 'ImageProcessingPipeline':
        """添加后处理组件"""
        self.postprocess.append(process)
        return self
        
    def add_endprocess(self, process: PipelineEndProcess) -> 'ImageProcessingPipeline':
        """添加最终处理组件"""
        self.endprocess.append(process)
        return self
    
    def _load_image(self, image_path: str) -> Image.Image:
        """加载图像文件"""
        try:
            image = Image.open(image_path)
            return image
        except Exception as e:
            logging.error(f"无法加载图像 {image_path}: {e}")
            raise ValueError(f"无法加载图像: {e}")
    
    def compress_image(self, image_path: str, output_path: str, res_path: str) -> CompressionResult:
        """
        自动降低贴图分辨率(预处理 + 处理，不包括后处理)
        
        Args:
            image_path: 图像路径
            quality_threshold: 质量阈值，覆盖配置中的默认值
        
        Returns:
            最佳结果，如果都不满足质量要求则返回None
        """
        threshold = self.quality_threshold
        # 1. 加载图像
        original_image = self._load_image(image_path)
        # 确保图像格式正确
        if original_image.mode == "P" and "transparency" in original_image.info:
            original_image = original_image.convert("RGBA")
        original_array = np.array(original_image)

        original_width, original_height = original_image.size
        original_size = os.path.getsize(image_path)
        
        # 图像预处理：分析图像特性
        analysis_results = self.do_preprocess(original_image, original_array, image_path, res_path)
        if analysis_results.get("error"):
            return self._get_error_result(image_path, original_size, original_width, original_height, analysis_results["error"])

        # 根据处理结果，修改参数
        if analysis_results.get("quality_threshold"):
            threshold = analysis_results.get("quality_threshold")
        
        # 图像处理：对图像进行压缩
        transform_results = self.do_process(
            original_image, image_path, output_path,
            analysis_results.get("image_type", ImageType.PixelArt), original_size, threshold)
        # 封装result
        transform_results.is_sfx_image = analysis_results.get("is_ui_sfx", False)
        transform_results.is_text_image = analysis_results.get("is_text", False)

        return transform_results

    def do_preprocess(self, original_image: Image.Image, original_array: np.ndarray, image_path: str, res_path: str) -> Dict:
        """
        图像预处理，对图像进行分析，并返回分析结果

        Args:
            original_image: 原始图像对象
            original_array: 原始图像数组
            image_path: 图像路径
            res_path: 资源包根目录

        Returns:
            分析结果，包含以下字段：
            image_type: 图像类型，如"pixel_art", "regular"等
            has_text: 是否包含文本
            is_sfx: 是否为序列帧
        """
        if not self.preprocess:
            return {}
        # 获取原始尺寸
        original_width, original_height = original_image.size

        analysis_results = {
            "file_path": image_path, 
            "res_path": res_path, 
            "original_width": original_width,
            "original_height": original_height,
        }
        
        # 后一个分析器，会以前一个分析器的结果，作为分析的依据
        for analyzer in self.preprocess:
            try:
                result = analyzer.do(original_array, analysis_results)
                analysis_results.update(result)
                logging.debug(f"预处理 {analyzer.get_name()} 结果: {result}")
                if result.get("error"):
                    # 任意一环处理失败，则结束分析逻辑
                    break
            except Exception as e:
                logging.error(f"预处理 {analyzer.get_name()} 失败: {e}", exc_info=True)
        return analysis_results

    def do_process(self, 
                image: Image.Image, image_path: str, output_path: str,
                image_type: str, original_size: int, threshold: float) -> CompressionResult:
        """
        处理单个图像
        
        Args:
            image_path: 图像文件路径
            target_scales: 可选的压缩比例列表，如果为None则使用默认范围
        
        Returns:
            返回处理后的结果
        """
        min_scale: float = 0.1
        max_scale: float = 1.0

        original_width, original_height = image.size
        # 图像分辨率的奇偶
        widthIsOdd = original_width % 2 == 1
        heightIsOdd = original_height % 2 == 1
        # 文件名
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)

        # 最好的压缩结果
        best_result: CompressionResult = None
        # 压缩失败的结果
        fail_result: CompressionResult = None

        # 使用二分搜索策略，进行压缩
        low = min_scale
        high = max_scale
        while high - low > 0.11:  # 收敛阈值不能为0.1，有可能high - low = 0.10000001
            mid = round(int((low + high) * 0.5 * 10) * 0.1, 1)  # 保留1位小数

            # 计算新的分辨率（取偶）
            target_width = max(1, int(original_width * mid) & ~1)
            target_height = max(1, int(original_height * mid) & ~1)

            # 如果压缩后，小于最小分辨率，则停止压缩
            if target_width < config.MIN_RESOLUTION and target_height < config.MIN_RESOLUTION:
                fail_result = self._get_error_result(
                    image_path, original_size, original_width, original_height,
                    "压缩后分辨率过低，不处理"
                )
                break

            # 保持和原始图片的奇偶一致
            if widthIsOdd:
                target_width += 1
            if heightIsOdd:
                target_height += 1

            # 压缩图片、并评分
            result = self._do_compress_image(
                image_path=image_path,
                output_dir=output_path,
                filename=name,
                fileext=ext,
                image=image,
                original_width=original_width,
                original_height=original_height,
                original_size=original_size,
                target_width=target_width,
                target_height=target_height,
                image_type=image_type,
                # TODO: 测试时，可将save_to_disk设置为True
                save_to_disk=False
            )

            # 评估质量
            if result.get_quality_score() >= threshold:
                # 质量满足要求，尝试更低的分辨率
                high = mid
                best_result = result
            else:
                # 质量不满足，尝试更高分辨率
                low = mid
                # 保存最近的结果，并改为压缩失败
                fail_result = result
                fail_result.is_compressed = False
                fail_result.info = "无压缩空间(压缩后质量太低)"
        
        # 如果没有成功的结果，则返回失败结果
        if best_result is None:
            return fail_result
        
        return best_result

    def do_postprocess(self, compress_result: CompressionResult, output_dir: str, res_path: str) -> Dict:
        """
        后处理

        Args:
            result: 图像处理结果
            output_dir: 项目输出目录
            res_path: 资源包根目录

        Returns:
            处理结果，包含以下字段：
            result: 图像处理结果
        """
        if not self.postprocess:
            return {}
        results = {
            "result": compress_result, 
            "output_dir": output_dir, 
            "res_path": res_path, 
        }

        # 后一个处理器，会以前一个处理器的结果，作为处理的依据
        for process in self.postprocess:
            try:
                result = process.do(compress_result.image_data, results)
                results.update(result)
                logging.debug(f"后处理 {process.get_name()} 结果: {result}")
                if result.get("error"):
                    # 任意一环处理失败，则结束处理逻辑
                    break
            except Exception as e:
                logging.error(f"后处理 {process.get_name()} 失败: {e}", exc_info=True)
        return results

    def do_endprocess(self, compress_results: List[CompressionResult], input_path: str, output_path: str) -> Dict:
        """
        最终处理逻辑，当所有图片都处理完成后，才执行
        """
        if not self.endprocess:
            return {}
        results = {
            "results": compress_results, 
            "input_path": input_path, 
            "output_path": output_path, 
        }

        # 后一个处理器，会以前一个处理器的结果，作为处理的依据
        for process in self.endprocess:
            try:
                result = process.do(results)
                results.update(result)
                logging.debug(f"后处理 {process.get_name()} 结果: {result}")
                if result.get("error"):
                    # 任意一环处理失败，则结束处理逻辑
                    break
            except Exception as e:
                logging.error(f"后处理 {process.get_name()} 失败: {e}", exc_info=True)
        return results

    def _get_error_result(self, image_path: str, original_size: int, original_width: int, original_height: int, info: str) -> CompressionResult:
        """
        封装错误结果
        """
        return CompressionResult(
            original_path=image_path,
            original_size=original_size,
            original_width=original_width,
            original_height=original_height,
            compressed_width=original_width,
            compressed_height=original_height,
            ssim_score=1.0,
            psnr_score=1.0,
            perceptual_score=1.0,
            feature_preservation=1.0,
            processing_time=1.0,
            is_compressed=False,
            info=info
        )

    def _do_compress_image(self,
                image_path: str, 
                output_dir: str, 
                filename: str,
                fileext: str,
                image: Image.Image,
                original_width: int,
                original_height: int,
                original_size: int,
                target_width: int, 
                target_height: int,
                image_type: str,
                save_to_disk: bool = False) -> CompressionResult:
        """
        压缩图片，并评估压缩质量
        
        Args:
            image_path: 原始图像路径
            output_dir: 输出目录
            target_width: 目标宽度
            target_height: 目标高度
            image_type: 图像类型（pixel_art, regular等）
            save_to_disk: 是否保存到磁盘，默认不保存
        """
        start_time = time.time()
        
        # 选择插值方法 - 为像素艺术使用最近邻插值，其他使用双三次插值
        interpolation = get_image_resample(image_type)
        
        # 缩放图像
        resized_image = image.resize((target_width, target_height), resample=interpolation)
        
        # 确保图像格式正确
        if resized_image.mode == "P" and "transparency" in resized_image.info:
            resized_image = resized_image.convert("RGBA")
        
        # 构建输出文件路径
        if config.DEBUG:
            output_path = os.path.join(output_dir, f"{filename}_{target_width}x{target_height}{fileext}")
        else:
            output_path = os.path.join(output_dir, f"{filename}{fileext}")
        
        # 估算大小，不写入磁盘
        # 创建一个字节缓冲区用于估算文件大小
        buffer = BytesIO()
        
        # 根据文件扩展名选择合适的保存参数
        if fileext.lower() == ".png":
            resized_image.save(buffer, format="PNG", optimize=True, compression_level=9)
        elif fileext.lower() in [".jpg", ".jpeg"]:
            resized_image.save(buffer, format="JPEG", optimize=True, quality=95)
        else:
            # 其他格式使用默认参数
            resized_image.save(buffer, format=fileext[1:].upper())
        
        # 获取编码后的大小
        compressed_size = buffer.tell()
        buffer.close()
        
        # 如果需要保存，则写入
        if save_to_disk and config.DEBUG:
            os.makedirs(output_dir, exist_ok=True)
            if fileext.lower() == ".png":
                resized_image.save(output_path, format="PNG", optimize=True, compression_level=9)
            elif fileext.lower() in [".jpg", ".jpeg"]:
                resized_image.save(output_path, format="JPEG", optimize=True, quality=95)
            else:
                # 其他格式使用默认参数
                resized_image.save(output_path)
        
        # 评估质量 - 直接使用内存中的图像
        try:
            # 直接使用已经在内存中的图像对象，而不是再次读取文件
            quality_metrics = evaluate_compression_quality(image, resized_image, image_type)
        except Exception as e:
            logging.error(f"质量评估错误 {e}", exc_info=True)
            quality_metrics = {"feature_preservation": 0.0,}
        
        # 从quality_metrics中提取关键指标
        perceptual_score = quality_metrics.get("perceptual_score", 0.0)
        ssim_score = quality_metrics.get("ssim", 0.0)
        psnr_score = quality_metrics.get("psnr", 0.0)
        feature_preservation = quality_metrics.get("feature_preservation", 0.0)
        
        # 创建结果对象，包含质量评估结果
        result = CompressionResult(
            original_path=image_path,
            original_size=original_size,
            original_width=original_width,
            original_height=original_height,
            compressed_path=output_path,
            compressed_size=compressed_size,
            compressed_width=target_width,
            compressed_height=target_height,
            ssim_score=ssim_score,
            psnr_score=psnr_score,
            perceptual_score=perceptual_score,  # 添加感知评分
            feature_preservation=feature_preservation,  # 添加特征保留率
            quality_metrics=quality_metrics,  # 保存完整的质量指标字典
            image_type=image_type,
            processing_time=time.time() - start_time,
            image_data=resized_image  # 保存图像数据，以便后续保存
        )
        
        return result

