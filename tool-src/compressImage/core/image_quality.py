import os
from typing import Dict
import numpy as np
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import logging
from defines.image_type import ImageType


"""
判断两张图的相似度
"""


DOWNSAMPLE_THRESHOLD = 2048
"""计算SSIM时，降采样的阈值"""

def safe_ssim(img1: np.ndarray, img2: np.ndarray, multichannel=False, channel_axis: int=None):
    """安全计算SSIM，自动处理窗口大小和图像尺寸问题"""
    # 仅对非常大的图像应用降采样，降低影响
    # 存储原始图像尺寸信息用于图像类型判断
    original_max_dim = max(max(img1.shape[0], img1.shape[1]), 
                           max(img2.shape[0], img2.shape[1]))
    
    # 仅当图像非常大且不是像素艺术时才降采样
    if original_max_dim > DOWNSAMPLE_THRESHOLD:
        # 计算降采样因子，确保最大维度降到阈值以下
        scale_factor = DOWNSAMPLE_THRESHOLD / original_max_dim
        
        # 使用高质量的AREA插值方法降采样
        if len(img1.shape) == 3:
            new_h1 = int(img1.shape[0] * scale_factor)
            new_w1 = int(img1.shape[1] * scale_factor)
            new_h2 = int(img2.shape[0] * scale_factor)
            new_w2 = int(img2.shape[1] * scale_factor)
            
            img1_small = cv2.resize(img1, (new_w1, new_h1), interpolation=cv2.INTER_AREA)
            img2_small = cv2.resize(img2, (new_w2, new_h2), interpolation=cv2.INTER_AREA)
            
            img1, img2 = img1_small, img2_small
        else:
            # 灰度图像
            new_h1 = int(img1.shape[0] * scale_factor)
            new_w1 = int(img1.shape[1] * scale_factor)
            new_h2 = int(img2.shape[0] * scale_factor)
            new_w2 = int(img2.shape[1] * scale_factor)
            
            img1_small = cv2.resize(img1, (new_w1, new_h1), interpolation=cv2.INTER_AREA)
            img2_small = cv2.resize(img2, (new_w2, new_h2), interpolation=cv2.INTER_AREA)
            
            img1, img2 = img1_small, img2_small

    # 检查图像最小尺寸
    min_size = min(img1.shape[0], img1.shape[1])
    
    if min_size < 7:
        # 图像太小，直接使用MSE计算
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        return 1.0 / (1.0 + mse/255.0)
    
    if multichannel:
        if len(img1.shape) >= 3 and img1.shape[2] >= 3:
            # 彩色图像(3+通道)
            pass
        else:
            # 灰度图(2通道)
            channel_axis = None
    
    # 图像足够大，使用默认设置
    try:
        return float(ssim(img1, img2, 
                            multichannel=multichannel, 
                            channel_axis=channel_axis))
    except Exception as e:
        logging.error(f"SSIM计算失败: {e}")
        # 出错时回退到MSE
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        return 1.0 / (1.0 + mse/255.0)

def safe_psnr(img1: np.ndarray, img2: np.ndarray, data_range: int=None):
    """安全计算PSNR，处理MSE为零的情况"""
    try:
        # 计算MSE
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        
        # 处理MSE为零的情况（如不处理，会出现警告log）
        if mse == 0:
            return 100.0
        
        return psnr(img1, img2, data_range=data_range)
    except Exception as e:
        logging.warning(f"PSNR计算出错: {e}")
        return 100.0  # 返回一个表示高质量的默认值

def evaluate_image_quality(original: Image.Image, compressed: Image.Image, image_type: str = ImageType.Regular) -> Dict[str, float]:
    """
    图像质量评估函数，针对不同图像类型采用不同的评估策略
    
    Args:
        original: 原始图像
        compressed: 压缩后的图像
        image_type: 图像类型 (pixel_art, regular)
        
    Returns:
        包含多种质量指标的字典
    """
    # 转换为numpy数组以便计算
    original_array = np.array(original)
    compressed_array = np.array(compressed)
    
    # 检查图像维度一致性
    if original_array.shape != compressed_array.shape:
        # raise ValueError(f"图像尺寸不匹配: {original_array.shape} vs {compressed_array.shape}")
        return {}
    
    result = {}
    
    # 判断是否有Alpha通道
    has_alpha = original.mode == 'RGBA' or len(original_array.shape) > 2 and original_array.shape[2] == 4
    
    # 1. 分离RGB和Alpha通道
    if has_alpha:
        original_rgb = original_array[:, :, :3]
        original_alpha = original_array[:, :, 3]
        compressed_rgb = compressed_array[:, :, :3]
        compressed_alpha = compressed_array[:, :, 3]
        
        # 创建布尔掩码 (alpha > 0)
        visible_mask = (original_alpha > 0)  # 布尔掩码更适合索引
        # 主要内容区域 (alpha > 200)
        content_mask = (original_alpha > 200)
        
        # 计算有效区域占比
        visible_ratio = float(np.mean(visible_mask.astype(np.float32)))
        content_ratio = float(np.mean(content_mask.astype(np.float32)))
        
        # 记录通道信息
        result["visible_ratio"] = visible_ratio
        result["content_ratio"] = content_ratio
        
    else:
        original_rgb = original_array
        compressed_rgb = compressed_array
        content_ratio = 1.0
        visible_mask = None
    
    # 2. 基本质量指标计算
    try:
        # 检查图像尺寸
        min_dim = min(original_rgb.shape[0], original_rgb.shape[1])
        
        # 如果有Alpha通道，只在非透明区域进行质量评估
        if has_alpha and np.any(visible_mask):
            # 提取非透明区域边界框
            y_indices, x_indices = np.where(visible_mask)
            if len(y_indices) > 0 and len(x_indices) > 0:
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                
                # 从边界框提取可见区域RGB数据
                orig_visible = original_rgb[y_min:y_max+1, x_min:x_max+1]
                comp_visible = compressed_rgb[y_min:y_max+1, x_min:x_max+1]
                
                # 创建局部掩码用于精确过滤
                local_mask = visible_mask[y_min:y_max+1, x_min:x_max+1]
                
                # 计算SSIM (只在可见区域)
                result["ssim"] = safe_ssim(orig_visible, comp_visible, multichannel=True, channel_axis=2)
                
                # 计算PSNR - 只在非透明像素上（实际没有使用）
                # 将掩码拓展到3通道
                # mask_3d = np.stack([local_mask] * 3, axis=2)
                orig_filtered = orig_visible[local_mask]
                # comp_filtered = comp_visible[local_mask]
                
                if len(orig_filtered) > 0:
                    result["psnr"] = safe_psnr(orig_visible, comp_visible)
                else:
                    result["psnr"] = 100.0  # 默认完美分数
            else:
                # 没有足够的可见像素
                result["ssim"] = 1.0
                result["psnr"] = 100.0
        else:
            # 无Alpha通道或所有像素都透明的情况
            if has_alpha and not np.any(visible_mask):
                # 完全透明
                result["ssim"] = 1.0
                result["psnr"] = 100.0
            else:
                # 标准RGB质量计算
                result["ssim"] = safe_ssim(original_rgb, compressed_rgb, multichannel=True, channel_axis=2)
                result["psnr"] = safe_psnr(original_rgb, compressed_rgb)
        
        # Alpha通道质量独立计算
        if has_alpha and min_dim >= 3:
            win_size = max(3, min(min_dim, 7))
            if win_size % 2 == 0:
                win_size -= 1
                
            # Alpha的SSIM计算应该基于有实际Alpha差异的区域
            # 创建Alpha差异掩码
            alpha_diff_mask = (original_alpha > 0) | (compressed_alpha > 0)
            
            if np.any(alpha_diff_mask):
                # 提取Alpha值不全为0的区域
                ay_indices, ax_indices = np.where(alpha_diff_mask)
                ay_min, ay_max = np.min(ay_indices), np.max(ay_indices)
                ax_min, ax_max = np.min(ax_indices), np.max(ax_indices)
                
                # 提取有意义的Alpha区域
                orig_alpha_region = original_alpha[ay_min:ay_max+1, ax_min:ax_max+1]
                comp_alpha_region = compressed_alpha[ay_min:ay_max+1, ax_min:ax_max+1]
                
                # 计算Alpha通道SSIM
                result["alpha_ssim"] = safe_ssim(orig_alpha_region, comp_alpha_region, multichannel=False)
            else:
                # 两个图像的Alpha都是全0，完全一致
                result["alpha_ssim"] = 1.0
        elif has_alpha:
            # Alpha区域太小，使用MSE
            # 只比较非全0区域的Alpha
            alpha_diff_mask = (original_alpha > 0) | (compressed_alpha > 0)
            if np.any(alpha_diff_mask):
                orig_alpha_valid = original_alpha[alpha_diff_mask]
                comp_alpha_valid = compressed_alpha[alpha_diff_mask]
                result["alpha_mse"] = float(np.mean((orig_alpha_valid - comp_alpha_valid) ** 2))
                result["alpha_ssim"] = 1.0 - min(1.0, result["alpha_mse"] / 255.0)
            else:
                result["alpha_ssim"] = 1.0
    
    except Exception as e:
        logging.error(f"质量评估计算错误: {e}", exc_info=True)
        result["ssim"] = 0.0
        result["psnr"] = 0.0
    
    # 3. 基于图像类型的特殊指标
    if image_type == "pixel_art":
        try:
            # 对于像素艺术，边缘一致性更重要
            if has_alpha and np.any(visible_mask):
                # 只处理非透明区域的边缘
                y_indices, x_indices = np.where(visible_mask)
                if len(y_indices) > 0 and len(x_indices) > 0:
                    y_min, y_max = np.min(y_indices), np.max(y_indices)
                    x_min, x_max = np.min(x_indices), np.max(x_indices)
                    
                    # 从边界框中提取有效区域
                    orig_visible = original_rgb[y_min:y_max+1, x_min:x_max+1]
                    comp_visible = compressed_rgb[y_min:y_max+1, x_min:x_max+1]
                    
                    # 安全的颜色空间转换 - 检查通道数
                    if len(orig_visible.shape) == 3 and orig_visible.shape[2] == 3:
                        # 正确的3通道RGB图像
                        edges_original = cv2.Canny(cv2.cvtColor(orig_visible, cv2.COLOR_RGB2GRAY), 50, 150)
                    elif len(orig_visible.shape) == 3 and orig_visible.shape[2] == 1:
                        # 已经是单通道图像
                        edges_original = cv2.Canny(orig_visible[:,:,0], 50, 150)
                    elif len(orig_visible.shape) == 2:
                        # 已经是灰度图像
                        edges_original = cv2.Canny(orig_visible, 50, 150)
                    else:
                        # 其他情况，尝试转换为灰度
                        orig_gray = np.mean(orig_visible, axis=2).astype(np.uint8) if len(orig_visible.shape) == 3 else orig_visible
                        edges_original = cv2.Canny(orig_gray, 50, 150)
                    
                    # 对压缩图像进行同样处理
                    if len(comp_visible.shape) == 3 and comp_visible.shape[2] == 3:
                        edges_compressed = cv2.Canny(cv2.cvtColor(comp_visible, cv2.COLOR_RGB2GRAY), 50, 150)
                    elif len(comp_visible.shape) == 3 and comp_visible.shape[2] == 1:
                        edges_compressed = cv2.Canny(comp_visible[:,:,0], 50, 150)
                    elif len(comp_visible.shape) == 2:
                        edges_compressed = cv2.Canny(comp_visible, 50, 150)
                    else:
                        comp_gray = np.mean(comp_visible, axis=2).astype(np.uint8) if len(comp_visible.shape) == 3 else comp_visible
                        edges_compressed = cv2.Canny(comp_gray, 50, 150)
                    
                    # # 边缘检测前转为灰度
                    # edges_original = cv2.Canny(cv2.cvtColor(orig_visible, cv2.COLOR_RGB2GRAY), 50, 150)
                    # edges_compressed = cv2.Canny(cv2.cvtColor(comp_visible, cv2.COLOR_RGB2GRAY), 50, 150)
                    
                    # 计算边缘相似性
                    if edges_original.size > 0 and edges_compressed.size > 0:
                        result["edge_similarity"] = safe_ssim(edges_original, edges_compressed, multichannel=False)
                    else:
                        result["edge_similarity"] = 1.0
                    
                    # 颜色数量比较 (只考虑非透明像素)
                    local_mask = visible_mask[y_min:y_max+1, x_min:x_max+1]
                    # mask_3d = np.stack([local_mask] * 3, axis=2)
                    
                    # 提取非透明区域的唯一颜色
                    # np.unique耗时久，更换了其他方式实现
                    if len(orig_visible.shape) == 2:
                        # 灰度图像
                        orig_visible_flat = orig_visible.flatten()
                        comp_visible_flat = comp_visible.flatten()
                        local_mask_flat = local_mask.flatten()
                        
                        # orig_colors = np.unique(orig_visible_flat[local_mask_flat])
                        # comp_colors = np.unique(comp_visible_flat[local_mask_flat])

                        # 使用集合计算唯一值数量
                        orig_unique_values = set(orig_visible_flat[local_mask_flat].tolist())
                        comp_unique_values = set(comp_visible_flat[local_mask_flat].tolist())
                        
                        original_colors = len(orig_unique_values)
                        compressed_colors = len(comp_unique_values)
                    else:
                        # 彩色图像
                        orig_visible_flat = orig_visible.reshape(-1, orig_visible.shape[2])
                        comp_visible_flat = comp_visible.reshape(-1, comp_visible.shape[2])
                        local_mask_flat = local_mask.flatten()
                        
                        # orig_colors = np.unique(orig_visible_flat[local_mask_flat], axis=0)
                        # comp_colors = np.unique(comp_visible_flat[local_mask_flat], axis=0)

                        # 使用位运算将RGB值转换为整数，加速集合操作
                        if orig_visible.shape[2] == 3:
                            # 提取被掩码选中的像素
                            orig_masked = orig_visible_flat[local_mask_flat]
                            comp_masked = comp_visible_flat[local_mask_flat]
                            
                            # 将RGB转换为单一整数
                            orig_r = orig_masked[:, 0].astype(np.int32)
                            orig_g = orig_masked[:, 1].astype(np.int32)
                            orig_b = orig_masked[:, 2].astype(np.int32)
                            orig_ints = (orig_r << 16) | (orig_g << 8) | orig_b
                            
                            comp_r = comp_masked[:, 0].astype(np.int32)
                            comp_g = comp_masked[:, 1].astype(np.int32)
                            comp_b = comp_masked[:, 2].astype(np.int32)
                            comp_ints = (comp_r << 16) | (comp_g << 8) | comp_b
                            
                            # 计算唯一颜色数量
                            original_colors = len(set(orig_ints))
                            compressed_colors = len(set(comp_ints))
                        else:
                            # 对于非三通道图像，使用元组转换
                            orig_masked = orig_visible_flat[local_mask_flat]
                            comp_masked = comp_visible_flat[local_mask_flat]
                            
                            original_colors = len({tuple(row) for row in orig_masked})
                            compressed_colors = len({tuple(row) for row in comp_masked})
                    
                    # original_colors = len(orig_colors)
                    # compressed_colors = len(comp_colors)
                    result["color_preservation"] = min(1.0, compressed_colors / max(1, original_colors))
                else:
                    result["edge_similarity"] = 1.0
                    result["color_preservation"] = 1.0
            else:
                # 无Alpha通道或Alpha全透明的标准处理
                # 安全转换为灰度
                if len(original_rgb.shape) == 3 and original_rgb.shape[2] == 3:
                    orig_gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
                else:
                    orig_gray = np.mean(original_rgb, axis=2).astype(np.uint8) if len(original_rgb.shape) == 3 else original_rgb
                
                if len(compressed_rgb.shape) == 3 and compressed_rgb.shape[2] == 3:
                    comp_gray = cv2.cvtColor(compressed_rgb, cv2.COLOR_RGB2GRAY)
                else:
                    comp_gray = np.mean(compressed_rgb, axis=2).astype(np.uint8) if len(compressed_rgb.shape) == 3 else compressed_rgb
                
                edges_original = cv2.Canny(orig_gray, 50, 150)
                edges_compressed = cv2.Canny(comp_gray, 50, 150)

                # edges_original = cv2.Canny(cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY), 50, 150)
                # edges_compressed = cv2.Canny(cv2.cvtColor(compressed_rgb, cv2.COLOR_RGB2GRAY), 50, 150)
                
                # 计算边缘相似性
                result["edge_similarity"] = safe_ssim(edges_original, edges_compressed, multichannel=False)
                
                # 颜色数量变化比例
                if len(original_rgb.shape) == 2:
                    # 灰度图像 - 只有2个维度
                    original_colors = len(np.unique(original_rgb))
                    compressed_colors = len(np.unique(compressed_rgb))
                else:
                    # 彩色图像 - 有3个维度
                    original_colors = len(np.unique(original_rgb.reshape(-1, original_rgb.shape[2]), axis=0))
                    compressed_colors = len(np.unique(compressed_rgb.reshape(-1, compressed_rgb.shape[2]), axis=0))
                result["color_preservation"] = min(1.0, compressed_colors / max(1, original_colors))
        except Exception as e:
            logging.error(f"像素艺术特殊指标计算错误: {e}", exc_info=True)
            result["edge_similarity"] = 0.5
            result["color_preservation"] = 0.5
    
    # 4. 加权综合评分
    if image_type == "pixel_art":
        # 像素艺术重视边缘保持和颜色保持
        combined_score = (
            result["ssim"] * 0.4 + 
            min(1.0, result["psnr"] / 40.0) * 0.1 + 
            result.get("edge_similarity", 0.0) * 0.3 + 
            result.get("color_preservation", 0.0) * 0.2
        )
    else:  # regular
        # 普通图像更平衡地考量各因素
        combined_score = (
            result["ssim"] * 0.6 + 
            min(1.0, result["psnr"] / 40.0) * 0.4
        )
    
    # 如果有Alpha通道，将Alpha质量也纳入考虑
    if has_alpha:
        # 根据内容区域占比调整Alpha通道的权重
        alpha_weight = min(0.3, content_ratio * 0.5)  # 最高0.3，内容少则权重降低
        regular_weight = 1.0 - alpha_weight
        combined_score = combined_score * regular_weight + result.get("alpha_ssim", 0.0) * alpha_weight
    
    result["combined_score"] = float(combined_score)
    
    return result

def evaluate_compression_quality(original_image: Image.Image, compressed_image_or_path: str, image_type: str = ImageType.Regular) -> Dict[str, float]:
    """
    更精确的压缩质量评估方法

    Args:
        original_image: 原始图像的numpy数组
        compressed_image_or_path: 压缩图像的numpy数组或文件路径
        image_type: 图像类型 ("regular", "pixel_art"等)
    
    Returns:
        包含质量评估指标的字典
    """
    # 1. 确保正确加载压缩图像
    if isinstance(compressed_image_or_path, str):
        try:
            # 如果是字符串路径，尝试读取图像
            if not os.path.exists(compressed_image_or_path):
                logging.error(f"错误：文件不存在 {compressed_image_or_path}", exc_info=True)
                return {"combined_score": 0.0}
                
            compressed_image = cv2.imread(compressed_image_or_path, cv2.IMREAD_UNCHANGED)
            if compressed_image is None:
                logging.error(f"错误：无法读取图像 {compressed_image}", exc_info=True)
                return {"combined_score": 0.0}
        except Exception as e:
            logging.error(f"读取压缩图像错误 {e}", exc_info=True)
            return {"combined_score": 0.0}
    else:
        # 如果已经是numpy数组，直接使用
        compressed_image = compressed_image_or_path
    
    # 获取尺寸
    orig_w, orig_h = original_image.size
    comp_w, comp_h = compressed_image.size
    
    result = {}
    
    # 1. 特征保留评估 - 使用特征检测器比较关键点
    try:
        # 将PIL图像转换为OpenCV格式
        orig_cv = np.array(original_image.convert('RGB'))
        orig_cv = orig_cv[:, :, ::-1].copy()  # RGB to BGR
        
        # 将压缩图像放大到原始尺寸用于特征比较
        if orig_h != comp_h or orig_w != comp_w:
            comp_upscaled = compressed_image.resize((orig_w, orig_h), resample=Image.Resampling.BICUBIC)
            comp_cv = np.array(comp_upscaled.convert('RGB'))
            comp_cv = comp_cv[:, :, ::-1].copy()  # RGB to BGR
        else:
            comp_cv = np.array(compressed_image.convert('RGB'))
            comp_cv = comp_cv[:, :, ::-1].copy()  # RGB to BGR
        
        # 只在灰度图上进行特征检测
        orig_gray = cv2.cvtColor(orig_cv, cv2.COLOR_BGR2GRAY)
        comp_gray = cv2.cvtColor(comp_cv, cv2.COLOR_BGR2GRAY)
        
        # 使用SIFT/ORB/AKAZE等特征检测器
        # SIFT精度高、效果好，但速度慢不少
        # detector = cv2.SIFT_create() if hasattr(cv2, 'SIFT_create') else cv2.ORB_create()
        detector = cv2.ORB_create()
        
        # 检测关键点
        kp1, des1 = detector.detectAndCompute(orig_gray, None)
        kp2, des2 = detector.detectAndCompute(comp_gray, None)
        
        # 计算特征点比例
        if kp1 and kp2:
            feature_ratio = min(1.0, len(kp2) / max(1, len(kp1)))
            result["feature_preservation"] = feature_ratio
        else:
            result["feature_preservation"] = 0.5  # 默认中等分数
    except Exception as e:
        logging.error(f"特征分析错误 {e}", exc_info=True)
        result["feature_preservation"] = 0.5
    
    # 2. 图像视觉差异 - 考虑尺寸因素
    # 将原图按照实际压缩比例缩小，计算"期望"压缩图的样子
    if orig_h != comp_h or orig_w != comp_w:
        scale_ratio = min(comp_h/orig_h, comp_w/orig_w)
        expected_h = max(1, int(orig_h * scale_ratio))
        expected_w = max(1, int(orig_w * scale_ratio))
        
        # 选择合适的插值方法
        interpolation = Image.Resampling.NEAREST if image_type == "pixel_art" else Image.Resampling.LANCZOS
        
        # 将原图缩小到预期尺寸
        expected_small = original_image.resize((expected_w, expected_h), resample=interpolation)
        
        # 计算压缩图与"预期小图"的相似度
        try:
            # 如果尺寸仍有差异，将压缩图调整到预期尺寸
            if expected_h != comp_h or expected_w != comp_w:
                comp_resized = compressed_image.resize((expected_w, expected_h), resample=interpolation)
            else:
                comp_resized = compressed_image
                
            # 计算SSIM和MSE等指标
            small_metrics = evaluate_image_quality(expected_small, comp_resized, image_type)
            # 保留这些指标
            for key, value in small_metrics.items():
                result[key] = value
                
            # 将压缩保真度占比较高
            compression_fidelity = small_metrics.get("combined_score", 0.5)
        except Exception as e:
            logging.error(f"小尺寸比较错误 {e}", exc_info=True)
            compression_fidelity = 0.5
    else:
        # 原图和压缩图尺寸相同，直接计算质量指标
        same_size_metrics = evaluate_image_quality(original_image, compressed_image, image_type)
        for key, value in same_size_metrics.items():
            result[key] = value
        compression_fidelity = same_size_metrics.get("combined_score", 0.5)
    
    # 3. 压缩感知评分 - 结合压缩率和质量
    try:
        # 计算实际压缩比例
        compression_ratio = (orig_w * orig_h) / (comp_w * comp_h)
        
        # 特殊图像类型的额外权重
        if image_type == "pixel_art":
            # 像素艺术对质量更敏感，对压缩率不那么敏感
            quality_weight = 0.8
        else:
            # 普通图像更平衡
            quality_weight = 0.7
            
        # 计算感知评分，平衡质量和压缩率
        result["perceptual_score"] = (
            compression_fidelity * quality_weight + 
            result.get("feature_preservation", 0.5) * (1.0 - quality_weight)
        )
        
        # 对特殊图像类型的额外惩罚
        if image_type == "pixel_art" and compression_ratio > 3:
            # 像素艺术压缩过多会严重影响质量
            result["perceptual_score"] *= (3 / compression_ratio) ** 0.5
    except Exception as e:
        logging.error(f"感知评分计算错误 {e}", exc_info=True)
        result["perceptual_score"] = compression_fidelity  # 回退到基本质量分数
    
    return result
