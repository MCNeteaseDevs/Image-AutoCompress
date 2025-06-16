import cv2
import numpy as np
from typing import Dict

from defines import config
from pipeline.abstract_comps import PipelinePreprocess


class TextPreProcess(PipelinePreprocess):
    """文本图预处理器 - 分析图像是否包含文字"""
    
    def do(self, image: np.ndarray, last_result: Dict = None) -> Dict:
        """
        分析图像中是否包含文字
        
        Args:
            image: 要分析的图像(numpy数组)
            last_result: 上一个分析器的处理结果
            params: 额外参数
        
        Returns:
            包含处理结果的字典
        """
        # 如果是ui的序列帧，则不需要分析
        if last_result.get("is_ui_sfx"):
            return last_result

        # 分析图像是否包含文字
        result = self.detect_text_image(image)
        is_text = result.get("has_text", False)
        last_result["is_text"] = is_text

        # 如果是带文字的图片，则评估质量需要更高
        if is_text:
            last_result["quality_threshold"] = config.TEXT_QUALITY

        return last_result

    def detect_text_image(self, image: np.ndarray, *, image_path: str = None, threshold=0.6, max_image_size=1024) -> dict:
        """
        检测图片里是否包含文字内容，返回检测结果dict
        """
        # --- 1. 预处理: 图像加载与尺寸调整 ---
        if image_path:
            try:
                # 直接使用OpenCV加载图像减少PIL到NumPy的转换开销
                image = cv2.imread(image_path)
            except Exception as e:
                return {"has_text": False, "confidence": 0, "error": str(e)}

        if image is None:
            return {"has_text": False, "confidence": 0, "error": "无法加载图像"}

        # 尺寸调整
        height, width = image.shape[:2]
        max_dim = max(width, height)
        if max_dim > max_image_size:
            scale_factor = max_image_size / max_dim
            image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor, 
                            interpolation=cv2.INTER_NEAREST)
            height, width = image.shape[:2]

        # 快速检查: 图像太小或已知为空白
        if min(height, width) < 20:
            return {"has_text": False, "confidence": 0.1, 
                    "debug_info": {"reason": "图像太小，跳过检测"}}

        # 检查图像通道数并转换为灰度
        if len(image.shape) == 2:
            # 图像已经是灰度的
            gray = image
        elif len(image.shape) == 3:
            # 彩色图像，转换为灰度
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            return {"has_text": False, "confidence": 0, "error": "不支持的图像格式"}

        # 对比度增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)  # 增强后的灰度图替代原灰度图

        # --- 2. 轻量级预处理 ---
        # 用更快的双边滤波替代NLMeans，或者完全跳过降噪
        if max_dim > 300:  # 只对较大图像做降噪
            # 使用高斯模糊替代更昂贵的NLMeans
            denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        else:
            denoised = gray
        
        # --- 3. 快速MSER检测 ---
        # 调整MSER参数加速检测
        mser = cv2.MSER_create(
            delta=3,
            min_area=max(8, int(height * width * 0.0005)),
            max_area=int(height * width * 0.12),
            max_variation=0.25,  # 控制区域灰度变化的容忍度
        )
        
        # 提取MSER区域
        regions, _ = mser.detectRegions(denoised)
        
        # 快速预检查
        if len(regions) < 3:
            return {"has_text": False, "confidence": 0.1, 
                    "debug_info": {"reason": "MSER区域数量不足"}}
        
        # --- 4. 优化MSER区域处理 ---
        # 预分配数组提高效率
        valid_text_regions = []
        total_mser_area = 0
        
        # 减少内存分配和临时变量创建
        total_pixels = height * width
        
        # 有效区域检测
        for region in regions:
            # 计算外接矩形
            x, y, w, h = cv2.boundingRect(region)
            
            # 快速过滤不合理区域
            aspect_ratio = w / float(h) if h > 0 else 0
            if not (0.1 < aspect_ratio < 8):
                continue
                
            region_area = w * h
            area_ratio = region_area / total_pixels
            if not (0.00005 < area_ratio < 0.05):
                continue
                
            # 只对通过基本过滤的区域计算更复杂的特征
            hull = cv2.convexHull(region)
            hull_area = cv2.contourArea(hull)
            density = hull_area / region_area if region_area > 0 else 0
            
            if 0.3 < density < 0.95:
                valid_text_regions.append({
                    "rect": (x, y, w, h),
                    "area": region_area
                })
                total_mser_area += region_area
                
        valid_region_count = len(valid_text_regions)
        
        # 提前判断
        if valid_region_count < 2:
            return {"has_text": False, "confidence": 0.2, 
                    "debug_info": {"reason": "有效文本区域不足"}}
        
        # --- 5. 布局分析简化版 ---
        alignment_score = 0
        
        # 只有足够区域时进行布局分析
        if valid_region_count >= 3:
            # 按垂直位置分组
            min_dim = min(height, width)
            bin_size = max(min_dim // 40, 1) * 10
            
            # 提取中心点
            y_centers = [r["rect"][1] + r["rect"][3]//2 for r in valid_text_regions]
            y_bins = [y // bin_size for y in y_centers]
            
            # 统计每个bin内的区域数
            bin_counts = {}
            for bin_id in y_bins:
                bin_counts[bin_id] = bin_counts.get(bin_id, 0) + 1
            
            # 只分析主要行
            max_line_count = max(bin_counts.values()) if bin_counts else 0
            
            # 使用最大行区域数量直接估算对齐分数
            alignment_score = min(max_line_count / 5.0, 1.0) * 0.8
            
            # 提前判断：如果存在很好的行对齐，可能是文本
            if max_line_count >= 5:
                mser_score = min(valid_region_count / 15.0, 1.0) * 0.7
                layout_score = alignment_score
                
                confidence = (mser_score * 0.5 + layout_score * 0.5)
                
                if confidence >= threshold:
                    result = {
                        "has_text": True,
                        "confidence": float(confidence),
                        "method_scores": {
                            "mser_score": float(mser_score),
                            "layout_score": float(layout_score)
                        },
                        "debug_info": {
                            "valid_regions": valid_region_count,
                            "decision": "布局分析快速判断",
                        }
                    }
                    
                    return result
        
        # --- 6. 分析进一步决定是否进行更耗时的特征提取 ---
        # 仅当MSER特征和布局分析不足以判断时继续
        need_deeper_analysis = (valid_region_count < 10 or alignment_score < 0.6)
        
        # 默认值
        mid_freq_ratio = 0
        direction_uniformity = 0
        lbp_uniformity = 0
        
        # --- 7. 频域与梯度分析 (条件执行) ---
        if need_deeper_analysis:
            # 频域分析 - 只对不确定的情况进行
            # 降采样以加速FFT
            max_fft_size = 256  # 进一步降低上限
            if min(height, width) > max_fft_size:
                scale = max_fft_size / min(height, width)
                fft_img = cv2.resize(denoised, (0, 0), fx=scale, fy=scale, 
                                interpolation=cv2.INTER_AREA)
            else:
                fft_img = denoised
            
            # 计算FFT特征
            f_transform = np.fft.fft2(fft_img)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
            
            # 提取中频能量比
            h, w = magnitude_spectrum.shape
            center_h, center_w = h // 2, w // 2
            inner_radius = min(h, w) // 16
            outer_radius = min(h, w) // 4
            
            y_grid, x_grid = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((y_grid - center_h)**2 + (x_grid - center_w)**2)
            mid_freq_mask = (dist_from_center >= inner_radius) & (dist_from_center <= outer_radius)
            
            mid_freq_energy = np.mean(magnitude_spectrum[mid_freq_mask])
            total_energy = np.mean(magnitude_spectrum)
            mid_freq_ratio = mid_freq_energy / total_energy if total_energy > 0 else 0
            
            # 简化的梯度分析
            # 使用更小的图像计算梯度方向
            if min(height, width) > 300:
                scale = 300 / min(height, width)
                grad_img = cv2.resize(denoised, (0, 0), fx=scale, fy=scale, 
                                    interpolation=cv2.INTER_AREA)
            else:
                grad_img = denoised
                
            # 计算梯度
            sobelx = cv2.Sobel(grad_img, cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(grad_img, cv2.CV_32F, 0, 1, ksize=3)
            
            # 计算梯度方向
            gradient_direction = np.arctan2(sobely, sobelx) * 180 / np.pi
            
            # 计算方向均匀性
            hist_bins = 18
            direction_hist, _ = np.histogram(gradient_direction.flat, bins=hist_bins, range=(-180, 180))
            direction_hist = direction_hist / direction_hist.sum()
            
            # 简化熵计算
            non_zeros = direction_hist > 1e-10
            if np.any(non_zeros):
                direction_entropy = -np.sum(direction_hist[non_zeros] * np.log2(direction_hist[non_zeros]))
                direction_uniformity = 1 - (direction_entropy / np.log2(hist_bins))
            
            # --- 8. LBP分析 (进一步优化和条件执行) ---
            # 仅当其他特征仍不确定时计算LBP
            if valid_region_count < 5 and alignment_score < 0.5:
                # 使用更高效的LBP计算
                # 限制图像尺寸
                max_lbp_size = 200  # 降低最大尺寸限制
                if min(height, width) > max_lbp_size:
                    scale = max_lbp_size / min(height, width)
                    lbp_img = cv2.resize(denoised, (0, 0), fx=scale, fy=scale, 
                                    interpolation=cv2.INTER_AREA)
                else:
                    lbp_img = denoised
                
                # 优化的LBP计算 - 采样而不是全计算
                h, w = lbp_img.shape
                
                # 采样计算LBP (如果图像足够大，只取1/4的像素)
                step = 2 if h*w > 10000 else 1
                
                # 快速计算LBP特征
                lbp_size = ((h-2)//step) * ((w-2)//step)
                lbp = np.zeros(lbp_size, dtype=np.uint8)
                idx = 0
                
                for y in range(1, h-1, step):
                    for x in range(1, w-1, step):
                        if idx >= lbp_size:
                            break
                        center = lbp_img[y, x]
                        code = 0
                        
                        # 快速计算8邻域LBP
                        if lbp_img[y-1, x-1] >= center: code += 1
                        if lbp_img[y-1, x] >= center: code += 2
                        if lbp_img[y-1, x+1] >= center: code += 4
                        if lbp_img[y, x+1] >= center: code += 8
                        if lbp_img[y+1, x+1] >= center: code += 16
                        if lbp_img[y+1, x] >= center: code += 32
                        if lbp_img[y+1, x-1] >= center: code += 64
                        if lbp_img[y, x-1] >= center: code += 128
                        
                        lbp[idx] = code
                        idx += 1
                        
                    if idx >= lbp_size:
                        break
                
                # 计算直方图
                hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
                lbp_hist = hist.astype("float") / hist.sum()
                
                # 计算熵
                non_zeros = lbp_hist > 1e-10
                lbp_entropy = -np.sum(lbp_hist[non_zeros] * np.log2(lbp_hist[non_zeros]))
                lbp_uniformity = 1 - (lbp_entropy / np.log2(256))
        
        # --- 9. 简化评分计算 ---
        # MSER区域评分
        mser_coverage_ratio = total_mser_area / total_pixels if total_pixels > 0 else 0
        
        # 计算各项分数
        mser_score = min(valid_region_count / 15.0, 1.0) * 0.7 if valid_region_count >= 5 else (
                    min(valid_region_count / 10.0, 0.7) * 0.7 if valid_region_count >= 3 else 
                    min(valid_region_count / 5.0, 0.3) * 0.7)
        
        layout_score = alignment_score
        
        # 频率域评分
        freq_score = 0.8 if 1.1 < mid_freq_ratio < 1.8 else (0.5 if 0.9 < mid_freq_ratio < 2.0 else 0.0)
        
        # 方向评分
        gradient_score = direction_uniformity * 0.7
        
        # 纹理评分
        texture_score = lbp_uniformity * 0.6
        
        # 加权计算最终分数
        final_score = (
            mser_score * 0.4 +      # 增加MSER权重
            layout_score * 0.3 +
            freq_score * 0.15 +
            gradient_score * 0.1 +
            texture_score * 0.05    # 降低纹理特征权重
        )
        
        # 应用规则调整
        if valid_region_count > 10 and alignment_score < 0.3:
            final_score *= 0.7
        
        if valid_region_count <= 3 and alignment_score > 0.8:
            final_score = max(final_score, 0.6)
        
        # 创建结果
        result = {
            "has_text": final_score >= threshold,
            "confidence": float(final_score),
            "method_scores": {
                "mser_score": float(mser_score),
                "layout_score": float(layout_score),
                "frequency_score": float(freq_score),
                "gradient_score": float(gradient_score),
                "texture_score": float(texture_score)
            },
            "debug_info": {
                "valid_regions": valid_region_count,
                "mser_coverage": float(mser_coverage_ratio),
                "mid_freq_ratio": float(mid_freq_ratio),
            }
        }
        
        return result

