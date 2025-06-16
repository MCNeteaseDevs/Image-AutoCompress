import cv2
import numpy as np
from typing import Dict
from scipy import ndimage, signal

from pipeline.abstract_comps import PipelinePreprocess


class SFXPreProcess(PipelinePreprocess):
    """序列帧预处理器 - 检测是否是序列帧动画"""
    
    def do(self, image: np.ndarray, last_result: Dict = None) -> Dict:
        """
        检测图像是否是序列帧动画
        
        Args:
            image: 要分析的图像(numpy数组)
            last_result: 上一个分析器的处理结果
            params: 额外参数
        
        Returns:
            包含处理结果的字典
        """
        if last_result.get("is_icon"):
            return last_result
        if last_result.get("is_particle"):
            return last_result
        # 如果是ui的序列帧，则不需要分析
        if last_result.get("is_ui_sfx"):
            return last_result
        # 如果是文字图，则不再分析
        if last_result.get("is_text"):
            return last_result
        
        result = self.detect_sfx_image(image)
        last_result["is_sfx"] = result.get("success") and result.get("frames_count", 0) > 1
        return last_result

    def detect_sfx_image(self, image: np.ndarray, *, image_path: str = None) -> dict:
        """
        检测序列帧图片并识别各帧的分割线和UV坐标
        
        参数:
            image: 输入图像数组
            image_path: 图像路径(可选)
            
        返回:
            包含检测结果、分割线位置和UV坐标的字典
        """
        # 这个方法，能识别大部分序列帧（如果返回frames_count>1即表示是序列帧），但如果图本身的内容，就有一些是重复的，就会误识别。比如title_img、tips等
        
        # 确保图像已加载
        if image is None and image_path:
            image = cv2.imread(image_path)
        
        if image is None:
            return {"success": False, "error": "无法加载图像"}
        
        # 获取图像尺寸
        if len(image.shape) == 3:
            height, width, _ = image.shape
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            height, width = image.shape
            gray = image.copy()
        
        # 预处理步骤
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # 1. 首先判断是否为单帧图像，避免过度分割
        # 计算图像复杂度特征
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        edge_density = np.count_nonzero(edges) / (width * height)
        
        # 提取水平和垂直投影
        h_projection = np.sum(edges, axis=1)
        v_projection = np.sum(edges, axis=0)
        
        # 平滑投影曲线
        h_projection_smooth = ndimage.gaussian_filter1d(h_projection, sigma=3)
        v_projection_smooth = ndimage.gaussian_filter1d(v_projection, sigma=3)
        
        # 检测是否为单帧图像 (使用更严格的条件)
        is_single_frame = False
        
        # 测试1: 图像尺寸比例 - 序列帧通常较宽或较高
        aspect_ratio = width / height
        if 0.8 < aspect_ratio < 1.2 and min(width, height) < 300:
            # 几乎是正方形的小图像可能是单帧
            is_single_frame = True
        
        # 测试2: 投影峰值分析 - 寻找明显的周期性
        # 分析垂直投影的自相关性来检测周期模式
        max_value = np.max(v_projection_smooth)
        if max_value > 0:
            v_normalized = v_projection_smooth / max_value
        else:
            v_normalized = np.zeros_like(v_projection_smooth)
        v_autocorr = signal.correlate(v_normalized, v_normalized, mode='full')
        v_autocorr = v_autocorr[len(v_normalized)-1:]
        # 归一化
        if v_autocorr[0] > 0:
            v_autocorr = v_autocorr / v_autocorr[0]
        else:
            v_autocorr = np.zeros_like(v_autocorr)
        
        # 序列帧通常有明显的自相关峰值
        dist = width//20
        if dist < 1:
            dist = 1
        peaks, _ = signal.find_peaks(v_autocorr, height=0.5, distance=dist)
        peaks = peaks[peaks > width//10]  # 忽略太近的峰值
        
        # 如果没有明显的周期性峰值，可能是单帧图像
        if len(peaks) < 2:
            is_single_frame_evidence = 0.7  # 证据强度
        else:
            # 检查峰值是否规律
            distances = np.diff(peaks)
            variation = np.std(distances) / np.mean(distances) if len(distances) > 0 else 999
            
            if variation < 0.2:  # 很规律的间距表示可能是序列帧
                is_single_frame_evidence = 0.0
            else:
                is_single_frame_evidence = min(1.0, variation / 2)
        
        # 如果证据强度足够高，认为是单帧
        if is_single_frame_evidence > 0.6:
            is_single_frame = True
        
        # 如果是单帧图像，直接返回整个图像作为一帧
        if is_single_frame:
            return {
                "success": True,
                "arrangement": "single_frame",
                "frames_count": 1,
                "frames": [{
                    "id": 0,
                    "x": 0,
                    "y": 0,
                    "width": width,
                    "height": height,
                    "u1": 0.0,
                    "v1": 0.0,
                    "u2": 1.0,
                    "v2": 1.0
                }],
                "image_width": width,
                "image_height": height,
                "horizontal_splits": [],
                "vertical_splits": [],
                "is_detected_single_frame": True
            }
        
        # 2. 序列帧分析 - 找到真实的分割线
        # 寻找显著峰值 - 使用更严格的阈值
        peak_height_factor = 2.0  # 更高的阈值，减少误检
        min_peak_distance = max(width, height) // 15  # 增加最小距离，避免过密分割
        
        # 检测峰值
        h_peaks, _ = signal.find_peaks(h_projection_smooth, 
                                    height=np.mean(h_projection_smooth) * peak_height_factor, 
                                    distance=min_peak_distance)
        
        v_peaks, _ = signal.find_peaks(v_projection_smooth, 
                                    height=np.mean(v_projection_smooth) * peak_height_factor, 
                                    distance=min_peak_distance)
        
        # 3. 使用霍夫变换检测直线，但采用更严格的参数
        # 增加长度和间隔阈值，减少误判
        min_line_length = min(width, height) // 5  # 要求更长的线条
        max_line_gap = min(width, height) // 20    # 减小允许的线段间隔
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
        
        horizontal_lines = []
        vertical_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                
                # 更严格的角度限制
                if angle < 5 or angle > 175:  # 几乎水平
                    horizontal_lines.append((y1, length, x1, x2))
                elif 85 < angle < 95:  # 几乎垂直
                    vertical_lines.append((x1, length, y1, y2))
        
        # 4. 通过周期性分析验证分割位置
        # 分析垂直方向的周期性 - 用于水平排列的序列帧
        def analyze_periodicity(projection, distance=None):
            if len(projection) < 3:
                return 0, []
                
            # 使用自相关来检测周期
            normalized = projection / np.max(projection)
            autocorr = signal.correlate(normalized, normalized, mode='full')
            autocorr = autocorr[len(normalized)-1:]
            autocorr = autocorr / autocorr[0]
            
            # 寻找周期峰值
            if distance is None:
                distance = len(normalized) // 10
            if distance < 1:
                distance = 1
            
            peaks, _ = signal.find_peaks(autocorr, height=0.4, distance=distance)
            
            # 移除中心峰值
            peaks = peaks[peaks > 10]
            
            if len(peaks) < 2:
                return 0, []
                
            # 计算周期的规律性
            periods = np.diff(peaks)
            regularity = 1.0 - min(1.0, np.std(periods) / np.mean(periods))
            
            # 如果周期规律，使用它来推断分割位置
            mean_period = int(np.mean(periods))
            if regularity > 0.7 and mean_period > 0:
                # 生成规律的分割位置
                inferred_positions = []
                position = mean_period
                while position < len(normalized):
                    inferred_positions.append(position)
                    position += mean_period
                
                return regularity, inferred_positions
            else:
                return regularity, []
        
        # 5. 分析水平和垂直周期性
        h_regularity, h_inferred = analyze_periodicity(h_projection_smooth)
        v_regularity, v_inferred = analyze_periodicity(v_projection_smooth)
        
        # 6. 确定帧排列模式
        # 优先使用强周期性证据
        if v_regularity > 0.7 and len(v_inferred) > 1:
            # 强水平周期性 - 可能是横向排列的序列帧
            arrangement = "single_row"
            v_candidates = v_inferred
            h_candidates = []
        elif h_regularity > 0.7 and len(h_inferred) > 1:
            # 强垂直周期性 - 可能是纵向排列的序列帧
            arrangement = "single_column"
            h_candidates = h_inferred
            v_candidates = []
        else:
            # 使用检测到的峰值作为候选分割线
            arrangement = "grid" if len(h_peaks) > 0 and len(v_peaks) > 0 else "single_row"
            
            # 对候选分割线进行过滤
            # 使用更严格的过滤方法
            def filter_split_candidates(peaks, projection, min_prominence=0.3):
                if len(peaks) == 0:
                    return []
                    
                # 计算每个峰值的显著性
                max_projection = np.max(projection)
                prominences = np.array([projection[p] for p in peaks]) / max_projection
                
                # 只保留显著性高的峰值
                significant_peaks = peaks[prominences > min_prominence]
                
                # 如果太多分割线，只保留最显著的几个
                max_splits = 15  # 序列帧一般不会有太多分割
                if len(significant_peaks) > max_splits:
                    # 根据峰值高度排序，保留最明显的
                    peak_values = np.array([projection[p] for p in significant_peaks])
                    top_indices = np.argsort(peak_values)[-max_splits:]
                    significant_peaks = significant_peaks[top_indices]
                    
                return sorted(significant_peaks)
            
            h_candidates = filter_split_candidates(h_peaks, h_projection_smooth)
            v_candidates = filter_split_candidates(v_peaks, v_projection_smooth)
            
            # 更新排列模式
            if len(h_candidates) > 0 and len(v_candidates) > 0:
                arrangement = "grid"
            elif len(h_candidates) > 0:
                arrangement = "single_column"
            elif len(v_candidates) > 0:
                arrangement = "single_row"
            else:
                arrangement = "single_frame"  # 如果没有明显分割线
        
        # 7. 精确定位分割线 - 使用边缘图像
        def refine_split_position(pos, is_horizontal, margin=5):
            if is_horizontal:
                if pos < margin or pos >= height - margin:
                    return pos
                # 检查局部区域寻找精确的边缘位置    
                region = edges[pos-margin:pos+margin, :]
                edge_density = np.sum(region, axis=1)
                local_peaks = signal.find_peaks(edge_density)[0]
                
                if len(local_peaks) > 0:
                    # 返回局部最强边缘位置
                    return pos - margin + local_peaks[np.argmax(edge_density[local_peaks])]
                return pos
            else:
                if pos < margin or pos >= width - margin:
                    return pos
                # 检查局部区域寻找精确的边缘位置
                region = edges[:, pos-margin:pos+margin]
                edge_density = np.sum(region, axis=0)
                local_peaks = signal.find_peaks(edge_density)[0]
                
                if len(local_peaks) > 0:
                    # 返回局部最强边缘位置
                    return pos - margin + local_peaks[np.argmax(edge_density[local_peaks])]
                return pos
        
        # 精确调整分割线位置
        h_splits = [refine_split_position(pos, True) for pos in h_candidates]
        v_splits = [refine_split_position(pos, False) for pos in v_candidates]
        
        # 额外检查: 确保帧数量合理
        total_frames = (len(h_splits) + 1) * (len(v_splits) + 1) if arrangement == "grid" else \
                    (len(h_splits) + 1) if arrangement == "single_column" else \
                    (len(v_splits) + 1) if arrangement == "single_row" else 1
        
        # 如果帧数量过多，可能是误检，需要调整分割策略
        if total_frames > 20:  # 一般序列帧不太可能超过20帧
            # 尝试减少分割线数量，保留最明显的几个
            max_frame_factor = 10  # 最大帧数因子
            
            if arrangement == "grid" and total_frames > max_frame_factor:
                # 优先调整网格尺寸，保持大致框架
                target_splits = int(np.sqrt(max_frame_factor)) - 1
                
                if len(h_splits) > target_splits:
                    # 按强度排序保留最强的几个水平分割
                    h_projection_at_splits = [h_projection_smooth[pos] for pos in h_splits]
                    top_h_indices = np.argsort(h_projection_at_splits)[-target_splits:]
                    h_splits = [h_splits[i] for i in sorted(top_h_indices)]
                
                if len(v_splits) > target_splits:
                    # 按强度排序保留最强的几个垂直分割
                    v_projection_at_splits = [v_projection_smooth[pos] for pos in v_splits]
                    top_v_indices = np.argsort(v_projection_at_splits)[-target_splits:]
                    v_splits = [v_splits[i] for i in sorted(top_v_indices)]
            elif arrangement == "single_row" and len(v_splits) > max_frame_factor:
                # 保留最强的几个垂直分割
                v_projection_at_splits = [v_projection_smooth[pos] for pos in v_splits]
                top_indices = np.argsort(v_projection_at_splits)[-max_frame_factor:]
                v_splits = [v_splits[i] for i in sorted(top_indices)]
            elif arrangement == "single_column" and len(h_splits) > max_frame_factor:
                # 保留最强的几个水平分割
                h_projection_at_splits = [h_projection_smooth[pos] for pos in h_splits]
                top_indices = np.argsort(h_projection_at_splits)[-max_frame_factor:]
                h_splits = [h_splits[i] for i in sorted(top_indices)]
                
        # 添加边界作为帧的起始和结束点
        h_boundaries = [0] + sorted(h_splits) + [height]
        v_boundaries = [0] + sorted(v_splits) + [width]
        
        # 生成帧信息
        frames = []
        frame_id = 0
        
        # 根据排列方式生成帧信息
        if arrangement == "single_frame":
            frames.append({
                "id": 0,
                "x": 0,
                "y": 0,
                "width": width,
                "height": height,
                "u1": 0.0,
                "v1": 0.0,
                "u2": 1.0,
                "v2": 1.0
            })
        elif arrangement == "grid":
            for i in range(len(h_boundaries) - 1):
                for j in range(len(v_boundaries) - 1):
                    y1, y2 = h_boundaries[i], h_boundaries[i+1]
                    x1, x2 = v_boundaries[j], v_boundaries[j+1]
                    
                    # 跳过太小的帧
                    if (y2 - y1) < height * 0.05 or (x2 - x1) < width * 0.05:
                        continue
                    
                    frames.append({
                        "id": frame_id,
                        "x": int(x1),
                        "y": int(y1),
                        "width": int(x2 - x1),
                        "height": int(y2 - y1),
                        "u1": float(x1) / width,
                        "v1": float(y1) / height,
                        "u2": float(x2) / width,
                        "v2": float(y2) / height
                    })
                    frame_id += 1
        elif arrangement == "single_row":
            for j in range(len(v_boundaries) - 1):
                x1, x2 = v_boundaries[j], v_boundaries[j+1]
                
                # 跳过太小的帧
                if (x2 - x1) < width * 0.05:
                    continue
                
                frames.append({
                    "id": frame_id,
                    "x": int(x1),
                    "y": 0,
                    "width": int(x2 - x1),
                    "height": height,
                    "u1": float(x1) / width,
                    "v1": 0.0,
                    "u2": float(x2) / width,
                    "v2": 1.0
                })
                frame_id += 1
        elif arrangement == "single_column":
            for i in range(len(h_boundaries) - 1):
                y1, y2 = h_boundaries[i], h_boundaries[i+1]
                
                # 跳过太小的帧
                if (y2 - y1) < height * 0.05:
                    continue
                
                frames.append({
                    "id": frame_id,
                    "x": 0,
                    "y": int(y1),
                    "width": width,
                    "height": int(y2 - y1),
                    "u1": 0.0,
                    "v1": float(y1) / height,
                    "u2": 1.0,
                    "v2": float(y2) / height
                })
                frame_id += 1
        
        return {
            "success": True,
            "arrangement": arrangement,
            "frames_count": len(frames),
            "frames": frames,
            "image_width": width,
            "image_height": height,
            "horizontal_splits": [int(y) for y in h_splits],
            "vertical_splits": [int(x) for x in v_splits],
            "h_regularity": float(h_regularity),
            "v_regularity": float(v_regularity)
        }

