import numpy as np
from typing import Dict
import cv2

from defines.image_type import ImageType
from pipeline.abstract_comps import PipelinePreprocess


class ImageTypePreProcess(PipelinePreprocess):
    """图像类型预处理器 - 区分像素艺术和普通图像"""
    
    def do(self, image: np.ndarray, last_result: Dict = None) -> Dict:
        """
        分析图像类型：像素艺术 vs 普通图像
        
        Args:
            image: 要分析的图像(numpy数组)
            last_result: 上一个分析器的处理结果
            params: 额外参数
        
        Returns:
            包含处理结果的字典
        """
        # 因图像类型识别的代码准确率太低，而统一当作像素图处理，效果还行
        last_result["image_type"] = ImageType.PixelArt
        return last_result

    # region 试验方法，留档
    # 第一版：纯色块图，会识别为regular
    def detect_image_type_v1(self, image: np.ndarray) -> str:
        """
        识别图像类型: 像素艺术、纯色块或普通图像
        """
        # 转换为灰度图像
        if len(image.shape) == 3:
            if image.shape[2] >= 3:
                gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
            else:
                gray = image
        else:
            gray = image
        
        # 计算唯一颜色数量
        if len(image.shape) == 3 and image.shape[2] >= 3:
            channels_to_use = 3
            flat_img = image[:, :, :channels_to_use].reshape(-1, channels_to_use)
            unique_colors = len(np.unique(flat_img, axis=0))
            
            # 检测纯色块图像 - 计算主要颜色占比
            if unique_colors <= 5:  # 颜色非常少
                # 使用聚类找到主要颜色
                pixels = flat_img.astype(np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                _, labels, centers = cv2.kmeans(pixels, min(unique_colors, 2), None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                
                # 计算最大颜色簇的占比
                dominant_ratio = np.sum(labels == np.argmax(np.bincount(labels.flatten()))) / labels.size
                
                # 如果有一个颜色占据了大部分区域（超过85%），判定为纯色块
                if dominant_ratio > 0.85:
                    return ImageType.PixelArt  # 将纯色块归为像素艺术类别
        else:
            unique_colors = len(np.unique(gray))
        
        # 尺寸小于某个阈值的图像更可能是像素艺术
        is_small_image = gray.shape[0] < 64 or gray.shape[1] < 64
        
        # 使用更低的阈值进行边缘检测
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.count_nonzero(edges) / (gray.shape[0] * gray.shape[1])
        
        # 检测边缘集中在图像边界的情况（纯色块特征）
        if unique_colors <= 5:
            # 创建内部区域掩码（排除边缘5像素）
            h, w = edges.shape
            if h > 10 and w > 10:  # 确保图像足够大
                inner_mask = np.zeros_like(edges)
                inner_mask[5:-5, 5:-5] = 1
                
                # 计算边缘在内部区域的比例
                inner_edges = np.count_nonzero(edges * inner_mask)
                inner_ratio = inner_edges / np.count_nonzero(edges) if np.count_nonzero(edges) > 0 else 0
                
                # 如果大部分边缘在边框处，更可能是纯色块
                if inner_ratio < 0.2:  # 内部边缘比例很低
                    return ImageType.PixelArt
        
        # 对于颜色少的图像，可以使用更宽松的边缘比例阈值
        edge_threshold = 0.03 if unique_colors <= 5 else (0.05 if unique_colors < 10 else 0.08)
        
        # 判断条件优化
        if (unique_colors < 64 and edge_ratio > edge_threshold) or (unique_colors < 10 and is_small_image):
            return ImageType.PixelArt
        else:
            return ImageType.Regular

    # 第二版：类似UI的背景图，会识别为regular
    def detect_image_type_v2(self, image: np.ndarray) -> str:
        """
        识别图像类型: 像素艺术、UI界面或普通图像
        """
        # 转换为灰度图像
        if len(image.shape) == 3:
            if image.shape[2] >= 3:
                gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
                has_color = True
            else:
                gray = image
                has_color = False
        else:
            gray = image
            has_color = False
        
        # 图像尺寸
        h, w = gray.shape[:2]
        
        # 1. UI界面检测 - 增加专门的UI检测逻辑
        def is_ui_screenshot():
            # 检测水平和垂直线条 - UI通常有明显的水平/垂直线
            h_kernel = np.ones((1, w//20), np.uint8)  # 水平线检测核
            v_kernel = np.ones((h//20, 1), np.uint8)  # 垂直线检测核
            
            # 二值化图像以便检测线条
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # 检测水平线
            h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
            h_line_count = np.count_nonzero(h_lines) / (h * w)
            
            # 检测垂直线
            v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)  
            v_line_count = np.count_nonzero(v_lines) / (h * w)
            
            # 线条检测阈值
            line_threshold = 0.01  # 1%的像素构成线条即认为有明显线条
            has_lines = (h_line_count > line_threshold) or (v_line_count > line_threshold)
            
            # 检测大面积纯色区域
            # 用聚类找主要颜色
            pixels = gray.reshape(-1, 1).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(pixels, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # 计算最大颜色区域占比
            largest_region_ratio = np.max([np.sum(labels == i) / labels.size for i in range(len(centers))])
            
            # 计算局部方差 - UI界面通常有大片平滑区域
            local_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            smooth_image = local_var < 100  # 方差小说明图像整体平滑
            
            # 综合判断
            return (has_lines and smooth_image) or largest_region_ratio > 0.3
        
        # 2. 计算颜色聚类 - 使用K均值聚类来获取实际视觉上的主要颜色
        if has_color:
            pixels = image[:, :, :3].reshape(-1, 3).astype(np.float32)
            cluster_count = 10  # 固定聚类数以避免过度细分
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(pixels, cluster_count, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # 计算每个聚类的大小并排序
            unique_labels, counts = np.unique(labels, return_counts=True)
            sorted_indices = np.argsort(counts)[::-1]  # 从大到小排序
            
            # 计算前3个颜色占比
            if len(counts) >= 3:
                top3_ratio = sum(counts[sorted_indices[:3]]) / labels.size
            else:
                top3_ratio = sum(counts) / labels.size
                
            # 计算直方图的峰值陡度 - UI图像通常有明显的颜色聚集
            hist_peaks = np.sum(counts > (0.05 * labels.size))  # 占比超过5%的颜色数量
            color_concentration = top3_ratio > 0.7  # 前3种颜色占比超过70%
        else:
            # 灰度图使用直方图分析
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist / np.sum(hist)  # 归一化
            
            # 计算主要灰度值(峰值)的数量和占比
            peaks = np.where(hist > 0.05)[0]  # 占比超过5%的灰度值
            hist_peaks = len(peaks)
            
            # 计算前3个主要灰度值的占比
            sorted_hist = np.sort(hist.flatten())[::-1]  # 从大到小排序
            if len(sorted_hist) >= 3:
                top3_ratio = np.sum(sorted_hist[:3])
            else:
                top3_ratio = np.sum(sorted_hist)
                
            color_concentration = top3_ratio > 0.7
        
        # 3. 尝试识别UI界面特征
        ui_detection = is_ui_screenshot()
        
        # 4. 边缘分析 - 调整以适应UI图像
        # 使用多级Canny边缘检测
        edges_low = cv2.Canny(gray, 30, 100)  # 低阈值检测
        edges_high = cv2.Canny(gray, 100, 200)  # 高阈值检测
        
        edge_ratio_low = np.count_nonzero(edges_low) / (h * w)
        edge_ratio_high = np.count_nonzero(edges_high) / (h * w)
        
        # 检查是否有规则的边缘模式 - UI通常有规则的边缘
        regular_edges = edge_ratio_high < 0.1 and edge_ratio_low < 0.25
        
        # 5. 综合判断
        # UI界面判断条件
        is_ui = (ui_detection or color_concentration) and regular_edges
        
        # 像素艺术判断条件 (原始逻辑)
        is_small_image = h < 64 or w < 64
        few_colors = hist_peaks < 20
        
        # 最终判断
        if is_ui or (few_colors and edge_ratio_low > 0.05) or (few_colors and is_small_image):
            return ImageType.PixelArt
        else:
            return ImageType.Regular

    # 第三版：会将现实的风景图识别为PixelArt
    def detect_image_type_v3(self, image: np.ndarray) -> str:
        """
        识别图像类型: 像素艺术、UI界面或普通图像
        """
        # 转换为灰度图像
        if len(image.shape) == 3:
            if image.shape[2] >= 3:
                gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
                has_color = True
            else:
                gray = image
                has_color = False
        else:
            gray = image
            has_color = False
        
        # 图像尺寸
        h, w = gray.shape[:2]
        
        # 1. UI界面检测 - 增加专门的UI检测逻辑
        def is_ui_screenshot():
            # 检测水平和垂直线条 - UI通常有明显的水平/垂直线
            h_kernel = np.ones((1, w//20), np.uint8)  # 水平线检测核
            v_kernel = np.ones((h//20, 1), np.uint8)  # 垂直线检测核
            
            # 二值化图像以便检测线条
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # 检测水平线
            h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
            h_line_count = np.count_nonzero(h_lines) / (h * w)
            
            # 检测垂直线
            v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)  
            v_line_count = np.count_nonzero(v_lines) / (h * w)
            
            # 线条检测阈值
            line_threshold = 0.01  # 1%的像素构成线条即认为有明显线条
            has_lines = (h_line_count > line_threshold) or (v_line_count > line_threshold)
            
            # 检测大面积纯色区域
            # 用聚类找主要颜色
            pixels = gray.reshape(-1, 1).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(pixels, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # 计算最大颜色区域占比
            largest_region_ratio = np.max([np.sum(labels == i) / labels.size for i in range(len(centers))])
            
            # 计算局部方差 - UI界面通常有大片平滑区域
            local_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            smooth_image = local_var < 100  # 方差小说明图像整体平滑
            
            # 综合判断
            return (has_lines and smooth_image) or largest_region_ratio > 0.3
        
        # 2. 计算颜色聚类 - 使用K均值聚类来获取实际视觉上的主要颜色
        if has_color:
            pixels = image[:, :, :3].reshape(-1, 3).astype(np.float32)
            cluster_count = 10  # 固定聚类数以避免过度细分
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(pixels, cluster_count, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # 计算每个聚类的大小并排序
            unique_labels, counts = np.unique(labels, return_counts=True)
            sorted_indices = np.argsort(counts)[::-1]  # 从大到小排序
            
            # 计算前3个颜色占比
            if len(counts) >= 3:
                top3_ratio = sum(counts[sorted_indices[:3]]) / labels.size
            else:
                top3_ratio = sum(counts) / labels.size
                
            # 计算直方图的峰值陡度 - UI图像通常有明显的颜色聚集
            hist_peaks = np.sum(counts > (0.05 * labels.size))  # 占比超过5%的颜色数量
            color_concentration = top3_ratio > 0.7  # 前3种颜色占比超过70%
        else:
            # 灰度图使用直方图分析
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist / np.sum(hist)  # 归一化
            
            # 计算主要灰度值(峰值)的数量和占比
            peaks = np.where(hist > 0.05)[0]  # 占比超过5%的灰度值
            hist_peaks = len(peaks)
            
            # 计算前3个主要灰度值的占比
            sorted_hist = np.sort(hist.flatten())[::-1]  # 从大到小排序
            if len(sorted_hist) >= 3:
                top3_ratio = np.sum(sorted_hist[:3])
            else:
                top3_ratio = np.sum(sorted_hist)
                
            color_concentration = top3_ratio > 0.7
        
        # 3. 尝试识别UI界面特征
        ui_detection = is_ui_screenshot()
        
        # 4. 边缘分析 - 调整以适应UI图像
        # 使用多级Canny边缘检测
        edges_low = cv2.Canny(gray, 30, 100)  # 低阈值检测
        edges_high = cv2.Canny(gray, 100, 200)  # 高阈值检测
        
        edge_ratio_low = np.count_nonzero(edges_low) / (h * w)
        edge_ratio_high = np.count_nonzero(edges_high) / (h * w)
        
        # 检查是否有规则的边缘模式 - UI通常有规则的边缘
        regular_edges = edge_ratio_high < 0.1 and edge_ratio_low < 0.25
        
        # 5. 综合判断
        # UI界面判断条件
        is_ui = (ui_detection or color_concentration) and regular_edges
        
        # 像素艺术判断条件 (原始逻辑)
        is_small_image = h < 64 or w < 64
        few_colors = hist_peaks < 20
        
        # 最终判断
        if is_ui or (few_colors and edge_ratio_low > 0.05) or (few_colors and is_small_image):
            return ImageType.PixelArt
        else:
            return ImageType.Regular
    # endregion
