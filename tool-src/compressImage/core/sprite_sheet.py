from datetime import datetime
import numpy as np
import cv2
import json
from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt

"""
切分序列帧图

# 该代码是尝试验证的中间产物，实际仍不能投入使用。
"""


class SpriteSheetSlicer:
    def __init__(self, image_path, output_dir="sliced_sprites"):
        """初始化精灵图切片器"""
        self.image_path = image_path
        self.output_dir = output_dir
        
        # 加载图像和创建输出目录
        self.image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if self.image.shape[2] == 3:  # 如果没有Alpha通道，添加一个
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2BGRA)
        
        # 确保输出目录存在
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 提取文件名作为前缀
        self.file_prefix = Path(image_path).stem
    
    def auto_slice(self, alpha_threshold=10):
        """自动检测并切片精灵图"""
        # 提取Alpha通道
        alpha_channel = self.image[:, :, 3]
        
        # 二值化Alpha通道，将Alpha值低于阈值的像素标记为0，其他标记为255
        _, binary = cv2.threshold(alpha_channel, alpha_threshold, 255, cv2.THRESH_BINARY)
        
        # 寻找连通区域
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤掉太小的区域
        min_area = 100  # 最小区域阈值，可调整
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        print(f"检测到 {len(valid_contours)} 个潜在精灵")
        
        # 提取每个精灵的边界矩形
        sprites = []
        for i, contour in enumerate(valid_contours):
            x, y, w, h = cv2.boundingRect(contour)
            # 稍微扩大边界以确保不裁剪到精灵边缘
            padding = 2
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(self.image.shape[1] - x, w + padding * 2)
            h = min(self.image.shape[0] - y, h + padding * 2)
            sprites.append((x, y, w, h))
        
        # 检测是否是规则网格排列
        grid_info = self._detect_grid(sprites)
        
        # 如果检测到规则网格，使用网格切片，否则使用单独精灵切片
        if grid_info:
            rows, cols, cell_width, cell_height = grid_info
            sprites = self._grid_slice(rows, cols, cell_width, cell_height)
            print(f"检测到规则网格: {rows}行 x {cols}列, 单元格尺寸: {cell_width}x{cell_height}")
        
        # 保存切片后的精灵
        self._save_sprites(sprites)
        
        return sprites
    
    def _detect_grid(self, sprites):
        """尝试检测规则的网格排列"""
        if not sprites:
            return None
        
        # 提取所有精灵的宽度和高度
        widths = [w for _, _, w, _ in sprites]
        heights = [h for _, _, _, h in sprites]
        
        # 计算宽度和高度的众数（最常见值）
        def most_common(lst):
            return max(set(lst), key=lst.count)
        
        # 允许一定的误差范围
        def close_enough(a, b, tolerance=5):
            return abs(a - b) <= tolerance
        
        # 将相近的尺寸归类
        def cluster_values(values, tolerance=5):
            clusters = []
            for v in values:
                for i, cluster in enumerate(clusters):
                    if close_enough(v, cluster[0], tolerance):
                        cluster.append(v)
                        break
                else:
                    clusters.append([v])
            
            # 返回每个集群的平均值
            return [sum(cluster) / len(cluster) for cluster in clusters if cluster]
        
        # 聚类宽度和高度
        clustered_widths = cluster_values(widths)
        clustered_heights = cluster_values(heights)
        
        # 如果只有一种主要宽度和高度，可能是规则网格
        if len(clustered_widths) == 1 and len(clustered_heights) == 1:
            cell_width = int(clustered_widths[0])
            cell_height = int(clustered_heights[0])
            
            # 检测X坐标的分布
            x_coords = sorted([x for x, _, _, _ in sprites])
            y_coords = sorted([y for _, y, _, _ in sprites])
            
            # 尝试检测规则的X坐标间隔
            x_diffs = [x_coords[i+1] - x_coords[i] for i in range(len(x_coords)-1)]
            y_diffs = [y_coords[i+1] - y_coords[i] for i in range(len(y_coords)-1)]
            
            x_intervals = cluster_values(x_diffs)
            y_intervals = cluster_values(y_diffs)
            
            # 如果有规则的间隔，可能是规则网格
            if len(x_intervals) == 1 and len(y_intervals) == 1:
                # 计算列数和行数
                cols = len(set(x_coords))
                rows = len(set(y_coords))
                
                if cols * rows == len(sprites):
                    return rows, cols, cell_width, cell_height
        
        # 通过图像尺寸和精灵大小推测网格
        img_height, img_width = self.image.shape[:2]
        
        # 使用众数作为单元格尺寸
        common_width = int(most_common(widths))
        common_height = int(most_common(heights))
        
        # 估算列数和行数
        est_cols = img_width // common_width
        est_rows = img_height // common_height
        
        # 验证估算的行列数
        if abs(est_cols * est_rows - len(sprites)) <= 2:  # 允许有少量误差
            return est_rows, est_cols, common_width, common_height
            
        return None
    
    def _grid_slice(self, rows, cols, cell_width, cell_height):
        """基于规则网格切片精灵"""
        sprites = []
        for row in range(rows):
            for col in range(cols):
                x = col * cell_width
                y = row * cell_height
                sprites.append((x, y, cell_width, cell_height))
        return sprites
    
    def slice_by_grid(self, rows, cols):
        """手动指定行列数进行网格切片"""
        img_height, img_width = self.image.shape[:2]
        cell_width = img_width // cols
        cell_height = img_height // rows
        
        sprites = self._grid_slice(rows, cols, cell_width, cell_height)
        self._save_sprites(sprites)
        return sprites
    
    def _save_sprites(self, sprites):
        """保存切片后的精灵"""
        for i, (x, y, w, h) in enumerate(sprites):
            sprite = self.image[y:y+h, x:x+w]
            # OpenCV使用BGR顺序，转换为RGB以便PIL处理
            sprite_rgb = cv2.cvtColor(sprite, cv2.COLOR_BGRA2RGBA)
            # 转换为PIL Image以便简单保存透明PNG
            pil_img = Image.fromarray(sprite_rgb)
            output_path = os.path.join(self.output_dir, f"{self.file_prefix}_{i+1}.png")
            pil_img.save(output_path, "PNG")
            print(f"保存精灵: {output_path}")
    
    def visualize_detection(self, sprites):
        """可视化检测到的精灵边界"""
        viz_image = self.image.copy()
        
        for x, y, w, h in sprites:
            cv2.rectangle(viz_image, (x, y), (x+w, y+h), (0, 255, 0, 255), 2)
        
        # 转换为RGB显示
        viz_rgb = cv2.cvtColor(viz_image, cv2.COLOR_BGRA2RGBA)
        plt.figure(figsize=(10, 10))
        plt.imshow(viz_rgb)
        plt.title(f"检测到 {len(sprites)} 个精灵")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


class GridSpriteSheetSlicer:
    """优化用于切割规则网格排列序列帧的精灵图工具"""
    
    def __init__(self, image_path, output_dir="output_sprites"):
        """
        初始化切片器
        
        参数:
            image_path (str): 精灵图路径
            output_dir (str): 输出目录
        """
        self.image_path = image_path
        self.file_prefix = os.path.splitext(os.path.basename(image_path))[0]
        
        # 读取图像 (保留Alpha通道)
        self.image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if self.image is None:
            raise ValueError(f"无法加载图片: {image_path}")
            
        # 创建输出目录
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 图像尺寸
        self.height, self.width = self.image.shape[:2]
        print(f"已加载精灵图: {self.width}x{self.height} 像素")
    
    def detect_grid(self, gap_threshold=5):
        """
        自动检测网格结构 - 改进版
        
        参数:
            gap_threshold (int): 透明间隙最小像素数
                
        返回:
            dict: 包含网格信息的字典
        """
        # 如果图像有Alpha通道，用它来检测间隙
        if self.image.shape[2] == 4:
            alpha = self.image[:, :, 3]
        else:
            # 如果没有Alpha，尝试使用RGB差异检测边界
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            _, alpha = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # 添加高斯模糊以减少噪声影响
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
        
        # 水平投影：检测行
        h_projection = np.sum(alpha, axis=1)
        h_gaps = np.where(h_projection <= gap_threshold * self.width)[0]
        
        # 垂直投影：检测列
        v_projection = np.sum(alpha, axis=0)
        v_gaps = np.where(v_projection <= gap_threshold * self.height)[0]
        
        # 如果没有检测到间隙，尝试使用均匀网格
        if len(h_gaps) == 0 or len(v_gaps) == 0:
            print("警告: 未检测到明显间隙，尝试估算均匀网格")
            # 尝试使用图像分析估计元素数量
            return self._estimate_uniform_grid()
        
        # 分析行间隙
        row_boundaries = self._find_boundaries(h_gaps)
        col_boundaries = self._find_boundaries(v_gaps)
        
        # 根据边界计算单元格
        row_starts = [0] + [b + 1 for b in row_boundaries]
        row_ends = [b - 1 for b in row_boundaries] + [self.height - 1]
        if len(row_starts) != len(row_ends):
            print("警告: 行边界检测异常")
            return self._estimate_uniform_grid()
            
        col_starts = [0] + [b + 1 for b in col_boundaries]
        col_ends = [b - 1 for b in col_boundaries] + [self.width - 1]
        if len(col_starts) != len(col_ends):
            print("警告: 列边界检测异常")
            return self._estimate_uniform_grid()
        
        # 计算平均单元格尺寸
        cell_heights = [row_ends[i] - row_starts[i] + 1 for i in range(len(row_starts))]
        cell_widths = [col_ends[i] - col_starts[i] + 1 for i in range(len(col_starts))]
        
        if not cell_heights or not cell_widths:
            print("警告: 未检测到单元格")
            return self._estimate_uniform_grid()
            
        avg_cell_height = sum(cell_heights) / len(cell_heights)
        avg_cell_width = sum(cell_widths) / len(cell_widths)
        
        # 计算行列数
        rows = len(row_starts)
        cols = len(col_starts)
        
        print(f"检测到网格: {rows}行 x {cols}列, 单元格平均尺寸: {avg_cell_width:.1f}x{avg_cell_height:.1f}像素")
        
        # 返回网格定义与精确的单元格边界
        return {
            'rows': rows,
            'cols': cols,
            'avg_cell_width': avg_cell_width,
            'avg_cell_height': avg_cell_height,
            'row_starts': row_starts,
            'row_ends': row_ends,
            'col_starts': col_starts,
            'col_ends': col_ends
        }
    
    def _find_boundaries(self, gaps):
        """找出间隙的分隔边界"""
        if len(gaps) == 0:
            return []
            
        # 分组连续的间隙像素
        gap_groups = []
        current_group = [gaps[0]]
        
        for i in range(1, len(gaps)):
            if gaps[i] == gaps[i-1] + 1:
                current_group.append(gaps[i])
            else:
                gap_groups.append(current_group)
                current_group = [gaps[i]]
        
        gap_groups.append(current_group)
        
        # 找出每个间隙的中点作为分割线
        boundaries = []
        for group in gap_groups:
            if len(group) >= 1:  # 最小间隙阈值
                boundaries.append(int(sum(group) / len(group)))
        
        return boundaries

    def _estimate_uniform_grid(self):
        """当无法检测到明显间隙时，尝试估计均匀网格"""
        # 设置一个合理的默认值作为起点
        default_sprite_size = 32  # 像素
        
        # 尝试根据图像尺寸和典型精灵尺寸估计
        est_cols = max(1, self.width // default_sprite_size)
        est_rows = max(1, self.height // default_sprite_size)
        
        # 根据估计的网格均匀划分
        cell_width = self.width // est_cols
        cell_height = self.height // est_rows
        
        row_starts = [i * cell_height for i in range(est_rows)]
        row_ends = [(i+1) * cell_height - 1 for i in range(est_rows)]
        col_starts = [i * cell_width for i in range(est_cols)]
        col_ends = [(i+1) * cell_width - 1 for i in range(est_cols)]
        
        # 确保最后一行/列覆盖到图像边缘
        row_ends[-1] = self.height - 1
        col_ends[-1] = self.width - 1
        
        print(f"估计均匀网格: {est_rows}行 x {est_cols}列, 单元格大小: {cell_width}x{cell_height}像素")
        
        return {
            'rows': est_rows,
            'cols': est_cols,
            'avg_cell_width': cell_width,
            'avg_cell_height': cell_height,
            'row_starts': row_starts,
            'row_ends': row_ends,
            'col_starts': col_starts,
            'col_ends': col_ends
        }

    def slice_sprites(self, grid_info=None, visualize=True):
        """
        根据检测到的网格切片精灵
        
        参数:
            grid_info: 网格信息，如果为None则自动检测
            visualize: 是否可视化切片结果
            
        返回:
            list: 切片后的精灵列表
        """
        if grid_info is None:
            grid_info = self.detect_grid()
            
        if grid_info is None:
            print("无法切片: 未能检测到有效网格")
            return None
            
        sprites = []
        
        # 使用精确的边界进行切片
        for row_idx in range(grid_info['rows']):
            y_start = grid_info['row_starts'][row_idx]
            y_end = grid_info['row_ends'][row_idx]
            
            for col_idx in range(grid_info['cols']):
                x_start = grid_info['col_starts'][col_idx]
                x_end = grid_info['col_ends'][col_idx]
                
                # 验证边界有效性
                if y_start > y_end or x_start > x_end or y_end >= self.height or x_end >= self.width:
                    print(f"警告: 帧 {row_idx*grid_info['cols']+col_idx+1} 边界无效: ({x_start},{y_start}) - ({x_end},{y_end})")
                    continue
                    
                # 确保有效切片区域
                if y_end - y_start <= 0 or x_end - x_start <= 0:
                    print(f"警告: 帧 {row_idx*grid_info['cols']+col_idx+1} 尺寸无效: 宽={x_end-x_start+1}, 高={y_end-y_start+1}")
                    continue
                
                sprite = self.image[y_start:y_end+1, x_start:x_end+1]
                
                # 验证sprite不为空
                if sprite.size == 0:
                    print(f"警告: 帧 {row_idx*grid_info['cols']+col_idx+1} 切片结果为空")
                    continue
                
                # 保存精灵
                frame_number = row_idx * grid_info['cols'] + col_idx + 1
                output_path = os.path.join(self.output_dir, f"{self.file_prefix}_frame{frame_number:03d}.png")
                
                # OpenCV使用BGR(A)顺序，转换为RGB(A)以便PIL处理
                try:
                    if sprite.shape[2] == 4:
                        sprite_rgb = cv2.cvtColor(sprite, cv2.COLOR_BGRA2RGBA)
                    else:
                        sprite_rgb = cv2.cvtColor(sprite, cv2.COLOR_BGR2RGB)
                        
                    pil_img = Image.fromarray(sprite_rgb)
                    pil_img.save(output_path, "PNG")
                    
                    sprites.append({
                        'image': sprite,
                        'path': output_path,
                        'position': (x_start, y_start, x_end-x_start+1, y_end-y_start+1),
                        'frame': frame_number
                    })
                    
                    print(f"保存精灵帧 {frame_number}: {output_path}")
                except Exception as e:
                    print(f"处理帧 {frame_number} 时出错: {e}")
                    print(f"  - 切片区域: ({x_start},{y_start}) 到 ({x_end},{y_end})")
                    print(f"  - 精灵形状: {sprite.shape if hasattr(sprite, 'shape') else '无效'}")
        
        if visualize and sprites:  # 只在有成功切片时可视化
            self.visualize_grid(grid_info)
            
        return sprites
    
    def visualize_grid(self, grid_info=None):
        """可视化检测到的网格"""
        if grid_info is None:
            grid_info = self.detect_grid()
            if grid_info is None:
                print("无法可视化: 未能检测到有效网格")
                return
        
        # 创建可视化图像
        viz_image = self.image.copy()
        
        # 绘制行分隔线
        for row in range(1, grid_info['rows']):
            y = grid_info['row_starts'][row] - 1
            cv2.line(viz_image, (0, y), (self.width, y), (0, 255, 0, 255), 1)
        
        # 绘制列分隔线
        for col in range(1, grid_info['cols']):
            x = grid_info['col_starts'][col] - 1
            cv2.line(viz_image, (x, 0), (x, self.height), (0, 255, 0, 255), 1)
        
        # 绘制单元格编号
        for row in range(grid_info['rows']):
            for col in range(grid_info['cols']):
                y_center = (grid_info['row_starts'][row] + grid_info['row_ends'][row]) // 2
                x_center = (grid_info['col_starts'][col] + grid_info['col_ends'][col]) // 2
                
                frame_number = row * grid_info['cols'] + col + 1
                cv2.putText(viz_image, str(frame_number), (x_center-10, y_center+5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0, 255), 1)
        
        # 显示网格信息
        info_text = f"{grid_info['rows']}x{grid_info['cols']} 网格 ({grid_info['cols']*grid_info['rows']} 帧)"
        cv2.putText(viz_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255, 255), 2)
        
        # 转换为RGB显示
        if viz_image.shape[2] == 4:
            viz_rgb = cv2.cvtColor(viz_image, cv2.COLOR_BGRA2RGBA)
        else:
            viz_rgb = cv2.cvtColor(viz_image, cv2.COLOR_BGR2RGB)
            
        plt.figure(figsize=(10, 10))
        plt.imshow(viz_rgb)
        plt.title(f"精灵图网格检测: {info_text}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def slice_by_dimensions(self, rows, cols):
        """
        手动指定行列数进行切片
        
        参数:
            rows (int): 行数
            cols (int): 列数
        """
        # 计算单元格尺寸
        cell_width = self.width // cols
        cell_height = self.height // rows
        
        # 构建网格信息
        grid_info = {
            'rows': rows,
            'cols': cols,
            'avg_cell_width': cell_width,
            'avg_cell_height': cell_height,
            'row_starts': [i * cell_height for i in range(rows)],
            'row_ends': [(i+1) * cell_height - 1 for i in range(rows)],
            'col_starts': [i * cell_width for i in range(cols)],
            'col_ends': [(i+1) * cell_width - 1 for i in range(cols)]
        }
        
        return self.slice_sprites(grid_info)


class UniformGridSpriteSheetSlicer:
    """针对UV尺寸统一的规则网格精灵图的专用切片器"""
    
    def __init__(self, image_path, output_dir=None):
        """
        初始化精灵图切片器
        
        参数:
            image_path (str): 精灵图路径
            output_dir (str, optional): 输出目录，如不指定则使用图像所在目录
        """
        self.image_path = image_path
        
        # 读取图像
        self.image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if self.image is None:
            raise ValueError(f"无法读取图像: {image_path}")
            
        self.height, self.width = self.image.shape[:2]
        
        # 设置输出目录
        if output_dir is None:
            self.output_dir = os.path.dirname(os.path.abspath(image_path))
        else:
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)
            
        self.file_prefix = os.path.splitext(os.path.basename(image_path))[0]
        
        print(f"加载精灵图 {os.path.basename(image_path)}: {self.width}x{self.height}像素")
    
    def slice_uniform_grid(self, frame_width, frame_height, horizontal_gap=0, vertical_gap=0, visualize=True):
        """
        按照固定帧大小和间距切分精灵图
        
        参数:
            frame_width (int): 每帧宽度
            frame_height (int): 每帧高度
            horizontal_gap (int): 水平间隔
            vertical_gap (int): 垂直间隔
            visualize (bool): 是否可视化结果
            
        返回:
            list: 切片后的精灵帧信息列表
        """
        # 计算可容纳的行列数
        total_frame_width = frame_width + horizontal_gap
        total_frame_height = frame_height + vertical_gap
        
        cols = (self.width + horizontal_gap) // total_frame_width
        rows = (self.height + vertical_gap) // total_frame_height
        
        # 确保估算合理
        if cols == 0 or rows == 0:
            print(f"警告: 指定的帧尺寸 {frame_width}x{frame_height} 太大，无法切分")
            return []
            
        print(f"按 {rows}行 x {cols}列 的网格切分，每帧 {frame_width}x{frame_height} 像素")
        
        sprites = []
        
        # 为每个帧计算UV坐标和切分位置
        for row in range(rows):
            for col in range(cols):
                # 计算在图像中的位置
                x_start = col * total_frame_width
                y_start = row * total_frame_height
                
                # 确保不超出图像边界
                if x_start >= self.width or y_start >= self.height:
                    continue
                
                x_end = min(x_start + frame_width, self.width)
                y_end = min(y_start + frame_height, self.height)
                
                # 确保有足够的尺寸
                if x_end - x_start < 2 or y_end - y_start < 2:
                    continue
                
                # 切分图像
                try:
                    sprite = self.image[y_start:y_end, x_start:x_end]
                    
                    # 检查切分的有效性
                    if sprite.size == 0:
                        continue
                    
                    # 计算UV坐标 (归一化到0-1范围)
                    u1 = x_start / self.width
                    v1 = y_start / self.height
                    u2 = x_end / self.width
                    v2 = y_end / self.height
                    
                    # 保存精灵
                    frame_number = row * cols + col + 1
                    output_path = os.path.join(self.output_dir, f"{self.file_prefix}_frame{frame_number:03d}.png")
                    
                    # 转换颜色格式并保存
                    if sprite.shape[2] == 4:
                        sprite_rgb = cv2.cvtColor(sprite, cv2.COLOR_BGRA2RGBA)
                    else:
                        sprite_rgb = cv2.cvtColor(sprite, cv2.COLOR_BGR2RGB)
                        
                    pil_img = Image.fromarray(sprite_rgb)
                    pil_img.save(output_path, "PNG")
                    
                    sprites.append({
                        'image': sprite,
                        'path': output_path,
                        'position': (x_start, y_start, x_end - x_start, y_end - y_start),
                        'frame': frame_number,
                        'uv': (u1, v1, u2, v2)
                    })
                    
                    print(f"保存精灵帧 {frame_number}: {output_path} (UV: {u1:.3f},{v1:.3f} - {u2:.3f},{v2:.3f})")
                except Exception as e:
                    print(f"处理第 {row} 行 {col} 列的帧时出错: {e}")
        
        if visualize and sprites:
            self.visualize_grid(frame_width, frame_height, horizontal_gap, vertical_gap, rows, cols)
            
        return sprites
    
    def auto_detect_frame_size(self, min_size=8, max_size=256):
        """
        尝试自动检测帧大小
        
        参数:
            min_size (int): 最小可能的帧尺寸
            max_size (int): 最大可能的帧尺寸
            
        返回:
            tuple: (帧宽度, 帧高度, 水平间隔, 垂直间隔)
        """
        print("尝试自动检测帧大小...")
        
        # 准备Alpha通道或灰度图像用于边界检测
        if self.image.shape[2] == 4:
            mask = self.image[:, :, 3]  # Alpha通道
        else:
            # 转为灰度并二值化
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            
        # 为了提高边界检测的准确性，可以应用图像处理
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        
        # 计算水平和垂直投影
        h_projection = np.sum(mask, axis=1)
        v_projection = np.sum(mask, axis=0)
        
        # 检测行边界
        row_edges = self._find_edges(h_projection, threshold=np.max(h_projection) * 0.05)
        col_edges = self._find_edges(v_projection, threshold=np.max(v_projection) * 0.05)
        
        # 如果无法检测到足够的边界，回退到简单的除法估算
        if len(row_edges) < 2 or len(col_edges) < 2:
            common_sizes = [16, 32, 48, 64, 96, 128]  # 常见的精灵尺寸
            
            # 尝试找到合适的尺寸
            best_size = None
            for size in common_sizes:
                if self.width % size == 0 and self.height % size == 0:
                    best_size = size
                    break
            
            if best_size is None:
                best_size = min(self.width // 4, self.height // 4)  # 默认估计
                
            return best_size, best_size, 0, 0
        
        # 分析连续的行和列来确定帧大小和间隔
        row_sizes = [row_edges[i+1] - row_edges[i] for i in range(0, len(row_edges)-1, 2)]
        col_sizes = [col_edges[i+1] - col_edges[i] for i in range(0, len(col_edges)-1, 2)]
        
        # 计算常见尺寸（取众数）
        frame_height = self._most_common(row_sizes) if row_sizes else min(self.height, max_size)
        frame_width = self._most_common(col_sizes) if col_sizes else min(self.width, max_size)
        
        # 检测间隔
        row_gaps = [row_edges[i+1] - row_edges[i-1] for i in range(1, len(row_edges)-1, 2)] if len(row_edges) > 2 else [0]
        col_gaps = [col_edges[i+1] - col_edges[i-1] for i in range(1, len(col_edges)-1, 2)] if len(col_edges) > 2 else [0]
        
        vertical_gap = self._most_common(row_gaps) if row_gaps else 0
        horizontal_gap = self._most_common(col_gaps) if col_gaps else 0
        
        # 确保在合理范围内
        frame_width = max(min_size, min(max_size, frame_width))
        frame_height = max(min_size, min(max_size, frame_height))
        
        print(f"检测到帧尺寸: {frame_width}x{frame_height}，间隔: 水平={horizontal_gap}, 垂直={vertical_gap}")
        return frame_width, frame_height, horizontal_gap, vertical_gap
    
    def _find_edges(self, projection, threshold=0):
        """找出投影中的边缘位置"""
        edges = []
        is_content = False
        
        for i, value in enumerate(projection):
            if not is_content and value > threshold:
                edges.append(i)  # 内容开始
                is_content = True
            elif is_content and value <= threshold:
                edges.append(i-1)  # 内容结束
                is_content = False
                
        # 如果图像以内容结束但没有闭合
        if is_content:
            edges.append(len(projection) - 1)
            
        return edges
    
    def _most_common(self, values, tolerance=3):
        """找出列表中最常见的值（允许一定的误差）"""
        if not values:
            return 0
            
        # 对接近的值进行聚类
        clusters = []
        for value in sorted(values):
            # 查找匹配的聚类
            found = False
            for i, cluster in enumerate(clusters):
                if abs(cluster['value'] - value) <= tolerance:
                    cluster['count'] += 1
                    cluster['sum'] += value
                    found = True
                    break
                    
            if not found:
                clusters.append({'value': value, 'count': 1, 'sum': value})
        
        # 找出最常见的聚类
        most_common_cluster = max(clusters, key=lambda x: x['count'])
        
        # 返回聚类平均值
        return round(most_common_cluster['sum'] / most_common_cluster['count'])
    
    def generate_spritesheet_data(self, sprites):
        """
        生成精灵图数据导出格式
        
        参数:
            sprites (list): 切片精灵列表
            
        返回:
            dict: 精灵图数据
        """
        frames = {}
        meta = {
            "image": os.path.basename(self.image_path),
            "size": {"w": self.width, "h": self.height},
            "scale": 1,
            "format": "RGBA8888" if self.image.shape[2] == 4 else "RGB888"
        }
        
        for sprite in sprites:
            frame_id = f"frame{sprite['frame']:03d}"
            x, y, w, h = sprite['position']
            
            frames[frame_id] = {
                "frame": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "spriteSourceSize": {"x": 0, "y": 0, "w": int(w), "h": int(h)},
                "sourceSize": {"w": int(w), "h": int(h)},
                "rotated": False,
                "trimmed": False,
                "pivot": {"x": 0.5, "y": 0.5},
                "uv": list(sprite['uv'])  # 添加UV坐标
            }
        
        return {
            "frames": frames,
            "meta": meta
        }
    
    def visualize_grid(self, frame_width, frame_height, h_gap, v_gap, rows, cols):
        """可视化网格切分"""
        viz_image = self.image.copy()
        
        # 计算总帧大小（包含间隙）
        total_frame_width = frame_width + h_gap
        total_frame_height = frame_height + v_gap
        
        # 绘制网格线
        for row in range(rows + 1):
            y = min(row * total_frame_height, self.height - 1)
            cv2.line(viz_image, (0, y), (self.width, y), (0, 255, 0, 255), 1)
            
        for col in range(cols + 1):
            x = min(col * total_frame_width, self.width - 1)
            cv2.line(viz_image, (x, 0), (x, self.height), (0, 255, 0, 255), 1)
            
        # 绘制每帧的编号
        for row in range(rows):
            for col in range(cols):
                frame_number = row * cols + col + 1
                x = col * total_frame_width + 5
                y = row * total_frame_height + 15
                
                if x < self.width and y < self.height:
                    cv2.putText(viz_image, str(frame_number), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255, 255), 1)
        
        # 保存和显示可视化结果
        viz_path = os.path.join(self.output_dir, f"{self.file_prefix}_grid_viz.png")
        
        # 转换颜色空间以确保正确保存
        if viz_image.shape[2] == 4:
            cv2.imwrite(viz_path, cv2.cvtColor(viz_image, cv2.COLOR_RGBA2BGRA))
        else:
            cv2.imwrite(viz_path, viz_image)
            
        print(f"网格可视化已保存到: {viz_path}")
        
        # 创建窗口并显示图像
        try:
            cv2.imshow("精灵图切分网格", viz_image)
            cv2.waitKey(1)  # 显示但不阻塞
        except:
            print("注意: 无法显示可视化窗口，但已保存图像")
    
    def export_json(self, sprites, output_path=None):
        """
        导出精灵数据到JSON文件
        
        参数:
            sprites (list): 精灵列表
            output_path (str, optional): 输出路径，默认为精灵图同目录
        
        返回:
            str: JSON文件路径
        """
        if not sprites:
            print("警告: 没有精灵可导出")
            return None
            
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"{self.file_prefix}_frames.json")
            
        data = self.generate_spritesheet_data(sprites)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"精灵数据已导出到: {output_path}")
        return output_path
    
    def save_animation_preview(self, sprites, fps=10, output_path=None):
        """
        创建GIF动画预览
        
        参数:
            sprites (list): 精灵列表
            fps (int): 帧率
            output_path (str, optional): 输出路径
            
        返回:
            str: GIF文件路径
        """
        if not sprites:
            print("警告: 没有精灵可预览")
            return None
            
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"{self.file_prefix}_preview.gif")
            
        # 将OpenCV图像转换为PIL格式
        pil_frames = []
        
        # 按帧号排序
        sorted_sprites = sorted(sprites, key=lambda s: s['frame'])
        
        for sprite in sorted_sprites:
            if sprite['image'].shape[2] == 4:
                pil_img = Image.fromarray(cv2.cvtColor(sprite['image'], cv2.COLOR_BGRA2RGBA))
            else:
                pil_img = Image.fromarray(cv2.cvtColor(sprite['image'], cv2.COLOR_BGR2RGB))
                
            pil_frames.append(pil_img)
            
        # 创建GIF动画
        if pil_frames:
            duration = 1000 // fps  # 毫秒
            pil_frames[0].save(output_path, format='GIF', append_images=pil_frames[1:],
                             save_all=True, duration=duration, loop=0)
            print(f"动画预览已保存到: {output_path}")
            return output_path
        
        return None


# 使用示例
if __name__ == "__main__":
    # 替换为您的精灵图路径
    sprite_sheet_path = "./assets/input/living_animals_generator_1.png"
    # sprite_sheet_path = "./assets/input/sfx/gasoline_anim.png"
    # sprite_sheet_path = "./assets/input/sfx/generating_power_sfx.png"
    
    # # 测试情况
    # # generating_power_sfx 通过，位置略微错乱
    # # gasoline_anim 失败，切了4帧空白
    # # living_animals_generator_1 成功大半，是依据形状来切分，而非等距切分
    # slicer = SpriteSheetSlicer(sprite_sheet_path)
    # # 自动切片
    # sprites = slicer.auto_slice(alpha_threshold=10)
    # # 可视化切片效果
    # slicer.visualize_detection(sprites)
    # # # 如果自动切片效果不好，可以手动指定网格
    # # # slicer.slice_by_grid(rows=4, cols=4)
    
    # # 测试情况
    # # generating_power_sfx 失败，切的非常细
    # # gasoline_anim 失败，切的非常细
    # # living_animals_generator_1 失败，切分的不对
    # slicer = GridSpriteSheetSlicer(sprite_sheet_path)
    # # 自动检测网格并切片
    # sprites = slicer.slice_sprites()

    # # 测试情况
    # # generating_power_sfx 失败，切的非常细
    # # gasoline_anim 完美成功
    # # living_animals_generator_1 失败，多切了一部分帧数
    # 创建切片器
    slicer = UniformGridSpriteSheetSlicer(sprite_sheet_path, "output_sprites")
    # 方法1: 自动检测帧大小并切分
    frame_width, frame_height, h_gap, v_gap = slicer.auto_detect_frame_size()
    sprites = slicer.slice_uniform_grid(
        frame_width=frame_width,
        frame_height=frame_height,
        horizontal_gap=h_gap,
        vertical_gap=v_gap,
        visualize=True
    )
    # # 导出精灵数据到JSON
    # json_path = slicer.export_json(sprites)
    # 创建动画预览
    gif_path = slicer.save_animation_preview(sprites, fps=10)



