import os
import json
import logging
import time
from typing import List
from defines.compression_result import CompressionResult


class CompressionReport:
    """生成压缩报告的类"""
    
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir  # 用于输出相对路径
        if os.path.isfile(output_dir):
            self.output_dir = os.path.dirname(output_dir)
        else:
            self.output_dir = output_dir
        self.compression_results: List[CompressionResult] = []
        self.compression_fail_results: List[CompressionResult] = []
        # 后续将修改存储路径，可通过gui一键打开html
        # 文件名，增加路径前缀、年月日时分秒日期
        current_time = time.strftime("%Y%m%d_%H%M%S")  # 格式：年月日_时分秒
        file_name = f"{os.path.basename(self.output_dir)}-{current_time}"
        self.report_path = os.path.join(self.output_dir, f"{file_name}.json")
        self.html_report_path = os.path.join(self.output_dir, f"{file_name}.html")
    
    def add_result(self, result: CompressionResult):
        """添加单个压缩结果"""
        if result.is_compressed:
            # 压缩完成的信息
            self.compression_results.append(result)
        else:
            # 压缩失败的信息
            self.compression_fail_results.append(result)
    
    def save_json_report(self):
        """保存JSON格式的报告"""
        # total_original = sum(r["original_size_kb"] for r in self.compression_results)
        # total_compressed = sum(r["compressed_size_kb"] for r in self.compression_results)
        # total_reduction = total_original - total_compressed
        # total_percent = (total_reduction / total_original) * 100 if total_original > 0 else 0
        
        # report = {
        #     "summary": {
        #         "total_images": len(self.compression_results),
        #         "total_original_size_kb": total_original,
        #         "total_compressed_size_kb": total_compressed,
        #         "total_reduction_kb": total_reduction,
        #         "average_reduction_percent": total_percent
        #     },
        #     "details": self.compression_results
        # }
        
        # with open(self.report_path, 'w') as f:
        #     json.dump(report, f, indent=2)
        
        return self.report_path
    
    def _get_image_type_str(self, result: CompressionResult):
        # 图像类型
        img_type = ""
        if result.is_sfx_image:
            img_type = "序列帧"
        elif result.is_text_image:
            img_type = "文字图"
        return img_type

    def save_html_report(self):
        """生成HTML格式的可视化报告"""
        if not self.compression_results and not self.compression_fail_results:
            return None
        
        # 计算汇总数据
        total_original = sum(r.original_size / 1024 for r in self.compression_results)
        total_compressed = sum(r.compressed_size / 1024 for r in self.compression_results)
        total_reduction = total_original - total_compressed
        total_percent = (total_reduction / total_original) * 100 if total_original > 0 else 0
        # 分辨率数据
        total_original_px = sum(r.original_width * r.original_height for r in self.compression_results)
        total_compressed_px = sum(r.compressed_width * r.compressed_height for r in self.compression_results)
        total_reduction_px = total_original_px - total_compressed_px
        total_percent_px = (total_reduction_px / total_original_px) * 100 if total_original > 0 else 0
        # 使用perceptual计算平均质量
        avg_quality = sum(r.perceptual_score for r in self.compression_results) / len(self.compression_results) if len(self.compression_results) > 0 else 0
        
        success_len = len(self.compression_results)
        fail_len = len(self.compression_fail_results)

        # 生成HTML报告
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>贴图分辨率压缩报告</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #4CAF50; color: white; padding: 10px; }
                .summary { background-color: #f2f2f2; padding: 15px; margin: 10px 0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .chart-container { width: 100%; height: 400px; }
            </style>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <div class="header">
                <h1>贴图分辨率压缩报告</h1>
            </div>
            
            <div class="summary">
                <h2>压缩摘要</h2>
                <p>总图像数: """ + str(success_len + fail_len) + """</p>
                <p>压缩完成数: """ + str(success_len) + """</p>
                <p>不处理数: """ + str(fail_len) + """</p>

                <p>原始总分辨率值(width*height): """ + f"{total_original_px:.2f} px" + """</p>
                <p>压缩后总分辨率值(width*height): """ + f"{total_compressed_px:.2f} px" + """</p>
                <p>总分辨率压缩率: """ + f"{total_reduction_px:.2f} px ({total_percent_px:.1f}%)" + """</p>

                <p>原始总大小: """ + f"{total_original:.2f} KB" + """</p>
                <p>压缩后总大小: """ + f"{total_compressed:.2f} KB" + """</p>
                <p>总节省空间: """ + f"{total_reduction:.2f} KB ({total_percent:.1f}%)" + """</p>

                <p>平均感知质量: """ + f"{avg_quality:.4f}" + """</p>
            </div>
            
            <h2>详细压缩结果</h2>
            <h3>如果图片是序列帧，需人工修改引用序列帧的代码、json，因为分辨率发生了改变，相关代码需要同步修改！</h3>
            <table>
                <tr>
                    <th>图像</th>
                    <th>图像类型（疑似）</th>
                    <th>自动修改的文件（需人工检查、确认）</th>
                    <th>原始分辨率</th>
                    <th>压缩分辨率</th>
                    <th>分辨率减少</th>
                    <th>原始大小 (KB)</th>
                    <th>压缩大小 (KB)</th>
                    <th>大小减少 (%)</th>
                    <th>感知评分</th>
                    <th>SSIM</th>
                    <th>PSNR</th>
                </tr>
        """
        
        # 添加每个图像的结果行
        for r in self.compression_results:
            imgPath = os.path.relpath(r.original_path, self.input_dir)
            # 分辨率降低比例
            resolution_reduction = 100 - (r.compressed_width * r.compressed_height) / (r.original_width * r.original_height) * 100
            reduction_percent = (100 - (r.compressed_size / r.original_size) * 100) if r.original_size > 0 else 0
            # 文件修改记录
            file_update_str = ""
            if r.update_json_record:
                file_update_str = ", ".join([file for file in r.update_json_record])
            html += f"""
                <tr>
                    <td>{imgPath}</td>
                    <td>{self._get_image_type_str(r)}</td>
                    <td>{file_update_str}</td>
                    <td>{r.original_width}x{r.original_height}</td>
                    <td>{r.compressed_width}x{r.compressed_height}</td>
                    <td>{resolution_reduction:.1f}%</td>
                    <td>{(r.original_size / 1024):.2f}</td>
                    <td>{(r.compressed_size / 1024):.2f}</td>
                    <td>{reduction_percent:.1f}%</td>
                    <td>{r.perceptual_score:.4f}</td>
                    <td>{r.ssim_score:.4f}</td>
                    <td>{r.psnr_score:.2f}</td>
                </tr>
            """
        
        # 添加失败信息
        if self.compression_fail_results:
            html += f"""
                </table>
                <h2></h2>
                <h2>不处理图片详情</h2>
                <table>
                    <tr>
                        <th>图像</th>
                        <th>图像类型（疑似）</th>
                        <th>分辨率</th>
                        <th>失败原因</th>
                        <th>压缩后分辨率</th>
                        <th>感知评分</th>
                    </tr>
            """
            
            # 添加每个图像的结果行
            for r in self.compression_fail_results:
                imgPath = os.path.relpath(r.original_path, self.input_dir)
                html += f"""
                    <tr>
                        <td>{imgPath}</td>
                        <td>{self._get_image_type_str(r)}</td>
                        <td>{r.original_width}x{r.original_height}</td>
                        <td>{r.info}</td>
                        <td>{r.compressed_width}x{r.compressed_height}</td>
                        <td>{r.perceptual_score}</td>
                    </tr>
                """

        # 添加图表和关闭HTML - 重点展示分辨率变化而非文件大小
        html += """
            </table>
        </body>
        </html>
        """
        # 不显示图表
        # <h2>分辨率变化效果图表</h2>
        # <div class="chart-container">
        #     <canvas id="resolutionChart"></canvas>
        # </div>
        # <div class="chart-container">
        #     <canvas id="qualityChart"></canvas>
        # </div>
        # <script>
        #     // 分辨率比较图表
        #     const resCtx = document.getElementById('resolutionChart').getContext('2d');
        #     const resChart = new Chart(resCtx, {
        #         type: 'bar',
        #         data: {
        #             labels: [""" + ", ".join([f"'{os.path.basename(r['original_path'])}'" for r in self.compression_results]) + """],
        #             datasets: [{
        #                 label: '原始分辨率 (像素数)',
        #                 data: [""" + ", ".join([f"{int(r['original_resolution'].split('x')[0]) * int(r['original_resolution'].split('x')[1])}" for r in self.compression_results]) + """],
        #                 backgroundColor: 'rgba(54, 162, 235, 0.5)',
        #                 borderColor: 'rgb(54, 162, 235)',
        #                 borderWidth: 1
        #             }, {
        #                 label: '压缩分辨率 (像素数)',
        #                 data: [""" + ", ".join([f"{int(r['compressed_resolution'].split('x')[0]) * int(r['compressed_resolution'].split('x')[1])}" for r in self.compression_results]) + """],
        #                 backgroundColor: 'rgba(75, 192, 192, 0.5)',
        #                 borderColor: 'rgb(75, 192, 192)',
        #                 borderWidth: 1
        #             }]
        #         },
        #         options: {
        #             responsive: true,
        #             plugins: {
        #                 title: {
        #                     display: true,
        #                     text: '压缩前后分辨率对比'
        #                 },
        #             },
        #             scales: {
        #                 y: {
        #                     beginAtZero: true,
        #                     title: {
        #                         display: true,
        #                         text: '像素总数'
        #                     }
        #                 }
        #             }
        #         }
        #     });
        # </script>
        
        with open(self.html_report_path, 'w') as f:
            f.write(html)
            print(f"\nHTML报告已生成: {self.html_report_path}")
        
        return self.html_report_path

