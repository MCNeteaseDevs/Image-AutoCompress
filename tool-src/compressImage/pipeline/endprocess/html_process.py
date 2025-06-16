from typing import Dict, List
from typing import Dict

from utils.report import CompressionReport
from defines.compression_result import CompressionResult
from pipeline.abstract_comps import PipelineEndProcess


class HTMLEndProcess(PipelineEndProcess):
    """保存到html 处理器"""
    
    def do(self, last_results: Dict, params: Dict = None) -> Dict:
        """
        存储图片到本地

        Args:
            last_results: 上一个的处理结果
            params: 额外参数
            
        Returns:
            综合处理结果的字典
        """
        results: List[CompressionResult] = last_results["results"]
        if results:
            input_path = last_results["input_path"]
            output_path = last_results["output_path"]
            # 封装html数据，并保存到本地
            report = CompressionReport(input_path, output_path)
            for result in results:
                report.add_result(result)
            report.save_html_report()
            last_results["save_html"] = True
        return last_results

