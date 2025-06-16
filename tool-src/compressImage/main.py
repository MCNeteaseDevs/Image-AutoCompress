import os
import sys
import time
import argparse
import multiprocessing
import logging

from pipeline.gameassets_compressor import GameAssetsCompressor
from utils.read_config import load_config_from_json


# region 主函数
def main():
    """主函数，处理命令行参数并启动压缩系统"""
    parser = argparse.ArgumentParser(description='游戏资产图像智能压缩系统')
    parser.add_argument('input_dir', type=str, help='输入目录，包含原始图像')
    parser.add_argument('output_dir', type=str, help='输出目录，保存压缩后的图像')
    parser.add_argument('--res_dir', type=str, default="", help='资源包根目录')
    parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count(), help='并行工作线程数')
    parser.add_argument('--report', type=bool, default=True, help='生成详细压缩报告')
    
    # 判断是否有传递参数
    if len(sys.argv) == 1:
        print("该工具可以自动将资源包内的贴图进行降低分辨率处理，当处理完成后，需人工检查一遍处理过的贴图，是否存在太糊、序列帧播放异常等情况。\n")
        print("在运行工具前，希望先做好备份！\n\n")
        # 没有参数时，进行交互式输入
        input_dir = ""
        while not input_dir:
            print("请输入需要压缩图片的资源包目录：")
            input_dir = input().strip()
            # 校验是否是完整路径
            if not os.path.isabs(input_dir):
                print("\n错误：请输入完整的绝对路径！\n")
                input_dir = ""

        output_dir = ""
        while not output_dir:
            print("\n请输入图片的输出目录（输出目录必须是空文件夹，可以是未创建的文件夹，会自动创建）：")
            output_dir = input().strip()
            # 校验是否是完整路径
            if not os.path.isabs(output_dir):
                print("\n错误：请输入完整的绝对路径！")
                output_dir = ""
            elif os.path.isdir(output_dir):
                # 路径存在，则判断路径下是否是空文件夹
                if os.listdir(output_dir):
                    print("\n错误：输出目录不为空，请选择空文件夹或不存在的目录(会自动创建)！")
                    output_dir = ""
        
        # 手动设置参数
        args = parser.parse_args([input_dir, output_dir])
    else:
        # 正常解析命令行参数
        args = parser.parse_args()
    
    # 初始化读取settings.json文件
    load_config_from_json()

    # 创建压缩系统实例
    compressor = GameAssetsCompressor(args.workers, args.report)
    
    # 开始批处理压缩
    print(f"\n开始处理图像: {args.input_dir} -> {args.output_dir}\n")
    start_time = time.time()

    # 对资源根目录做自动化处理
    input_dir = args.input_dir
    res_dir = args.res_dir
    if not res_dir:
        # 根据input_dir进行计算: 如果有textures，则截断到textures前面的路径；
        if "textures" in input_dir:
            res_dir = input_dir.split("textures")[0]
            res_dir = os.path.normpath(res_dir)
        else:
            # 拼接上textures，看是否存在该路径
            new_dir = os.path.join(input_dir, "textures")
            if os.path.exists(new_dir):
                res_dir = input_dir
            else:
                # 获取input_dir下的所有子目录，逐个遍历子目录，判断子目录下，是否有textures目录
                found = False
                for root, dirs, files in os.walk(input_dir):
                    for dir_name in dirs:
                        potential_texture_dir = os.path.join(root, dir_name, "textures")
                        if os.path.exists(potential_texture_dir):
                            res_dir = os.path.join(root, dir_name)
                            found = True
                            break
                    if found:
                        break
    
    if not res_dir:
        print("资源包根目录无法自动识别，请手动输入！！")
        while not res_dir:
            res_dir = input().strip()
            # 校验是否是完整路径
            if not os.path.isabs(res_dir):
                print("\n错误：请输入完整的绝对路径！")
                res_dir = ""
    
    output_dir = args.output_dir
    results = compressor.batch_process(
        input_dir,
        output_dir,
        res_dir
    )
    
    # 如果没有结果，提前退出
    if not results:
        logging.warning("没有找到需要处理的图像。")
        return
    
    # 打印总结
    successful = sum(1 for r in results if r.is_compressed)
    failed = sum(1 for r in results if not r.is_compressed)
    
    print(f"\n压缩处理完成:")
    print(f"- 共处理图像: {len(results)} 张")
    print(f"- 已压缩: {successful} 张")
    print(f"- 不作压缩: {failed} 张")
    print(f"- 总耗时: {time.time() - start_time:.1f} 秒")

    print(f"\n输出目录：{output_dir}，请人工检查一遍贴图，并手动将贴图覆盖掉资源包里的贴图。\n")

    # 自动打开输出目录的资源管理器
    if os.path.exists(output_dir):
        os.startfile(output_dir)
    # 增加窗口暂停
    os.system("pause")

if __name__ == '__main__':
    main()
# endregion



