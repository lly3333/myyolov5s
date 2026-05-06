#!/usr/bin/env python3
"""
VisDrone2019-DET 数据集转换为 YOLO 格式
严格遵循 Ultralytics 官方 YAML 中的类别定义

类别映射 (VisDrone原始类别 -> YOLO类别):
    1: pedestrian      -> 0
    2: people          -> 1
    3: bicycle         -> 2
    4: car             -> 3
    5: van             -> 4
    6: truck           -> 5
    7: tricycle        -> 6
    8: awning-tricycle -> 7
    9: bus             -> 8
    10: motor          -> 9
    0: ignored regions -> 跳过
"""

import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def convert_box(size, box):
    """
    将 VisDrone 的 [x, y, w, h] 转换为 YOLO 的归一化 [x_center, y_center, w, h]
    
    Args:
        size: (width, height) 图像尺寸
        box: (x, y, w, h) VisDrone 格式边界框
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_center = (box[0] + box[2] / 2.0) * dw
    y_center = (box[1] + box[3] / 2.0) * dh
    w = box[2] * dw
    h = box[3] * dh
    return x_center, y_center, w, h


def visdrone2yolo(dir_path):
    """
    将单个 VisDrone 子集的标注转换为 YOLO 格式
    
    Args:
        dir_path: Path 对象，指向如 VisDrone2019-DET-train 的目录
                  目录结构应为:
                  dir_path/
                  ├── images/       # .jpg 图像文件
                  └── annotations/  # .txt 标注文件 (VisDrone格式)
                  └── labels/       # 生成的 YOLO 格式标注
    """
    dir_path = Path(dir_path)
    
    # 创建 labels 目录
    labels_dir = dir_path / 'labels'
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有标注文件
    annotation_files = list((dir_path / 'annotations').glob('*.txt'))
    
    if not annotation_files:
        print(f"警告: 在 {dir_path / 'annotations'} 中未找到 .txt 标注文件")
        return
    
    pbar = tqdm(annotation_files, desc=f'Converting {dir_path.name}')
    
    for ann_file in pbar:
        # 对应的图像路径
        img_path = (dir_path / 'images' / ann_file.name).with_suffix('.jpg')
        
        # 检查图像是否存在
        if not img_path.exists():
            print(f"警告: 图像不存在 {img_path}")
            continue
        
        # 获取图像尺寸
        try:
            img_size = Image.open(img_path).size  # (width, height)
        except Exception as e:
            print(f"错误: 无法打开图像 {img_path}: {e}")
            continue
        
        # 读取 VisDrone 标注并转换
        lines = []
        try:
            with open(ann_file, 'r', encoding='utf-8') as f:
                for line in f.read().strip().splitlines():
                    if not line:
                        continue
                    
                    row = line.split(',')
                    
                    # 跳过 ignored regions (类别 0)
                    if row[4] == '0':
                        continue
                    
                    # 类别映射: VisDrone类别(1-10) -> YOLO类别(0-9)
                    cls = int(row[5]) - 1
                    
                    # 确保类别在有效范围内 [0, 9]
                    if not (0 <= cls <= 9):
                        print(f"警告: 非法类别 {cls+1} 在文件 {ann_file}")
                        continue
                    
                    # 转换边界框
                    box = tuple(map(int, row[:4]))
                    yolo_box = convert_box(img_size, box)
                    
                    # 格式化输出 (保留6位小数)
                    box_str = ' '.join(f'{x:.6f}' for x in yolo_box)
                    lines.append(f"{cls} {box_str}\n")
        except Exception as e:
            print(f"错误: 读取标注文件 {ann_file} 失败: {e}")
            continue
        
        # 写入 YOLO 格式标注文件
        label_file = labels_dir / ann_file.name
        with open(label_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)


def main():
    """
    主函数: 批量转换 train/val/test-dev 三个子集
    请根据实际路径修改 DATASET_ROOT
    """
    # ============================================
    # 修改这里: 设置你的数据集根目录路径
    # ============================================
    # 方式1: 使用相对路径 (相对于脚本位置)
    # DATASET_ROOT = Path(__file__).parent / 'datasets' / 'VisDrone'
    
    # 方式2: 使用绝对路径 (推荐)
    DATASET_ROOT = Path(r'D:\graduation_project\datasets\VisDrone')  # <-- 修改为你的实际路径
    
    # 或者从命令行参数读取
    import sys
    if len(sys.argv) > 1:
        DATASET_ROOT = Path(sys.argv[1])
    
    # 需要转换的子集
    subsets = [
        'VisDrone2019-DET-train',
        'VisDrone2019-DET-val', 
        'VisDrone2019-DET-test-dev',
        # 'VisDrone2019-DET-test-challenge',  # 如需转换取消注释
    ]
    
    print(f"数据集根目录: {DATASET_ROOT.absolute()}")
    print("-" * 50)
    
    for subset in subsets:
        subset_path = DATASET_ROOT / subset
        if subset_path.exists():
            print(f"\n开始转换: {subset}")
            visdrone2yolo(subset_path)
            print(f"完成: {subset}")
        else:
            print(f"跳过: 目录不存在 {subset_path}")
    
    print("\n" + "=" * 50)
    print("所有转换完成!")
    print(f"YOLO 格式标注已保存到各子集的 labels/ 目录中")


if __name__ == '__main__':
    main()