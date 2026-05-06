import os
import cv2
from pathlib import Path
from tqdm import tqdm

def convert_visdrone_to_yolo(data_dir):
    """
    转换VisDrone格式标注为YOLO格式（原地转换）
    输入: labels/train/xxx.txt (VisDrone格式)
    输出: labels/train/xxx.txt (YOLO格式，覆盖原文件)
    """
    
    class_map = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9, 11:10}
    splits = ['train', 'val', 'test']
    
    for split in splits:
        print(f"\n处理 {split}...")
        
        img_dir = Path(data_dir) / 'images' / split
        label_dir = Path(data_dir) / 'labels' / split
        
        if not label_dir.exists():
            print(f"  跳过: 未找到 {label_dir}")
            continue
        
        anno_files = list(label_dir.glob('*.txt'))
        total_objects = 0
        
        for anno_file in tqdm(anno_files, desc=f"  {split}"):
            # 查找对应图像
            img_file = None
            for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
                candidate = img_dir / (anno_file.stem + ext)
                if candidate.exists():
                    img_file = candidate
                    break
            
            if img_file is None:
                print(f"  警告: 未找到图像 {anno_file.stem}")
                continue
            
            # 读取图像尺寸
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            img_h, img_w = img.shape[:2]
            
            # 转换标注
            yolo_lines = []
            with open(anno_file, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    parts = line.strip().split(',')
                    if len(parts) < 8:
                        continue
                    
                    x, y, w, h = map(int, parts[0:4])
                    category = int(parts[5])
                    
                    if category == 0 or category not in class_map:
                        continue
                    if w <= 0 or h <= 0:
                        continue
                    
                    # 归一化
                    x_c = (x + w/2) / img_w
                    y_c = (y + h/2) / img_h
                    w_n = w / img_w
                    h_n = h / img_h
                    
                    # 裁剪
                    x_c = max(0.0, min(1.0, x_c))
                    y_c = max(0.0, min(1.0, y_c))
                    w_n = max(0.0, min(1.0, w_n))
                    h_n = max(0.0, min(1.0, h_n))
                    
                    yolo_lines.append(f"{class_map[category]} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")
                    total_objects += 1
            
            # 覆盖写入YOLO格式
            with open(anno_file, 'w', encoding='utf-8') as f:
                f.writelines(yolo_lines)
        
        print(f"  完成: {len(anno_files)}张图像, {total_objects}个目标")
    
    print(f"\n✅ 全部转换完成!")

if __name__ == '__main__':
    # 你的实际路径
    data_dir = r'D:\graduation_project\yolov5\data\VisDrone'
    convert_visdrone_to_yolo(data_dir)