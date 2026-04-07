import os
import numpy as np
import tifffile
from glob import glob
from tqdm import tqdm


    
    # 重叠裁剪
    # 共 36 个 384x384 的图块，并按顺序保存到一个新的目录中。
    

# ===================================================================
# 1. 配置您的路径和参数
# ===================================================================

# --- 请修改以下路径 ---
# 包含原始 150 帧 [2048,2048] 图像的根目录
SOURCE_DIR = r"/home/ubuntu/data/zhengqinxu/object_detection/DATA/test_video_1028" 
# 您希望保存新生成的 [384,384] 重叠图块的根目录
OUTPUT_DIR = r"/home/ubuntu/data/zhengqinxu/object_detection/DATA/test_video_1028_384" 


ORIGINAL_SIZE = 2048     # 原始图像的尺寸
PATCH_SIZE = 384         # 每个图块的目标尺寸
GRID_COUNT = 6           # 在每个维度上裁剪出多少个图块 (6x6 = 36 个)

# ===================================================================

def split_images_with_overlap():

    
    # --- 0. 准备工作 ---
    print(f"原始图像尺寸: {ORIGINAL_SIZE}x{ORIGINAL_SIZE}")
    print(f"目标图块尺寸: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"将生成 {GRID_COUNT}x{GRID_COUNT} ({GRID_COUNT**2}) 个重叠图块。")

    # 创建输出目录 (新数据将作为一个单独的序列 "1" 存储)
    output_img_dir = os.path.join(OUTPUT_DIR, "img", "1")
    output_mask_dir = os.path.join(OUTPUT_DIR, "mask", "1")
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    
    source_img_paths = sorted(glob(os.path.join(SOURCE_DIR, 'img', '1',  '*.tiff')))
    source_mask_paths = sorted(glob(os.path.join(SOURCE_DIR, 'mask', '1', '*.tiff')))

    if not source_img_paths or not source_mask_paths:
        print(f"错误：在 '{SOURCE_DIR}' 中没有找到 .tiff 文件。请检查 SOURCE_DIR 路径。")
        return
    
    
    # 在 [0, 原始尺寸 - 图块尺寸] 的范围内生成 N 个均匀间隔的起始点
    start_points = np.linspace(0, ORIGINAL_SIZE - PATCH_SIZE, GRID_COUNT).astype(int)
    print(f"生成的裁剪起始坐标: {start_points}")
    stride = start_points[1] - start_points[0]
    overlap = PATCH_SIZE - stride
    print(f"图块间的平均步长约为: {stride} 像素, 平均重叠约为: {overlap} 像素。")
    
    print(f"\n找到 {len(source_img_paths)} 帧原始图像，将生成 {len(source_img_paths) * GRID_COUNT**2} 个图块。")
    
   
    patch_counter = 1 # 用于为新图块命名
    
    for i in tqdm(range(len(source_img_paths)), desc="正在处理原始帧"):
        img_path = source_img_paths[i]
        mask_path = source_mask_paths[i]
        
        image = tifffile.imread(img_path)
        mask = tifffile.imread(mask_path)
        
        # --- 3. 在原始图像上进行重叠裁剪并保存 ---
        # 循环使用我们预先计算好的起始点坐标
        for start_y in start_points:
            for start_x in start_points:
                # 计算裁剪区域的结束坐标
                end_y = start_y + PATCH_SIZE
                end_x = start_x + PATCH_SIZE
                
                # 提取图块
                img_patch = image[start_y:end_y, start_x:end_x]
                mask_patch = mask[start_y:end_y, start_x:end_x]
                
                # 定义新文件名
                output_filename = f"{patch_counter:05d}.tiff"
                
                output_img_path = os.path.join(output_img_dir, output_filename)
                output_mask_path = os.path.join(output_mask_dir, output_filename)
                
                # 保存图块，保持 uint16 数据类型
                tifffile.imwrite(output_img_path, img_patch.astype(np.uint16))
                tifffile.imwrite(output_mask_path, mask_patch.astype(np.uint16))
                
                patch_counter += 1

    print("\n处理完成！")
    print(f"所有 {patch_counter - 1} 个重叠图块已保存到目录: '{OUTPUT_DIR}'")

if __name__ == '__main__':
    split_images_with_overlap()