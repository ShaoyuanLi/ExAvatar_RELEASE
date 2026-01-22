import numpy as np
import json
import os
import sys
import re
def qvec2rotmat(qvec):
    """
    将 COLMAP 的四元数 (w, x, y, z) 转换为 3x3 旋转矩阵。
    """
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
    ])

def read_colmap_data(colmap_dir):
    """
    从 cameras.txt 和 images.txt 读取相机内外参。
    
    Args:
        colmap_dir (str): 包含 COLMAP 输出文件的目录。

    Returns:
        list: 一个包含所有图像完整相机参数的列表。
    """
    # 1. 读取相机内参 (cameras.txt)
    cameras = {}
    cameras_path = os.path.join(colmap_dir, "cameras.txt")
    with open(cameras_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            
            parts = line.strip().split()
            camera_id = int(parts[0])
            model = parts[1]
            
            # 根据 PINHOLE 或 SIMPLE_PINHOLE 模型解析内参
            if model in ["SIMPLE_PINHOLE", "PINHOLE"]:
                fx = float(parts[4])
                fy = float(parts[5]) if model == "PINHOLE" else fx
                cx = float(parts[6]) if model == "PINHOLE" else float(parts[5])
                cy = float(parts[7]) if model == "PINHOLE" else float(parts[6])
                cameras[camera_id] = {
                    "focal": [fx, fy],
                    "princpt": [cx, cy]
                }
            else:
                print(f"警告: 不支持的相机模型 '{model}'，跳过相机 ID {camera_id}")

    # 2. 读取图像外参并组合内外参 (images.txt)
    all_frames_data = []
    images_path = os.path.join(colmap_dir, "images.txt")
    with open(images_path, "r") as f:
        lines = f.readlines()
        # 每次读取两行，第一行是外参，第二行是特征点（我们忽略它）
        for i in range(0, len(lines), 2):
            pose_line = lines[i].strip()
            if pose_line.startswith("#"):
                continue

            parts = pose_line.split()
            
            # 提取外参和关联信息
            qvec = np.array(parts[1:5], dtype=np.float64) # (QW, QX, QY, QZ)
            tvec = np.array(parts[5:8], dtype=np.float64) # (TX, TY, TZ)
            camera_id = int(parts[8])
            image_name = parts[9]
            
            # 检查相机ID是否存在
            if camera_id not in cameras:
                print(f"警告: 图像 '{image_name}' 的相机 ID {camera_id} 无效，跳过。")
                continue

            # 组合数据
            frame_data = {
                "R": qvec2rotmat(qvec).tolist(),
                "t": tvec.tolist(),
                "focal": cameras[camera_id]["focal"],
                "princpt": cameras[camera_id]["princpt"],
                "image_name": image_name
            }
            all_frames_data.append(frame_data)
            
    return all_frames_data

def main():
    """
    主执行函数。
    """
    # 检查命令行参数
    if len(sys.argv) != 2:
        print("用法: python colmap_to_json.py <colmap_data_directory>")
        sys.exit(1)
        
    colmap_dir = sys.argv[1]
    
    # 验证路径是否存在
    if not os.path.isdir(colmap_dir) or \
       not os.path.exists(os.path.join(colmap_dir, "cameras.txt")) or \
       not os.path.exists(os.path.join(colmap_dir, "images.txt")):
        print(f"错误: 目录 '{colmap_dir}' 不存在或缺少 'cameras.txt'/'images.txt' 文件。")
        sys.exit(1)

    print("正在读取 COLMAP 数据...")
    all_frames = read_colmap_data(colmap_dir)

    if not all_frames:
        print("未能读取任何有效的图像数据。")
        return

    print(f"找到了 {len(all_frames)} 张图像。正在生成 JSON 文件...")
    os.makedirs(os.path.join(colmap_dir, "camera_params"), exist_ok=True)

    # 遍历所有帧数据并保存为 JSON
    for frame in all_frames:
        original_name = frame["image_name"]
        numbers = re.findall(r'\d+', original_name)
        if not numbers:
            print(f"警告: 在文件名 '{original_name}' 中未找到数字，跳过此文件。")
            continue
        # 默认使用找到的第一个数字序列作为文件名
        output_base_name = numbers[0]
        
        output_filename = os.path.join(colmap_dir, "camera_params", f"{output_base_name}.json")
        
        # 准备要写入 JSON 的字典，不包含临时用的 image_name
        json_output = {
            "R": frame["R"],
            "t": frame["t"],
            "focal": frame["focal"],
            "princpt": frame["princpt"]
        }
        
        # 写入紧凑的单行 JSON 文件
        with open(output_filename, "w") as f:
            json.dump(json_output, f, separators=(',', ':'))

    print(f"处理完成！所有 JSON 文件已直接保存到 '{colmap_dir}' 目录。")

if __name__ == "__main__":
    main()
