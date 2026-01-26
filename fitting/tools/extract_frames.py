import os
import os.path as osp
from glob import glob
import cv2
import argparse
import numpy as np

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="从视频中均匀采样约900帧。")
    parser.add_argument('--root_path', type=str, required=True, help="包含视频文件的根目录。")
    parser.add_argument('--target_frames', type=int, default=1200, help="期望采样的目标帧数。")
    args = parser.parse_args()
    return args

def find_video_file(directory):
    """在指定目录中查找常见的视频文件。"""
    video_extensions = ['.mov', '.mp4', '.avi', '.mkv', '.flv', '.wmv']
    for ext in video_extensions:
        # 使用 glob 查找不区分大小写的文件
        files = glob(osp.join(directory, f"*{ext}")) + glob(osp.join(directory, f"*{ext.upper()}"))
        if files:
            # 返回找到的第一个视频文件
            return files[0]
    return None

def extract_frames(root_path, target_frames):
    """
    从视频中均匀采样帧并保存为图片。
    
    :param root_path: 包含视频文件的根目录。
    :param target_frames: 目标采样帧数。
    """
    # 在根目录中自动查找视频文件
    video_path = find_video_file(root_path)
    if not video_path:
        print(f"错误：在目录 '{root_path}' 中未找到支持的视频文件。")
        return

    print(f"找到视频文件: {video_path}")
    
    # 设置输出目录
    output_dir = osp.join(root_path, 'frames')
    os.makedirs(output_dir, exist_ok=True)

    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        print(f"错误：无法打开视频文件 -> {video_path}")
        return

    # 获取视频总帧数
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频总帧数: {total_frames}")

    if total_frames <= 0:
        print("错误：无法读取视频帧数，或视频为空。")
        vidcap.release()
        return

    # --- 核心修改：计算采样索引 ---
    if total_frames > target_frames:
        # 视频帧数 > 目标帧数，进行均匀采样
        # 使用 np.linspace 生成需要采样的帧的索引列表
        sample_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        print(f"视频帧数 ({total_frames}) > 目标帧数 ({target_frames})。将进行均匀采样。")
    else:
        # 视频帧数 <= 目标帧数，采样所有帧
        sample_indices = np.arange(total_frames)
        print(f"视频帧数 ({total_frames}) <= 目标帧数 ({target_frames})。将提取所有帧。")

    saved_frame_count = 0
    for frame_idx in sample_indices:
        # 使用 set() 直接跳转到目标帧，效率更高
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = vidcap.read()

        if success:
            # 使用一个连续的、补零的数字命名输出文件，便于排序
            output_filename = osp.join(output_dir, f"{saved_frame_count}.png")
            cv2.imwrite(output_filename, frame)
            saved_frame_count += 1
            # 打印进度
            print(f"已保存: {saved_frame_count}/{len(sample_indices)} 帧", end='\r')
        else:
            print(f"\n警告：在索引 {frame_idx} 处读取帧失败。")

    print(f"\n处理完成！总共保存了 {saved_frame_count} 帧图像到目录: {output_dir}")
    vidcap.release()


if __name__ == "__main__":
    args = parse_args()
    extract_frames(args.root_path, args.target_frames)
