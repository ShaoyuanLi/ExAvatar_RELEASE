import os
import os.path as osp
from glob import glob
import cv2
import argparse
import numpy as np
import re
from tqdm import tqdm

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="从视频或图片序列中按比例采样帧。")
    parser.add_argument('--root_path', type=str, required=True, help="包含视频文件或图片序列的根目录。")
    parser.add_argument('--sampling_ratio', type=float, default=1, help="采样比例 (0.0到1.0之间)。例如，0.5表示采样一半的帧。默认为1.0（全部采样）。")
    parser.add_argument('--target_width', type=int, default=4000, help="降采样的目标宽度。设置为-1可禁用。")
    args = parser.parse_args()
    return args

def find_video_file(directory):
    """在指定目录中查找常见的视频文件。 (保持不变)"""
    video_extensions = ['.mov', '.mp4', '.avi', '.mkv', '.flv', '.wmv']
    for ext in video_extensions:
        files = glob(osp.join(directory, f"*{ext}")) + glob(osp.join(directory, f"*{ext.upper()}"))
        if files:
            return files[0]
    return None

def resize_frame(frame, target_width):
    """辅助函数：统一的图片缩放逻辑 (保持不变)"""
    if target_width is not None and target_width > 0:
        h, w, _ = frame.shape
        if w > target_width:
            ratio = target_width / w
            new_h = int(h * ratio)
            frame = cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_AREA)
    return frame

def process_video(video_path, root_path, sampling_ratio, target_width):
    """
    逻辑分支1：处理视频文件 (已修改为使用 sampling_ratio)
    """
    print(f"模式：视频处理 -> {video_path}")

    output_dir = osp.join(root_path, 'frames')
    os.makedirs(output_dir, exist_ok=True)

    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        print(f"错误：无法打开视频文件 -> {video_path}")
        return

    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频总帧数: {total_frames}")

    if total_frames <= 0:
        print("错误：无法读取视频帧数。")
        return

    # 根据采样比例计算目标帧数
    target_frames = int(total_frames * sampling_ratio)
    if target_frames == 0 and total_frames > 0:
        target_frames = 1 # 保证至少采样一帧
        print("警告：采样比例过低，至少采样1帧。")

    # 计算采样索引
    if target_frames < total_frames:
        sample_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        print(f"根据采样比例 {sampling_ratio:.2f}，进行均匀采样，目标 {target_frames} 帧。")
    else:
        sample_indices = np.arange(total_frames)
        print(f"采样比例为1.0或更高，将提取所有 {total_frames} 帧。")

    saved_frame_count = 0
    for frame_idx in tqdm(sample_indices, desc="提取视频帧"):
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = vidcap.read()

        if success:
            frame = resize_frame(frame, target_width)
            output_filename = osp.join(output_dir, f"{saved_frame_count}.png")
            cv2.imwrite(output_filename, frame)
            saved_frame_count += 1
        else:
            print(f"\n警告：在索引 {frame_idx} 处读取帧失败。")

    vidcap.release()
    print(f"视频处理完成！共保存 {saved_frame_count} 帧到: {output_dir}")

def process_images(root_path, sampling_ratio, target_width):
    """
    逻辑分支2：处理目录下已有的图片 (已修改为使用 sampling_ratio)
    """
    print(f"模式：目录图片序列处理 -> {root_path}")
    img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in img_extensions:
        image_files.extend(glob(osp.join(root_path, ext)))
        image_files.extend(glob(osp.join(root_path, ext.upper())))
    
    if not image_files:
        print(f"错误：在 '{root_path}' 下既没有找到视频，也没有找到图片文件。")
        return
    
    # 自然排序
    def extract_number(filepath):
        filename = osp.basename(filepath)
        numbers = re.findall(r'\d+', filename)
        return int(numbers[-1]) if numbers else -1
    image_files.sort(key=extract_number)

    total_images = len(image_files)
    print(f"找到 {total_images} 张图片，准备处理...")

    # 根据采样比例计算目标帧数
    target_frames = int(total_images * sampling_ratio)
    if target_frames == 0 and total_images > 0:
        target_frames = 1 # 保证至少采样一帧
        print("警告：采样比例过低，至少采样1帧。")

    # 根据目标帧数对图片列表进行采样
    if target_frames < total_images:
        print(f"图片数量 ({total_images}) 大于采样目标数 ({target_frames})，根据比例 {sampling_ratio:.2f} 进行均匀采样。")
        sample_indices = np.linspace(0, total_images - 1, target_frames, dtype=int)
        sampled_image_files = [image_files[i] for i in sample_indices]
    else:
        print(f"采样比例为1.0或更高，将处理所有 {total_images} 张图片。")
        sampled_image_files = image_files
    
    print(f"采样后，将处理 {len(sampled_image_files)} 张图片。")

    output_dir = osp.join(root_path, 'frames')
    os.makedirs(output_dir, exist_ok=True)
    saved_count = 0
    
    for img_path in tqdm(sampled_image_files, desc="处理图片序列"):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"\n警告：无法读取图片 -> {img_path}")
            continue
        frame = resize_frame(frame, target_width)
        output_filename = osp.join(output_dir, f"{saved_count}.png")
        cv2.imwrite(output_filename, frame)
        saved_count += 1
        
    print(f"图片序列处理完成！共处理并保存 {saved_count} 张图片到: {output_dir}")

def main_process(root_path, sampling_ratio, target_width):
    """
    主控制逻辑 (已修改为使用 sampling_ratio)
    """
    if target_width == -1:
        target_width = None

    if not 0.0 <= sampling_ratio <= 1.0:
        print(f"错误：采样比例 --sampling_ratio 必须在 0.0 和 1.0 之间，但收到了 {sampling_ratio}。")
        return

    video_path = find_video_file(root_path)

    if video_path:
        process_video(video_path, root_path, sampling_ratio, target_width)
    else:
        print("未找到视频文件，尝试查找图片序列...")
        process_images(root_path, sampling_ratio, target_width)

if __name__ == "__main__":
    args = parse_args()
    main_process(args.root_path, args.sampling_ratio, args.target_width)
