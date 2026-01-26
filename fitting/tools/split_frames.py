import os
import glob
import random
import re

# --- 1. 请在这里配置您的参数 ---

# 包含所有图片帧的目录路径
# 注意：脚本会自动在这个路径下寻找名为 "frames" 的子目录
IMAGE_DIRECTORY = "/home/lishaoyuan/ExAvatar_RELEASE/fitting/data/Custom/data/Jiali/"

# 图片文件的扩展名 (例如: '.jpg', '.png', '.jpeg')
IMAGE_EXTENSION = ".png"

# 原始视频的帧率 (例如 30 fps)
ORIGINAL_FPS = 10

# 训练集的目标采样帧率 (例如 5 fps)
TARGET_FPS = 3

# 测试集的大小
TEST_SET_SIZE = 10

# --- 2. 脚本主要逻辑 (已按要求修改) ---

def get_index_from_filename(filename: str):
    """
    从文件名中提取数字序号。
    例如：'00123.png' -> 123
    如果找不到数字，则返回 None。
    """
    # 使用正则表达式查找文件名中的所有数字
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    return None

def generate_frame_lists_split(image_dir, image_ext, original_fps, target_fps, test_size):
    """
    根据实际文件名解析序号，生成 all、train 和 test 列表。
    - train 列表按固定步长采样。
    - test 列表从 train 列表之外随机选取。
    - 所有列表都输出从文件名解析出的真实序号。
    """
    # --- 步骤 1: 验证并查找图片文件 ---
    image_dir = os.path.expanduser(image_dir)
    image_frames_dir = os.path.join(image_dir, "frames")
    if not os.path.isdir(image_frames_dir):
        print(f"❌ 错误：目录 '{image_frames_dir}' 不存在。请检查您的路径配置。")
        return

    # 获取所有图片文件并按自然语言排序（保证 '2.png' 在 '10.png' 之前）
    image_files = sorted(glob.glob(os.path.join(image_frames_dir, f'*{image_ext}')), 
                         key=lambda f: int(re.search(r'\d+', os.path.basename(f)).group()))

    if not image_files:
        print(f"❌ 错误：在目录 '{image_frames_dir}' 中没有找到任何 '{image_ext}' 文件。")
        return

    print(f"✅ 在 '{image_frames_dir}' 中检测到 {len(image_files)} 张图片。")
    print("-" * 40)

    # --- 步骤 2: 从所有文件名中解析出真实序号 ---
    all_indices = []
    for f_path in image_files:
        filename = os.path.basename(f_path)
        index = get_index_from_filename(filename)
        if index is not None:
            all_indices.append(index)
        else:
            print(f"⚠️ 警告：无法从文件名 '{filename}' 中解析出序号，已跳过。")
    
    if not all_indices:
        print("❌ 错误：未能从任何文件中解析出有效的帧序号。")
        return

    # --- 步骤 3: 生成 frame_list_all.txt ---
    all_frames_path = os.path.join(image_dir, "frame_list_all.txt")
    print(f"📄 正在生成 {all_frames_path}...")
    with open(all_frames_path, 'w') as f:
        f.write('\n'.join(map(str, sorted(all_indices))) + '\n')
    print(f" -> 'frame_list_all.txt' 生成完毕，包含 {len(all_indices)} 个真实序号。")

    # --- 步骤 4: 生成 frame_list_train.txt ---
    if target_fps > 0 and original_fps >= target_fps:
        step = max(1, original_fps // target_fps)
    else:
        step = 1

    print(f"\n📄 正在以 {step} 帧为步长采样生成训练集...")
    train_frames_path = os.path.join(image_dir, "frame_list_train.txt")

    train_indices = set()
    # 注意：这里我们对 image_files 列表进行步长采样
    sampled_files = image_files[::step]
    
    with open(train_frames_path, 'w') as f_train:
        for f_path in sampled_files:
            index = get_index_from_filename(os.path.basename(f_path))
            if index is not None:
                f_train.write(f"{index}\n")
                train_indices.add(index)

    print(f" -> 'frame_list_train.txt' 生成完毕，包含 {len(train_indices)} 个帧序号。")

    # --- 步骤 5: 生成 frame_list_test.txt ---
    print("\n📄 正在生成测试集...")
    test_frames_path = os.path.join(image_dir, "frame_list_test.txt")

    # 从所有真实序号中，找出不属于训练集的序号
    test_candidate_indices = list(set(all_indices) - train_indices)

    if len(test_candidate_indices) < test_size:
        print(f"⚠️ 警告：可用于测试集的帧不足 {test_size} 帧 (只有 {len(test_candidate_indices)} 帧)。将使用所有可用的帧。")
        test_indices = test_candidate_indices
    else:
        random.seed(42)
        test_indices = random.sample(test_candidate_indices, test_size)

    with open(test_frames_path, 'w') as f_test:
        for index in sorted(test_indices):
            f_test.write(f"{index}\n")

    print(f" -> 'frame_list_test.txt' 生成完毕，包含 {len(test_indices)} 个随机帧序号。")
    print("-" * 40)
    print(f"🎉 所有文件已在目录 '{image_dir}' 中成功生成！")

# --- 3. 运行脚本 ---
if __name__ == "__main__":
    generate_frame_lists_split(IMAGE_DIRECTORY, IMAGE_EXTENSION, ORIGINAL_FPS, TARGET_FPS, TEST_SET_SIZE)
