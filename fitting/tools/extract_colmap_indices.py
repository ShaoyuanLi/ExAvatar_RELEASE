import os
import re
import argparse

def parse_colmap_images_txt(base_path):
    """
    解析 COLMAP 的 images.txt 文件，提取有效图片的帧序号。

    Args:
        base_path (str): COLMAP 项目的根目录路径。

    Returns:
        None: 脚本会直接生成一个文件。
    """
    # 1. 构建输入和输出文件的完整路径
    images_file_path = os.path.join(base_path, 'sparse', 'images.txt')
    output_file_path = os.path.join(base_path, 'frame_indices.txt')

    # 2. 检查 images.txt 文件是否存在
    if not os.path.exists(images_file_path):
        print(f"错误：文件不存在！请确保路径正确: '{images_file_path}'")
        return

    print(f"正在读取 COLMAP 文件: '{images_file_path}'")
    
    valid_indices = set()  # 使用集合来自动处理重复项

    # 3. 读取并解析文件
    with open(images_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # COLMAP 的 images.txt 文件中，有效图片信息在奇数行
    # 我们从第5行开始（索引为4），因为前4行是注释和统计信息
    # 步长为2，意味着我们只读取第5、7、9...行
    for i in range(4, len(lines), 2):
        line = lines[i].strip()
        if not line:
            continue

        # 分割行内容，图片路径是最后一个字段
        parts = line.split()
        image_name = parts[-1]  # 例如: 'images/00001.jpg' 或 'frame_001.png'

        # 使用正则表达式从文件名中提取数字
        # 这个正则会匹配文件名中最后出现的一组数字
        match = re.search(r'(\d+)\.(jpg|jpeg|png)$', image_name, re.IGNORECASE)
        
        if match:
            # match.group(1) 捕获的是数字部分
            index_str = match.group(1)
            try:
                valid_indices.add(int(index_str))
            except ValueError:
                print(f"警告：无法将 '{index_str}' 从文件名 '{image_name}' 转换为数字，已跳过。")

    # 4. 检查是否找到了任何序号
    if not valid_indices:
        print("未在文件中找到任何有效的图片帧序号。")
        return

    # 5. 对序号进行排序并写入输出文件
    sorted_indices = sorted(list(valid_indices))
    
    print(f"成功解析出 {len(sorted_indices)} 个有效图片帧。")
    print(f"正在将序号写入到: '{output_file_path}'")

    with open(output_file_path, 'w', encoding='utf-8') as f:
        for index in sorted_indices:
            f.write(f"{index}\n")

    print("\n处理完成！")


if __name__ == "__main__":
    # --- 设置命令行参数解析 ---
    parser = argparse.ArgumentParser(
        description="从 COLMAP 的 images.txt 文件中提取所有有效图片的帧序号，并保存到 frame_indices.txt。",
        formatter_class=argparse.RawTextHelpFormatter # 保持描述格式
    )
    parser.add_argument(
        "colmap_path", 
        type=str, 
        help="COLMAP 项目的根目录路径。\n例如: /path/to/your/project"
    )

    args = parser.parse_args()
    
    # --- 运行主函数 ---
    parse_colmap_images_txt(args.colmap_path)

