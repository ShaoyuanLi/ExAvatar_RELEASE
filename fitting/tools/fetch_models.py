import os
import requests
import shutil
from tqdm import tqdm
from pathlib import Path

# --- 动态路径配置 ---
# 这个脚本将把它的上一级目录作为所有操作的根目录。
# 例如，如果脚本在 .../fitting/tools/，那么 ROOT_DIR 就是 .../fitting/
try:
    # Path(__file__) -> 当前脚本的绝对路径
    # .parents[0] -> 脚本所在的目录
    # .parents[1] -> 脚本所在目录的上一级目录
    ROOT_DIR = Path(__file__).resolve().parents[1]
except NameError:
    # 在交互式环境（如Jupyter）中提供备用方案
    # 假设当前工作目录就是我们想要的根目录的子目录
    print("警告: 未在脚本模式下运行，将使用当前工作目录的上一级作为根目录。")
    ROOT_DIR = Path.cwd().parent

print(f"操作根目录已确定为: {ROOT_DIR}")


# --- 下载任务列表 ---
# 格式: (url, relative_destination_path_from_root, filename)
# 路径中的 'fitting/' 部分将被忽略，因为 ROOT_DIR 已经是 '.../fitting/'
DOWNLOAD_TASKS = [
    # SMPLX and FLAME models for 'common'
    ("https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPLX_FEMALE.npz", "common/utils/human_model_files/smplx", "SMPLX_FEMALE.npz"),
    ("https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPLX_MALE.npz", "common/utils/human_model_files/smplx", "SMPLX_MALE.npz"),
    ("https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPLX_NEUTRAL.npz", "common/utils/human_model_files/smplx", "SMPLX_NEUTRAL.npz"),
    ("https://huggingface.co/camenduru/SMPLer-X/resolve/main/MANO_SMPLX_vertex_ids.pkl", "common/utils/human_model_files/smplx", "MANO_SMPLX_vertex_ids.pkl"),
    ("https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPL-X__FLAME_vertex_ids.npy", "common/utils/human_model_files/smplx", "SMPL-X__FLAME_vertex_ids.npy"),
    ("https://huggingface.co/camenduru/show/resolve/main/data/smplx_flip_correspondences.npz", "common/utils/human_model_files/smplx", "smplx_flip_correspondences.npz"),
    ("https://huggingface.co/camenduru/show/raw/main/data/smplx_uv.obj", "common/utils/human_model_files/smplx/smplx_uv", "smplx_uv.obj"),
    ("https://huggingface.co/camenduru/show/resolve/main/data/flame_dynamic_embedding.npy", "common/utils/human_model_files/flame", "flame_dynamic_embedding.npy"),
    ("https://huggingface.co/camenduru/show/resolve/main/data/flame/FLAME_FEMALE.pkl", "common/utils/human_model_files/flame", "FLAME_FEMALE.pkl"),
    ("https://huggingface.co/camenduru/show/resolve/main/data/flame/FLAME_MALE.pkl", "common/utils/human_model_files/flame", "FLAME_MALE.pkl"),
    ("https://huggingface.co/camenduru/show/resolve/main/data/flame/FLAME_NEUTRAL.pkl", "common/utils/human_model_files/flame", "FLAME_NEUTRAL.pkl"),
    ("https://huggingface.co/camenduru/show/resolve/main/data/flame/flame_static_embedding.pkl", "common/utils/human_model_files/flame", "flame_static_embedding.pkl"),
    ("https://huggingface.co/camenduru/show/resolve/main/data/flame/FLAME_texture.npz", "common/utils/human_model_files/flame", "FLAME_texture.npz"),
    ("https://huggingface.co/camenduru/show/resolve/main/data/flame/2019/generic_model.pkl", "common/utils/human_model_files/flame/2019", "generic_model.pkl"),

    # Models for 'Hand4Whole_RELEASE'
    ("https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPL_NEUTRAL.pkl", "tools/Hand4Whole_RELEASE/common/utils/human_model_files/smpl", "SMPL_NEUTRAL.pkl"),
    ("https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPLX_to_J14.pkl", "tools/Hand4Whole_RELEASE/common/utils/human_model_files/smplx", "SMPLX_to_J14.pkl"),
    ("https://huggingface.co/camenduru/show/resolve/main/data/mano/MANO_LEFT.pkl", "tools/Hand4Whole_RELEASE/common/utils/human_model_files/mano", "MANO_LEFT.pkl"),
    ("https://huggingface.co/camenduru/show/resolve/main/data/mano/MANO_RIGHT.pkl", "tools/Hand4Whole_RELEASE/common/utils/human_model_files/mano", "MANO_RIGHT.pkl"),
    ("https://huggingface.co/camenduru/show/resolve/main/data/flame/FLAME_NEUTRAL.pkl", "tools/Hand4Whole_RELEASE/common/utils/human_model_files/flame", "FLAME_NEUTRAL.pkl"),
    ("https://huggingface.co/camenduru/show/resolve/main/data/flame_dynamic_embedding.npy", "tools/Hand4Whole_RELEASE/common/utils/human_model_files/flame", "flame_dynamic_embedding.npy"),
    ("https://huggingface.co/camenduru/show/resolve/main/data/flame/flame_static_embedding.pkl", "tools/Hand4Whole_RELEASE/common/utils/human_model_files/flame", "flame_static_embedding.pkl"),
    ("https://huggingface.co/camenduru/SMPLer-X/resolve/main/MANO_SMPLX_vertex_ids.pkl", "tools/Hand4Whole_RELEASE/common/utils/human_model_files/smplx", "MANO_SMPLX_vertex_ids.pkl"),
    ("https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPL-X__FLAME_vertex_ids.npy", "tools/Hand4Whole_RELEASE/common/utils/human_model_files/smplx", "SMPL-X__FLAME_vertex_ids.npy"),
    ("https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPLX_NEUTRAL.pkl", "tools/Hand4Whole_RELEASE/common/utils/human_model_files/smplx", "SMPLX_NEUTRAL.pkl"),
    ("https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPLX_NEUTRAL.npz", "tools/Hand4Whole_RELEASE/common/utils/human_model_files/smplx", "SMPLX_NEUTRAL.npz"),
    ("https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPLX_MALE.npz", "tools/Hand4Whole_RELEASE/common/utils/human_model_files/smplx", "SMPLX_MALE.npz"),
    ("https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPLX_FEMALE.npz", "tools/Hand4Whole_RELEASE/common/utils/human_model_files/smplx", "SMPLX_FEMALE.npz"),
    ("https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPL_MALE.pkl", "tools/Hand4Whole_RELEASE/common/utils/human_model_files/smpl", "SMPL_MALE.pkl"),
    ("https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPL_FEMALE.pkl", "tools/Hand4Whole_RELEASE/common/utils/human_model_files/smpl", "SMPL_FEMALE.pkl"),

    # MMPose models
    ("https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth", "tools/mmpose", "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"),
    ("https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth", "tools/mmpose", "dw-ll_ucoco_384.pth"),

    # Segment Anything model
    ("https://huggingface.co/HCMUE-Research/SAM-vit-h/resolve/main/sam_vit_h_4b8939.pth", "tools/segment-anything", "sam_vit_h_4b8939.pth"),

    # Depth Anything V2 model
    ("https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth", "tools/Depth-Anything-V2/checkpoints", "depth_anything_v2_vitl.pth"),
]


# --- 脚本核心函数 ---

def download_file(url: str, dest_path: Path):
    """使用 requests 和 tqdm 下载文件并显示进度条"""
    print(f"\n准备下载: {dest_path.name} -> {dest_path.parent}")
    if dest_path.exists():
        print(f"文件已存在，跳过下载。")
        return
        
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024

        with open(dest_path, 'wb') as f, tqdm(
            desc=dest_path.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                f.write(data)

        if total_size != 0 and bar.n != total_size:
            print(f"错误: 下载 {dest_path.name} 时文件不完整。")
        else:
            print(f"成功下载: {dest_path}")

    except requests.exceptions.RequestException as e:
        print(f"错误: 下载 {url} 失败: {e}")

def main():
    """主执行函数"""
    print("="*50)
    print("开始下载和设置模型资源文件...")
    print("="*50)

    # 1. 执行所有下载任务
    for url, rel_dir, filename in DOWNLOAD_TASKS:
        dest_dir = ROOT_DIR / rel_dir
        dest_file = dest_dir / filename
        
        # 确保目标目录存在
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        download_file(url, dest_file)

    # 2. 执行文件复制操作
    print("\n" + "="*50)
    print("开始执行文件复制操作...")
    print("="*50)

    # 定义复制任务 (路径同样基于 ROOT_DIR)
    # 注意：根据您的要求，第二个复制操作的目标路径需要特殊处理
    copy_tasks = [
        (
            ROOT_DIR / "common/utils/human_model_files/flame/2019/generic_model.pkl",
            ROOT_DIR / "tools/DECA/data/generic_model.pkl"
        ),
        (
            ROOT_DIR / "common/utils/human_model_files",
            # 目标是 /content/ExAvatar_RELEASE/avatar/common/utils/human_model_files
            # 这相当于在 ROOT_DIR (.../fitting) 的父目录 (.../ExAvatar_RELEASE) 下找 avatar 目录
            ROOT_DIR.parent / "avatar/common/utils/human_model_files"
        )
    ]

    for src, dst in copy_tasks:
        print(f"\n复制操作:")
        print(f"  源: {src}")
        print(f"  目标: {dst}")
        
        try:
            # 确保目标文件的父目录存在
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not src.exists():
                print(f"警告: 源路径不存在，跳过复制。")
                continue
            
            if src.is_dir():
                if dst.exists():
                    print("目标目录已存在，将进行覆盖...")
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else: # src is a file
                shutil.copy2(src, dst)
            print("复制成功！")
            
        except Exception as e:
            print(f"错误: 复制失败: {e}")

    print("\n" + "="*50)
    print("所有任务完成！")
    print("="*50)

if __name__ == "__main__":
    main()
