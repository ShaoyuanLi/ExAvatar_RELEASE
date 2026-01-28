#!/bin/sh

# --- 脚本使用说明 ---
#
# 用法: ./your_script_name.sh <subject_id>
#
# 示例:
#   ./your_script_name.sh Jiali
#   ./your_script_name.sh ZhangSan
#
# ---------------------

# 检查是否提供了 subject_id 参数
if [ -z "$1" ]; then
    echo "错误: 缺少 subject_id 参数！"
    echo "用法: $0 <subject_id>"
    exit 1
fi

# --- 配置区 ---
# 1. 将第一个命令行参数赋值给 SUBJECT_ID
SUBJECT_ID="$1"

# 2. 模型快照文件所在的目录 (使用变量)
MODEL_DIR="../output/model_dump/${SUBJECT_ID}"

# 3. 基础训练命令 (使用变量)
BASE_COMMAND="python train.py --subject_id ${SUBJECT_ID}"

# 4. 每次重试前的等待时间（秒）
RETRY_DELAY=10
# --- 配置结束 ---

echo ">>> 脚本启动，目标 Subject ID: ${SUBJECT_ID}"
echo ">>> 模型目录: ${MODEL_DIR}"

# 无限循环
while true
do
    # 在每次循环开始时，都检查快照文件是否存在
    # 使用 ls 和通配符来查找文件，>/dev/null 2>&1 技巧可以安静地检查文件是否存在
    # 注意：如果目录不存在，ls会报错，所以我们先确保目录存在
    if [ -d "${MODEL_DIR}" ] && ls "${MODEL_DIR}/snapshot_"*.pth >/dev/null 2>&1; then
        # 如果存在，设置命令为继续训练
        COMMAND="${BASE_COMMAND} --continue"
        echo ">>> [$(date +'%Y-%m-%d %H:%M:%S')] 检测到模型快照，将尝试从断点继续训练。"
    else
        # 如果不存在，设置命令为开始新训练
        COMMAND="${BASE_COMMAND}"
        echo ">>> [$(date +'%Y-%m-%d %H:%M:%S')] 未检测到模型快照或目录，将尝试开始新的训练。"
    fi

    echo "======================================================="
    echo ">>> [$(date +'%Y-%m-%d %H:%M:%S')] 正在尝试执行命令..."
    echo ">>> ${COMMAND}"
    echo "======================================================="

    # 直接执行命令。输出会直接显示在屏幕上。
    # 使用 eval 来确保命令中的变量被正确解析和执行
    eval ${COMMAND}

    # 获取刚刚执行命令的退出码
    exit_code=$?

    # 检查命令是否成功
    if [ ${exit_code} -eq 0 ]; then
        # 如果成功 (退出码为0)，打印成功信息并跳出循环
        echo "======================================================="
        echo ">>> [$(date +'%Y-%m-%d %H:%M:%S')] 命令成功完成！脚本退出。"
        echo "======================================================="
        break
    else
        # 如果失败 (退出码非0)，打印失败信息并准备重试
        echo # 添加一个空行以提高可读性
        echo "-------------------------------------------------------"
        echo ">>> [$(date +'%Y-%m-%d %H:%M:%S')] 命令执行失败！"
        echo ">>> 退出码: ${exit_code}"
        echo ">>> 将在 ${RETRY_DELAY} 秒后自动重试..."
        echo "-------------------------------------------------------"

        # 等待指定时间
        sleep ${RETRY_DELAY}
    fi
done
