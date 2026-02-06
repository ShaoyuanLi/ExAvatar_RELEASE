#!/bin/sh

# --- 脚本使用说明 ---
#
# 用法: ./your_script_name.sh <subject_id_1> [subject_id_2] ...
#
# 示例:
#   ./your_script_name.sh Jiali
#   ./your_script_name.sh Jiali ZhangSan LiSi
#
# ---------------------

# 检查是否提供了至少一个 subject_id 参数
# $# 表示参数的个数
if [ $# -eq 0 ]; then
    echo "错误: 缺少 subject_id 参数！"
    echo "用法: $0 <subject_id_1> [subject_id_2] ..."
    exit 1
fi

# --- 外层循环：遍历所有输入的 subject_id ---
# "$@" 代表命令行传入的所有参数列表
for SUBJECT_ID in "$@"
do
    echo ""
    echo "#######################################################"
    echo ">>> [$(date +'%Y-%m-%d %H:%M:%S')] 开始处理目标 Subject ID: ${SUBJECT_ID}"
    echo "#######################################################"

    # --- 配置区 (针对当前 SUBJECT_ID) ---
    # 1. 模型快照文件所在的目录
    MODEL_DIR="../output/model_dump/${SUBJECT_ID}"

    # 2. 基础训练命令
    BASE_COMMAND="python train.py --subject_id ${SUBJECT_ID}"

    # 3. 每次重试前的等待时间（秒）
    RETRY_DELAY=10
    # --- 配置结束 ---

    echo ">>> 模型目录: ${MODEL_DIR}"

    # --- 内层循环：针对当前任务的无限重试机制 ---
    while true
    do
        # 在每次循环开始时，都检查快照文件是否存在
        if [ -d "${MODEL_DIR}" ] && ls "${MODEL_DIR}/snapshot_"*.pth >/dev/null 2>&1; then
            # 如果存在，设置命令为继续训练
            COMMAND="${BASE_COMMAND} --continue"
            echo ">>> [$(date +'%Y-%m-%d %H:%M:%S')] 检测到模型快照，将尝试从断点继续训练。"
        else
            # 如果不存在，设置命令为开始新训练
            COMMAND="${BASE_COMMAND}"
            echo ">>> [$(date +'%Y-%m-%d %H:%M:%S')] 未检测到模型快照或目录，将尝试开始新的训练。"
        fi

        echo "-------------------------------------------------------"
        echo ">>> [$(date +'%Y-%m-%d %H:%M:%S')] 正在尝试执行命令..."
        echo ">>> ${COMMAND}"
        echo "-------------------------------------------------------"

        # 直接执行命令
        eval ${COMMAND}

        # 获取刚刚执行命令的退出码
        exit_code=$?

        # 检查命令是否成功
        if [ ${exit_code} -eq 0 ]; then
            # 如果成功 (退出码为0)
            echo "======================================================="
            echo ">>> [$(date +'%Y-%m-%d %H:%M:%S')] Subject ID: ${SUBJECT_ID} 任务成功完成！"
            echo "======================================================="
            
            # 【关键点】这里 break 只会跳出当前的 while 循环
            # 脚本会继续执行外层 for 循环的下一次迭代（即下一个 subject_id）
            break 
        else
            # 如果失败 (退出码非0)，打印失败信息并准备重试
            echo
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            echo ">>> [$(date +'%Y-%m-%d %H:%M:%S')] 命令执行失败！"
            echo ">>> Subject ID: ${SUBJECT_ID}"
            echo ">>> 退出码: ${exit_code}"
            echo ">>> 将在 ${RETRY_DELAY} 秒后自动重试..."
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

            # 等待指定时间
            sleep ${RETRY_DELAY}
        fi
    done
done

echo ""
echo ">>> 所有任务均已执行完毕。"
