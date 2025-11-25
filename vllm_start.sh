
#!/bin/bash

# 定义模型目录和参数
MODEL_DIR="/data/users/rwang/lib/models/Qwen"  # 替换为实际模型目录
MODEL_NAME="Qwen1.5-7B-Chat"          # 替换为实际模型名称
# TEMPLATE_PATH="./chat_template.jinja"  # 替换为实际模板路径

# 进入模型目录并启动服务
cd "$MODEL_DIR" || { echo "无法进入目录 $MODEL_DIR"; exit 1; }
CUDA_VISIBLE_DEVICES=0,1,2 vllm serve "$MODEL_NAME" \
--tensor-parallel-size 3 
--port 8000
# --chat-template "$TEMPLATE_PATH"

# CUDA_VISIBLE_DEVICES=0,1 vllmserve {local_path}\
# --served-model-name Qwen/Qwen3-8B 
# --max_model_len 4096 
# --tensor-parallel-size 2 
# --port 7890
