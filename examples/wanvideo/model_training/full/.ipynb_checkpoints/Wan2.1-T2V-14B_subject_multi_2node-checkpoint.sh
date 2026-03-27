ps -ef | grep multiply | grep -v grep | cut -c 9-16 | xargs kill -s 9
source /root/paddlejob/workspace/env_run/aisee_home/disk2/wanglei127/anaconda3/bin/activate wan_training


# --------- 基础：环境 ---------
# source /root/paddlejob/workspace/env_run/aisee_home/disk2/wanglei127/anaconda3/bin/activate wan_training

# 避免 MPI/OMPI 干扰
unset OMPI_COMM_WORLD_LOCAL_RANK
unset OMPI_COMM_WORLD_SIZE
unset OMPI_COMM_WORLD_RANK
unset PMIX_RANK || true

# 多卡训练程序：优先使用 torch 自带 NCCL
export TORCH_LIB_DIR=$(python - <<'PY'
import torch, os
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)
export LD_LIBRARY_PATH="$TORCH_LIB_DIR:${LD_LIBRARY_PATH:-}"

export PYTHONPATH=$PWD:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo $PYTHONPATH

# --------- 两机固定配置 ---------
# 只允许 POD_INDEX=0/1 参与
if [ -z "${POD_INDEX:-}" ]; then
  echo "[FATAL] POD_INDEX is not set"
  exit 1
fi

if [ "$POD_INDEX" -ne 0 ] && [ "$POD_INDEX" -ne 1 ]; then
  echo "POD_INDEX=$POD_INDEX not in {0,1}, exiting..."
  exit 0
fi

RANK=$POD_INDEX
echo "POD_INDEX=$POD_INDEX POD_IP=${POD_IP:-unknown} -> machine_rank=$RANK"

# 主节点（rank0）IP：用你提供的 POD_INDEX=0 的 POD_IP
MAIN_IP="10.54.130.166"
MAIN_PORT="45123"

# 两机总数
NUM_MACHINES=2

# 每机 GPU 数（默认 8；如果不是 8，请改这里）
GPUS_PER_MACHINE=8
NUM_PROCESSES=$((NUM_MACHINES * GPUS_PER_MACHINE))

# --------- 业务参数 ---------
expid=r2v-14b-dinov3-l

# （可选）更稳的清理：只杀自己用户下的相关进程，避免误伤
# pkill -9 -f "examples/wanvideo/model_training/train1.py" || true
# pkill -9 -f "accelerate launch" || true

accelerate launch \
  --config_file examples/wanvideo/model_training/full/accelerate_config_14B_multi.yaml \
  --deepspeed_multinode_launcher standard \
  --machine_rank $RANK \
  --num_machines $NUM_MACHINES \
  --num_processes $NUM_PROCESSES \
  --main_process_ip $MAIN_IP \
  --main_process_port $MAIN_PORT \
  examples/wanvideo/model_training/train1.py \
    --output_path /root/paddlejob/workspace/env_run/${expid} \
    --model_paths '[
      [
        "/root/paddlejob/workspace/env_run/aisee_home/disk2/wanglei127/DiffSynthStudio/models/train/wan14b/diffusion_pytorch_model-00001-of-00006.safetensors",
        "/root/paddlejob/workspace/env_run/aisee_home/disk2/wanglei127/DiffSynthStudio/models/train/wan14b/diffusion_pytorch_model-00002-of-00006.safetensors",
        "/root/paddlejob/workspace/env_run/aisee_home/disk2/wanglei127/DiffSynthStudio/models/train/wan14b/diffusion_pytorch_model-00003-of-00006.safetensors",
        "/root/paddlejob/workspace/env_run/aisee_home/disk2/wanglei127/DiffSynthStudio/models/train/wan14b/diffusion_pytorch_model-00004-of-00006.safetensors",
        "/root/paddlejob/workspace/env_run/aisee_home/disk2/wanglei127/DiffSynthStudio/models/train/wan14b/diffusion_pytorch_model-00005-of-00006.safetensors",
        "/root/paddlejob/workspace/env_run/aisee_home/disk2/wanglei127/DiffSynthStudio/models/train/wan14b/diffusion_pytorch_model-00006-of-00006.safetensors"
      ],
      "/root/paddlejob/workspace/env_run/aisee_home/disk2/wanglei127/DiffSynthStudio/models/train/wan14b/models_t5_umt5-xxl-enc-bf16.pth",
      "/root/paddlejob/workspace/env_run/aisee_home/disk2/wanglei127/DiffSynthStudio/models/train/wan14b/Wan2.1_VAE.pth"
    ]' \
    --dataset_metadata_path "/root/paddlejob/bos_dhc_rw/wanglei/Jsons-mask_and_bbox_High-Quality-filtered-5.5aes/Jsons-mask_and_bbox_High-Quality-total_part*" \
    --json_mode "multiv" \
    --data_file_keys "video,subject_image" \
    --height 480 \
    --width 832 \
    --dataset_repeat 1 \
    --learning_rate 5e-5 \
    --num_epochs 1 \
    --remove_prefix_in_ckpt "pipe.dit." \
    --trainable_models "dit" \
    --extra_inputs "subject_image" \
    --gradient_accumulation_steps 1