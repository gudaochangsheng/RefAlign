ps -ef | grep multiply | grep -v grep | cut -c 9-16 | xargs kill -s 9
source .../anaconda3/bin/activate wan_training

unset OMPI_COMM_WORLD_LOCAL_RANK
unset OMPI_COMM_WORLD_SIZE
unset OMPI_COMM_WORLD_RANK

# 多卡训练程序
# 统一用 torch 自带 NCCL
export TORCH_LIB_DIR=$(python - <<'PY'
import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)
export LD_LIBRARY_PATH="$TORCH_LIB_DIR:${LD_LIBRARY_PATH:-}"

# RANK=$1
RANK=$POD_INDEX

if [ "$POD_INDEX" -eq 0 ]; then
    echo "POD_INDEX is 0 (故障节点), exiting script..."
    exit 0
fi


if [ "$POD_INDEX" -eq 11 ]; then
    echo "POD_INDEX is 11 (故障节点), exiting script..."
    exit 0
fi

if [ "$POD_INDEX" -lt 11 ]; then
    RANK=$((POD_INDEX - 1))
    echo "POD_INDEX=$POD_INDEX < 11, setting RANK=$RANK"
elif [ "$POD_INDEX" -gt 11 ]; then
    RANK=$((POD_INDEX - 2))
    echo "POD_INDEX=$POD_INDEX (12-15), setting RANK=$RANK"
fi


export PYTHONPATH=$PWD:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


expid=r2v-stage2
accelerate launch \
  --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml \
  --deepspeed_multinode_launcher standard \
  --machine_rank $RANK \
  --num_machines 14 \
  --num_processes 112 \
  --main_process_ip 10.54.130.95 \
  examples/wanvideo/model_training/train.py \
    --output_path /root/paddlejob/workspace/env_run/${expid} \
    --model_paths '[
      [
       "stage1.safetensors",
      ],
      ".../DiffSynthStudio/models/train/wan14b/models_t5_umt5-xxl-enc-bf16.pth",
      ".../DiffSynthStudio/models/train/wan14b/Wan2.1_VAE.pth"
    ]' \
    --dataset_metadata_path data_path \
    --json_mode "multiv" \
    --data_file_keys "video,subject_image" \
    --height 480 \
    --width 832 \
    --dataset_repeat 1 \
    --learning_rate 1e-5 \
    --num_epochs 1 \
    --remove_prefix_in_ckpt "pipe.dit." \
    --trainable_models "dit" \
    --extra_inputs "subject_image" \
    --gradient_accumulation_steps 1