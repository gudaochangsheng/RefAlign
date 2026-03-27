# 1. 禁用 InfiniBand，强制走 TCP
export NCCL_IB_DISABLE=1
export NCCL_IB_CUDA_SUPPORT=0

# 2. 禁用 P2P (如果机器间没有 NVLink 或 PCIe P2P 支持有问题，也建议关掉)
export NCCL_P2P_DISABLE=1

# 3. 指定正确的网卡 (非常重要！)
# 请先在终端运行 `ip addr` 确认你的内网 IP 绑定的网卡名
# 假设是 eth0 或 xgbe0，如果是 xgbe0，请确保它支持 TCP
export NCCL_SOCKET_IFNAME=xgbe0  
# 如果不确定，可以尝试设为 eth0
# export NCCL_SOCKET_IFNAME=eth0



export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# ================= accelerate multi nodes config =================
MASTER_ADDR="yq02-inf-sci-k8s-a800-hbxgn6-0254.yq02.baidu.com"
MASTER_PORT="29968"
NNODES=2
GPUS_PER_NODE=8
NUM_PROCESSES=$(($NNODES * $GPUS_PER_NODE))

NODE_RANK=1
echo "Current Configuration:"
echo "  Master: $MASTER_ADDR:$MASTER_PORT"
echo "  Node Rank: $NODE_RANK / $NNODES"
echo "  Total Processes: $NUM_PROCESSES"


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file /root/paddlejob/workspace/env_run/wanglei/DiffSynthStudio/examples/wanvideo/model_training/full/accelerate_config_1.3B_ms.yaml \
  --num_machines $NNODES \
  --machine_rank $NODE_RANK \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  --num_processes $NUM_PROCESSES \
  --deepspeed_multinode_launcher standard \
  examples/wanvideo/model_training/train.py \
  --dataset_metadata_path /root/paddlejob/workspace/env_run/wanglei/opensv_5m_release_1_to_train.json /root/paddlejob/workspace/env_run/wanglei/opensv_5m_release_2_to_train.json /root/paddlejob/workspace/env_run/wanglei/opensv_5m_release_4_to_train.json \
  --json_mode "multiv" \
  --data_file_keys "video,subject_image" \
  --height 480 \
  --width 832 \
  --dataset_repeat 1 \
  --model_paths '[
      [
          "/root/paddlejob/workspace/env_run/wanglei/DiffSynthStudio/models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
      ],
      "/root/paddlejob/workspace/env_run/wanglei/DiffSynthStudio/models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
      "/root/paddlejob/workspace/env_run/wanglei/DiffSynthStudio/models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
  ]' \
  --learning_rate 5e-5 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --trainable_models "dit" \
  --output_path "./models/train/Wan2.1-T2V-1.3B_full_dino" \
  --extra_inputs "subject_image"
#     --lora_base_model "dit" \
#   --lora_target_modules "self_attn.q,self_attn.k,self_attn.v,self_attn.o" \
#   --lora_rank 64 \
#--model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth" \