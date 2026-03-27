# stage1
export NCCL_SOCKET_IFNAME=eth3
export GLOO_SOCKET_IFNAME=eth3
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_PORT=29968
export TORCH_DISTRIBUTED_DEBUG=DETAIL


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file .../DiffSynthStudio/examples/wanvideo/model_training/full/accelerate_config_1.3B.yaml \
  --deepspeed_multinode_launcher standard \
  examples/wanvideo/model_training/train.py \
  --dataset_metadata_path data_path \
  --json_mode "multiv" \
  --data_file_keys "video,subject_image" \
  --height 480 \
  --width 832 \
  --dataset_repeat 1 \
  --model_paths '[
      [
          ".../stage1.safetensors"
      ],
      ".../DiffSynthStudio/models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
      ".../DiffSynthStudio/models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
  ]' \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --trainable_models "dit" \
  --output_path "./your_path" \
  --extra_inputs "subject_image"
#     --lora_base_model "dit" \
#   --lora_target_modules "self_attn.q,self_attn.k,self_attn.v,self_attn.o" \
#   --lora_rank 64 \
#--model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth" \