CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file /root/paddlejob/workspace/env_run/wanglei/DiffSynthStudio/examples/wanvideo/model_training/full/accelerate_config_1.3B.yaml examples/wanvideo/model_training/train.py \
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