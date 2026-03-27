accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path /ssd3/vis/zhangxinyao/video/T2V/data/train \
  --dataset_metadata_path /ssd3/vis/zhangxinyao/video/T2V/data/train/metadata.csv \
  --height 512 \
  --width 512 \
  --dataset_repeat 10 \
  --model_paths '[
      [
          "/ssd3/vis/zhangxinyao/video/DiffSynthStudio/models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors",
          "/ssd3/vis/zhangxinyao/video/DiffSynthStudio/models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors",
          "/ssd3/vis/zhangxinyao/video/DiffSynthStudio/models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors",
          "/ssd3/vis/zhangxinyao/video/DiffSynthStudio/models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors",
          "/ssd3/vis/zhangxinyao/video/DiffSynthStudio/models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors",
          "/ssd3/vis/zhangxinyao/video/DiffSynthStudio/models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors"
      ],
      "/ssd3/vis/zhangxinyao/video/DiffSynthStudio/models/Wan-AI/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth",
      "/ssd3/vis/zhangxinyao/video/DiffSynthStudio/models/Wan-AI/Wan2.1-T2V-14B/Wan2.1_VAE.pth"
  ]' \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-T2V-14B_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  # --tokenizer_path models/Wan-AI/Wan2.1-T2V-14B/google/umt5-xxl --skip_download


#--model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-14B:Wan2.1_VAE.pth" \