import torch
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    # model_configs=[
    #     ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
    #     ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
    #     ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    # ],
    model_configs=[
    ModelConfig(path=[
        "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors",
        "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors",
        "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors",
        "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors",
        "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors",
        "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors",
    ]),
    ModelConfig(path="models/Wan-AI/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth"),
    ModelConfig(path="models/Wan-AI/Wan2.1-T2V-14B/Wan2.1_VAE.pth"),
]
)
# /ssd3/vis/zhangxinyao/video/DiffSynth-Studio/models/train/Wan2.1-T2V-14B_lora
pipe.load_lora(pipe.dit, "models/train/Wan2.1-T2V-14B_lora/epoch-4.safetensors", alpha=1)
pipe.enable_vram_management()

# from sunset to night, a small town, light, house, river
video = pipe(
    prompt="狂风巨浪的大海，镜头缓缓推进，一艘渺小的帆船在汹涌的波涛中挣扎漂荡。海面上白沫翻滚，帆船时隐时现，仿佛随时可能被巨浪吞噬。天空乌云密布，雷声轰鸣，海鸥在空中盘旋尖叫。帆船上的人们紧紧抓住缆绳，努力保持平衡。画面风格写实，充满紧张和动感。近景特写，强调风浪的冲击力和帆船的摇晃",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=1, tiled=True
)
save_video(video, "video_Wan2.1-T2V-14B_lora_sea.mp4", fps=15, quality=5)
