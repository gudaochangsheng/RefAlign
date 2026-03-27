import torch
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    # model_configs=[
    #     ModelConfig(model_id="Wan-AI/Wan2.1-FLF2V-14B-720P", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
    #     ModelConfig(model_id="Wan-AI/Wan2.1-FLF2V-14B-720P", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
    #     ModelConfig(model_id="Wan-AI/Wan2.1-FLF2V-14B-720P", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    #     ModelConfig(model_id="Wan-AI/Wan2.1-FLF2V-14B-720P", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", offload_device="cpu"),
    # ],
    model_configs=[
        ModelConfig(path=[
            "/root/paddlejob/workspace/env_run/gpubox03_ssd1/models/Wan-AI/Wan2.1-FLF2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors",
            "/root/paddlejob/workspace/env_run/gpubox03_ssd1/models/Wan-AI/Wan2.1-FLF2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors",
            "/root/paddlejob/workspace/env_run/gpubox03_ssd1/models/Wan-AI/Wan2.1-FLF2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors",
            "/root/paddlejob/workspace/env_run/gpubox03_ssd1/models/Wan-AI/Wan2.1-FLF2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors",
            "/root/paddlejob/workspace/env_run/gpubox03_ssd1/models/Wan-AI/Wan2.1-FLF2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors",
            "/root/paddlejob/workspace/env_run/gpubox03_ssd1/models/Wan-AI/Wan2.1-FLF2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors",
            "/root/paddlejob/workspace/env_run/gpubox03_ssd1/models/Wan-AI/Wan2.1-FLF2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors"
        ]),
        ModelConfig(path="models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(path="models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"),
        ModelConfig(path="/root/paddlejob/workspace/env_run/gpubox03_ssd1/models/Wan-AI/Wan2.1-FLF2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")
    ],
)
pipe.enable_vram_management()

# dataset_snapshot_download(
#     dataset_id="DiffSynth-Studio/examples_in_diffsynth",
#     local_dir="./",
#     allow_file_pattern=["data/examples/wan/first_frame.jpeg", "data/examples/wan/last_frame.jpeg"]
# )

# First and last frame to video
video = pipe(
    # prompt="写实风格，一个女生手持枯萎的花站在花园中，镜头逐渐拉远，记录下花园的全貌。",
    # prompt="Ultra HD 8K footage of a pitbull dog (left) vs tabby cat (right) boxing match, slow-motion punch impact, sweat droplets flying, Sony α7 IV camera with 100mm macro lens, cinematic lighting, motion blur perfectly captured",
    prompt="一只狗（左） 和一只 花猫（右）正在进行拳击比赛，慢动作击打冲击，汗滴飞溅，镜头固定，电影照明，运动模糊完美捕捉。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    input_image=Image.open("data/examples/wan/example1/test_first_frame_eidit.png").resize((720, 1280)),
    end_image=Image.open("data/examples/wan/example1/test_last_frame_eidit.png").resize((720, 1280)),
    seed=0, tiled=True,
    height=720, width=1280, num_frames=81,
    sigma_shift=16,
)
save_video(video, "video_example_FLF2V_14B.mp4", fps=16, quality=5) #960 ,33
