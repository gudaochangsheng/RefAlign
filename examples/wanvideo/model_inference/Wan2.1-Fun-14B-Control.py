import torch
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda:2",
    # model_configs=[
    #     ModelConfig(model_id="PAI/Wan2.1-Fun-14B-Control", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
    #     ModelConfig(model_id="PAI/Wan2.1-Fun-14B-Control", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
    #     ModelConfig(model_id="PAI/Wan2.1-Fun-14B-Control", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    #     ModelConfig(model_id="PAI/Wan2.1-Fun-14B-Control", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", offload_device="cpu"),
    # ],
    model_configs=[
        ModelConfig(path=[
            "/root/paddlejob/workspace/env_run/gpubox03_ssd1/models/PAI/Wan2.1-Fun-14B-Control/diffusion_pytorch_model.safetensors"
        ]),
        ModelConfig(path="/root/paddlejob/workspace/env_run/gpubox03_ssd1/models/PAI/Wan2.1-Fun-14B-Control/models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(path="/root/paddlejob/workspace/env_run/gpubox03_ssd1/models/PAI/Wan2.1-Fun-14B-Control/Wan2.1_VAE.pth"),
        ModelConfig(path="/root/paddlejob/workspace/env_run/gpubox03_ssd1/models/PAI/Wan2.1-Fun-14B-Control/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")
    ],
)
pipe.enable_vram_management()

# dataset_snapshot_download(
#     dataset_id="DiffSynth-Studio/examples_in_diffsynth",
#     local_dir="./",
#     allow_file_pattern=f"data/examples/wan/control_video.mp4"
# )

# Control video
# control_video = VideoData("data/examples/wan/control_video.mp4", height=832, width=576)
control_video = VideoData("/ssd3/vis/zhangxinyao/video/DiffSynthStudio/data/examples/wan/example1/example1_video-depth.mp4", height=720, width=1280)
video = pipe(
    prompt="一只狗（左） 和一只 花猫（右）正在进行拳击比赛，慢动作击打冲击，汗滴飞溅，镜头固定，电影照明，运动模糊完美捕捉。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    control_video=control_video, height=720, width=1280, num_frames=81,
    seed=1, tiled=True
)
save_video(video, "video_control_example.mp4", fps=16, quality=5)
