import torch
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new3 import WanVideoPipeline, ModelConfig
import inspect, re, functools
from torchvision import transforms
from PIL import Image, ImageOps
from diffsynth.models.set_hypernet import set_hyper
from diffsynth.models.set_inputmlp import set_inputmlp
from diffsynth.models.set_dual_LoRA import set_unsharedLoRA
# Load BiRefNet with weights
from transformers import AutoModelForImageSegmentation
birefnet = AutoModelForImageSegmentation.from_pretrained('.../BiRefNet', trust_remote_code=True)
from torchvision import transforms


@torch.inference_mode()
def birefnet_mask_only(birefnet, pil_img: Image.Image, device="cuda:5", div=32):
    """
    输入：PIL 任意尺寸
    输出：mask(PIL L)，尺寸 == 原图尺寸
    div：把输入 pad 到可被 div 整除（31/32 具体哪个更合适看模型，这里先用 32 更常见）
    """
    birefnet = birefnet.to(device).eval()

    img_rgb = pil_img.convert("RGB")
    W, H = img_rgb.size

    # --- pad 到 div 的整数倍 ---
    pad_h = (div - H % div) % div
    pad_w = (div - W % div) % div
    if pad_h or pad_w:
        padded = Image.new("RGB", (W + pad_w, H + pad_h), (0, 0, 0))
        padded.paste(img_rgb, (0, 0))
    else:
        padded = img_rgb

    x = transforms.ToTensor()(padded)
    x = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])(x)
    x = x.unsqueeze(0).to(device)  # float32

    out = birefnet(x)
    if isinstance(out, (list, tuple)):
        logits = out[-1]
    elif hasattr(out, "logits"):
        logits = out.logits
    else:
        logits = out[-1] if isinstance(out, dict) else out

    pred = logits.sigmoid()[0].detach().float().cpu().squeeze()
    mask = transforms.ToPILImage()(pred)

    # mask 先对齐到 padded 尺寸，再裁回原图尺寸
    mask = mask.resize(padded.size, Image.Resampling.BILINEAR)
    mask = mask.crop((0, 0, W, H))
    return mask

def apply_subject_mask(
    birefnet,
    pil_img: Image.Image,
    device="cuda:5",
    bg_color=(255, 255, 255),
    div=32,
):
    """
    输入：已 resize/crop 到目标尺寸的 PIL.Image (RGB)
    输出：背景被替换为 bg_color 的 PIL.Image (RGB)
    """
    mask = birefnet_mask_only(birefnet, pil_img, device=device, div=div)
    img = pil_img.convert("RGB")
    bg = Image.new("RGB", img.size, bg_color)
    out = Image.composite(img, bg, mask)  # mask 白=保留前景，黑=用背景
    return out



pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda:5",
    # model_configs=[
    #     ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
    #     ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
    #     ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    # ],
    model_configs=[
    ModelConfig(path=[
        "model.safetensors",
    ]),
    ModelConfig(path=".../DiffSynthStudio/models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"),
    ModelConfig(path=".../DiffSynthStudio/models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")],
)

pipe.enable_vram_management()

def short_resize_and_crop_pil(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """
    按比例缩放后填充，输出指定大小的 PIL.Image。
    - 会保持原图纵横比；
    - 缩放后以白色背景居中填充到目标尺寸；
    """
    W, H = image.size
    img_ratio = W / H
    target_ratio = target_width / target_height

    # 等比例缩放
    if img_ratio > target_ratio:  # 图片更宽
        new_width = target_width
        new_height = int(new_width / img_ratio)
    else:  # 图片更高
        new_height = target_height
        new_width = int(new_height * img_ratio)

    # Resize
    img = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 填充到目标尺寸
    delta_w = target_width - new_width
    delta_h = target_height - new_height
    padding = (
        delta_w // 2,
        delta_h // 2,
        delta_w - delta_w // 2,
        delta_h - delta_h // 2,
    )
    new_img = ImageOps.expand(img, padding, fill=(255, 255, 255))
    return new_img

# Text-to-video
# video = pipe(
#     prompt="纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
# )
# save_video(video, "video_Wan2.1-T2V-1.3B.mp4", fps=15, quality=5)

def make_blank_subject(width=832, height=480, rgb=127):
    """生成中性灰空白图，尽量等价于“零特征”注入。"""
    img = Image.new("RGB", (width, height), (rgb, rgb, rgb))
    # 如果你仍想用自己的等比填充函数，也可以套一层：
    # img = short_resize_and_crop_pil(img, width, height)
    return img

# 生成两张空白 subject（如果你传的是多主体位）
blank1 = make_blank_subject(832, 480, 255)
blank1.save("blank_subject1.png")


subject_image = Image.open(".../OpenS2V-Nexus/OpenS2V-Eval/Images/multihuman/woman/13.jpg")
subject_image = short_resize_and_crop_pil(subject_image, 832,480)
subject_image = apply_subject_mask(birefnet, subject_image, device="cuda:5", bg_color=(255,255,255))
subject_image.save("subject_resize1.png")
# assert 2==1
subject_image1 = Image.open(".../OpenS2V-Nexus/OpenS2V-Eval/Images/multihuman/man/14.jpg")


subject_image1 = short_resize_and_crop_pil(subject_image1, 832,480)
subject_image1 = apply_subject_mask(birefnet, subject_image1, device="cuda:1", bg_color=(255,255,255))
subject_image1.save("subject_resize2.png")
subject_image2 = Image.open(".../OpenS2V-Nexus/OpenS2V-Eval/Images/humanobj/thing/animal/dog/4.jpg")

subject_image2 = short_resize_and_crop_pil(subject_image2, 832,480)
subject_image2 = apply_subject_mask(birefnet, subject_image2, device="cuda:1", bg_color=(255,255,255))
subject_image2.save("subject_resize3.png")
subject_image3 = Image.open(".../OpenS2V-Nexus/OpenS2V-Eval/Images/humanobj/environment/school/6.jpg")
subject_image3 = short_resize_and_crop_pil(subject_image3, 832,480)
# subject_image3 = apply_subject_mask(birefnet, subject_image3, device="cuda:5", bg_color=(255,255,255))
subject_image3.save("subject_resize4.png")

# prompt = "The video features two individuals standing side by side in front of a car, with a building and some greenery in the background. The person on the left is wearing a light blue button-up shirt and is holding a car key fob in their right hand, raised slightly above shoulder level. The individual on the right is dressed in a light-colored, possibly beige, button-up shirt with a subtle pattern. Both individuals are facing the camera, and their posture suggests they are posing for the photo. Throughout the video, there are no significant changes in their positions, expressions, or the environment around them. The lighting remains consistent, indicating that the video was likely taken in a single continuous shot without any noticeable camera movement or action progression."
prompt = "A man and a woman are standing side by side in a well-lit outdoor setting, facing the camera with a relaxed yet confident demeanor. The woman, dressed in a light blue button-up shirt, is on the left and holding a car key fob in her right hand, raised slightly above shoulder level. The man, wearing a beige button-up shirt with a subtle pattern, stands to her right. Behind them, a sleek car is parked, with a modern building partially visible and framed by lush greenery in the background. The scene is composed as a wide shot, capturing the subjects in full height and situating them naturally within the environment. The lighting is even, with soft shadows indicating a clear day. The camera remains steady, subtly focused on the duo, as a gentle breeze animates the scene with a flutter of their shirts and a faint rustling of the greenery in the background."
# prompt = "a man playing with his dog in front of the house"
# prompt = "A cheerful man stands in the front yard of a modest, weathered house with rusted corrugated roofing and faded yellow walls, engaging in a lively game of fetch with his playful black Labrador. The man, dressed casually to match the relaxed atmosphere, laughs as he throws a small stick into the air. The dog eagerly leaps and sprints to catch it, kicking up bits of loose dirt from the dry ground. The camera captures the scene with a wide shot, gently panning to follow the dog as it races back to the man, tail wagging with excitement. The sunlight casts a warm glow over the entire setting, adding life to the lush green trees bordering the scene. The background reveals an aura of history, with the aging structure quietly hinting at stories of the past while the joy of the interaction brings energy and movement to the frame."
video = pipe(
    prompt=prompt,
    # negative_prompt='split-screen, multi-panel, collage, picture-in-picture, two scenes, multiple scenes, montage,duplicated subject, twin, clone, double subject, two people, extra person, extra body, multiple bodies,identity drift, face swap, wrong face, inconsistent face, inconsistent clothing, outfit change, age change, gender change,cutout, sticker, pasted, floating subject, halo, outline, edge artifacts, green screen, unnatural boundary,bad composition, off-center subject, cropped head, cropped body, out of frame,temporal flicker, frame-to-frame inconsistency, jitter, wobble, warping, melting, morphing, swimming textures,ghosting, motion smear, shimmering, crawling artifacts,text, subtitles, watermark, logo, caption,overexposure, oversaturated, lowres, blurry, jpeg artifacts, noisy, banding,bad anatomy, deformed body, disfigured face, extra fingers, fused fingers, missing fingers, malformed hands,messy background, crowded background, too many background people',
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    subject_image=[subject_image, subject_image1],
    seed=0, tiled=True,
    cfg_scale=5.0
)
save_video(video, "video_Wan2.1-T2V-1.3B-subject3.mp4", fps=16, quality=9)
