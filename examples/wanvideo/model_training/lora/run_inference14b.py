# -*- coding: utf-8 -*-
"""
8卡并行：从 JSON 读取条目，按 rank 划分到各GPU并行生成视频。
支持按 key 的“类型前缀”过滤样本，例如 singleobj_1 -> type=singleobj

启动方式示例：
  torchrun --nproc_per_node=8 generate_videos_dist.py --keep-order
  torchrun --nproc_per_node=8 generate_videos_dist.py --types humanobj
  torchrun --nproc_per_node=8 generate_videos_dist.py --types humanobj singlehuman
  torchrun --nproc_per_node=8 generate_videos_dist.py --types humanobj,singlehuman
"""

import os, json, traceback, argparse
from typing import List, Dict, Any, Optional, Tuple

import torch
from PIL import Image, ImageOps

from diffsynth import save_video
from diffsynth.pipelines.wan_video_new6 import WanVideoPipeline, ModelConfig


# ================== 可调参数 ==================
BASE_DIR = ".../OpenS2V-Nexus/OpenS2V-Eval"
JSON_PATH = os.path.join(BASE_DIR, "Open-Domain_Eval_with_prompt.json")
# JSON_PATH = os.path.join(BASE_DIR, "Open-Domain_Eval.json")
OUT_DIR  = os.path.join(BASE_DIR, "Generated_Videos-14b")

WIDTH, HEIGHT = 832, 480
DEFAULT_CFG = 5.0
DEFAULT_FPS = 16
DEFAULT_QUALITY = 9

# NEGATIVE_PROMPT = (
#     "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
#     "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，"
#     "畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
# )
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，"
    "畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走，"
    "动画风，二次元，动漫，漫画，插画，手绘，卡通，Q版，赛璐璐风格，概念艺术，数字绘画，非写实，风格化渲染，"
    "3D渲染，CG感，游戏CG，塑料质感，蜡像感，假人感，建模感，过度磨皮，过度锐化，"
    "不真实的光影，不真实的材质，不真实的肤色，艺术化处理，电影特效感，"
    "面部僵硬，表情不自然，五官失真，人脸不真实，身体比例异常，动作不自然，运动轨迹异常，"
    "帧间闪烁，轮廓抖动，局部扭曲，背景跳变，卡顿，不连贯，低真实感"
)

MODEL_DIFF = ".../model.safetensors"
MODEL_TXT  = ".../DiffSynthStudio/models/train/wan14b/models_t5_umt5-xxl-enc-bf16.pth"
MODEL_VAE  = ".../DiffSynthStudio/models/train/wan14b/Wan2.1_VAE.pth"
# =============================================


def short_resize_and_crop_pil(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """
    你的原实现：按短边缩放并 pad 到目标尺寸
    """
    W, H = image.size
    img_ratio = W / H
    target_ratio = target_width / target_height
    if img_ratio > target_ratio:
        new_width = target_width
        new_height = int(new_width / img_ratio)
    else:
        new_height = target_height
        new_width = int(new_height * img_ratio)
    img = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    delta_w = target_width - new_width
    delta_h = target_height - new_height
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
    return ImageOps.expand(img, padding, fill=(255, 255, 255))


def make_blank_subject(width=832, height=480, rgb=127):
    return Image.new("RGB", (width, height), (rgb, rgb, rgb))


def load_subjects(img_paths: List[str]) -> List[Image.Image]:
    """
    不做抠图：只负责读图 + resize/pad
    """
    subjects: List[Image.Image] = []
    if not img_paths:
        return subjects

    for rel_path in img_paths:
        abs_path = os.path.join(BASE_DIR, str(rel_path).lstrip("/"))
        if not os.path.exists(abs_path):
            print(f"[警告] 找不到图像：{abs_path}")
            continue

        try:
            img = Image.open(abs_path).convert("RGB")
            img = short_resize_and_crop_pil(img, WIDTH, HEIGHT)
            subjects.append(img)
        except Exception as e:
            print(f"[警告] 打开/处理图像失败：{abs_path}, 原因: {e}")

    return subjects


def init_pipeline(device_str: str) -> WanVideoPipeline:
    print(f"==> [{device_str}] 初始化 WanVideoPipeline ...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device_str,
        model_configs=[
            ModelConfig(path=[MODEL_DIFF]),
            ModelConfig(path=MODEL_TXT),
            ModelConfig(path=MODEL_VAE),
        ],
    )
    pipe.enable_vram_management()
    print(f"==> [{device_str}] 管线加载完成。")
    return pipe


def parse_bool(x: Any, default: bool = False) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    return default


def parse_types(raw_types: Optional[List[str]]) -> Optional[set]:
    """
    支持：
      --types humanobj singlehuman
      --types humanobj,singlehuman
    """
    if not raw_types:
        return None
    merged: List[str] = []
    for t in raw_types:
        merged.extend([x for x in str(t).split(",") if x.strip()])
    return set([x.strip() for x in merged if x.strip()])


def get_key_type(key: str) -> str:
    """
    你的 JSON key：singleobj_1 / humanobj_23 / singlehuman_7
    这里 type = key.split("_")[0]
    """
    parts = str(key).split("_", 1)
    return parts[0] if parts else str(key)


def worker(rank: int, world_size: int, keep_order: bool, type_set: Optional[set]):
    # 绑定设备
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    device_str = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

    # 读取 JSON
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        spec: Dict[str, Any] = json.load(f)

    items_all: List[Tuple[str, Dict[str, Any]]] = list(spec.items()) if keep_order else sorted(spec.items(), key=lambda kv: kv[0])

    # 先按 type 过滤（关键：过滤后再做分片，避免每卡负载不均）
    if type_set is not None:
        filtered: List[Tuple[str, Dict[str, Any]]] = []
        for k, v in items_all:
            kt = get_key_type(k)
            if kt in type_set:
                filtered.append((k, v))
        items = filtered
    else:
        items = items_all

    os.makedirs(OUT_DIR, exist_ok=True)

    # 划分子集：过滤后的 items 再按 i % world_size == rank
    my_subset = [(i, k, v) for i, (k, v) in enumerate(items) if i % world_size == rank]
    total = len(items)
    mine = len(my_subset)
    type_info = f", types={sorted(type_set)}" if type_set is not None else ""
    print(f"[Rank {rank}/{world_size}] 总样本 {total}{type_info}，本rank处理 {mine} 个。")

    if mine == 0:
        print(f"[Rank {rank}] 无样本可处理，退出。")
        return

    # 初始化管线
    pipe = init_pipeline(device_str)

    # 逐个生成
    for global_idx, key, item in my_subset:
        try:
            print(f"\n==== [Rank {rank}] 处理 {key} (全局idx={global_idx}) ====")

            # 取 prompt（优先 refine，如果你想用 refine）
            # prompt = (item.get("prompt_refine") or item.get("prompt") or "").strip()
            prompt = (item.get("prompt_refine") or "").strip()

            if not prompt:
                print(f"[Rank {rank}] [跳过] {key} 缺少 prompt。")
                continue

            # synthesis_flag 从 JSON 读
            # synthesis_flag = parse_bool(item.get("synthesis_flag"), default=False)

            img_paths = item.get("img_paths")
            subjects = load_subjects(img_paths)

            if not subjects:
                subjects = [make_blank_subject(WIDTH, HEIGHT)]

            print(f"[Rank {rank}] key_type={get_key_type(key)}, prompt前80字: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
            # print(f"[Rank {rank}] synthesis_flag={synthesis_flag}, 参考图数量={len(subjects)}")

            # 不传 seed -> 每次随机
            video = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                subject_image=subjects,
                tiled=True,
                cfg_scale=DEFAULT_CFG
            )

            out_path = os.path.join(OUT_DIR, f"{key}.mp4")
            save_video(video, out_path, fps=DEFAULT_FPS, quality=DEFAULT_QUALITY)
            print(f"[Rank {rank}] [完成] {out_path}")

        except Exception as e:
            print(f"[Rank {rank}] [错误] 处理 {key} 失败：{e}")
            traceback.print_exc()

    print(f"\n[Rank {rank}] 任务完成 ✅")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-order", action="store_true",
                        help="按 JSON 写入顺序处理（默认按键名排序）。")
    parser.add_argument(
        "--types", nargs="*", default=None,
        help="只推理指定 key 前缀类型。例：--types humanobj 或 --types humanobj singlehuman 或 --types humanobj,singlehuman"
    )
    parser.add_argument(
        "--use-refine", action="store_true",
        help="使用 prompt_refine（若存在），否则使用 prompt。"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 从 torchrun 环境变量获取 rank/world_size；未用 torchrun 时退化为单卡
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    type_set = parse_types(args.types)

    print(f"[Launcher] WORLD_SIZE={world_size}, LOCAL_RANK={rank}, types={sorted(type_set) if type_set else None}")

    # 如果你想用 refine，在这里切换 prompt 逻辑（保持最小改动，放一个简单开关）
    # 我这里不在 worker 传参改 prompt，避免改动太大；你需要 refine 时，把 worker 里 prompt 那行注释切换即可。
    worker(rank=rank, world_size=world_size, keep_order=args.keep_order, type_set=type_set)