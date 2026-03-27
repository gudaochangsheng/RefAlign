import torch, warnings, glob, os, types
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoImageProcessor, AutoModel
from qwen_vl_utils import process_vision_info
import numpy as np
from PIL import Image
from einops import repeat, reduce
from typing import Optional, Union
from dataclasses import dataclass
from modelscope import snapshot_download
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
from typing_extensions import Literal
import torch.nn.functional as F
from ..models import ModelManager, load_state_dict
from ..models.wan_video_dit import WanModel, RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_text_encoder import WanTextEncoder, T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_vae import WanVideoVAE, RMS_norm, CausalConv3d, Upsample
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vace import VaceWanModel
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..schedulers.flow_match import FlowMatchScheduler
from ..prompters import WanPrompter
from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear, WanAutoCastLayerNorm
from ..lora import GeneralLoRALoader
import torch.nn as nn
from transformers import CLIPVisionModel, AutoImageProcessor, Siglip2VisionModel
import random



class BasePipeline(torch.nn.Module):

    def __init__(
        self,
        device="cuda", torch_dtype=torch.float16,
        height_division_factor=64, width_division_factor=64,
        time_division_factor=None, time_division_remainder=None,
    ):
        super().__init__()
        # The device and torch_dtype is used for the storage of intermediate variables, not models.
        self.device = device
        self.torch_dtype = torch_dtype
        # The following parameters are used for shape check.
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.vram_management_enabled = False
        
        
    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        super().to(*args, **kwargs)
        return self


    def check_resize_height_width(self, height, width, num_frames=None):
        # Shape check
        if height % self.height_division_factor != 0:
            height = (height + self.height_division_factor - 1) // self.height_division_factor * self.height_division_factor
            print(f"height % {self.height_division_factor} != 0. We round it up to {height}.")
        if width % self.width_division_factor != 0:
            width = (width + self.width_division_factor - 1) // self.width_division_factor * self.width_division_factor
            print(f"width % {self.width_division_factor} != 0. We round it up to {width}.")
        if num_frames is None:
            return height, width
        else:
            if num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames = (num_frames + self.time_division_factor - 1) // self.time_division_factor * self.time_division_factor + self.time_division_remainder
                print(f"num_frames % {self.time_division_factor} != {self.time_division_remainder}. We round it up to {num_frames}.")
            return height, width, num_frames


    def preprocess_image(self, image, torch_dtype=None, device=None, pattern="B C H W", min_value=-1, max_value=1):
        # Transform a PIL.Image to torch.Tensor
        image = torch.Tensor(np.array(image, dtype=np.float32))
        image = image.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
        image = image * ((max_value - min_value) / 255) + min_value
        image = repeat(image, f"H W C -> {pattern}", **({"B": 1} if "B" in pattern else {}))
        return image


    # def preprocess_video(self, video, torch_dtype=None, device=None, pattern="B C T H W", min_value=-1, max_value=1):
    #     # Transform a list of PIL.Image to torch.Tensor
    #     video = [self.preprocess_image(image, torch_dtype=torch_dtype, device=device, min_value=min_value, max_value=max_value) for image in video]
    #     video = torch.stack(video, dim=pattern.index("T") // 2)
    #     return video
    def preprocess_video(self, video, torch_dtype=None, device=None,
                        pattern="B C T H W", min_value=-1, max_value=1):
        """
        支持两种输入：
        1) 单个视频: List[PIL.Image]              -> Tensor [1, C, T, H, W]
        2) batch 视频: List[List[PIL.Image]]      -> Tensor [B, C, T, H, W]
        """
        if len(video) == 0:
            raise ValueError("video 为空")

        # 情况 1：单个视频 -> List[Image]
        if isinstance(video[0], Image.Image):
            # 原来的逻辑：逐帧 preprocess_image，然后在 T 维 stack
            frames = [
                self.preprocess_image(
                    image,
                    torch_dtype=torch_dtype,
                    device=device,
                    min_value=min_value,
                    max_value=max_value,
                )  # 每帧: [1, C, H, W]
                for image in video
            ]
            # pattern 默认 "B C T H W"，"T" 的 index=4, 4//2=2 -> 在 dim=2 堆 T
            t_dim = pattern.index("T") // 2
            video_tensor = torch.stack(frames, dim=t_dim)  # [1, C, T, H, W]
            return video_tensor

        # 情况 2：batch 视频 -> List[List[Image]]
        elif isinstance(video[0], (list, tuple)):
            # 对每个样本都走一遍“单视频”逻辑（递归调用自己）
            video_tensors = [
                self.preprocess_video(
                    v,
                    torch_dtype=torch_dtype,
                    device=device,
                    pattern=pattern,
                    min_value=min_value,
                    max_value=max_value,
                )  # 每个: [1, C, T, H, W]
                for v in video
            ]
            # 把多个 [1, C, T, H, W] 在 batch 维拼起来 -> [B, C, T, H, W]
            batch_tensor = torch.cat(video_tensors, dim=0)
            return batch_tensor

        else:
            raise TypeError(
                f"preprocess_video 期望输入为 List[Image] 或 List[List[Image]]，实际是: {type(video[0])}"
            )


    def vae_output_to_image(self, vae_output, pattern="B C H W", min_value=-1, max_value=1):
        # Transform a torch.Tensor to PIL.Image
        if pattern != "H W C":
            vae_output = reduce(vae_output, f"{pattern} -> H W C", reduction="mean")
        image = ((vae_output - min_value) * (255 / (max_value - min_value))).clip(0, 255)
        image = image.to(device="cpu", dtype=torch.uint8)
        image = Image.fromarray(image.numpy())
        return image


    def vae_output_to_video(self, vae_output, pattern="B C T H W", min_value=-1, max_value=1):
        # Transform a torch.Tensor to list of PIL.Image
        if pattern != "T H W C":
            vae_output = reduce(vae_output, f"{pattern} -> T H W C", reduction="mean")
        video = [self.vae_output_to_image(image, pattern="H W C", min_value=min_value, max_value=max_value) for image in vae_output]
        return video


    def load_models_to_device(self, model_names=[]):
        if self.vram_management_enabled:
            # offload models
            for name, model in self.named_children():
                if name not in model_names:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        for module in model.modules():
                            if hasattr(module, "offload"):
                                module.offload()
                    else:
                        model.cpu()
            torch.cuda.empty_cache()
            # onload models
            for name, model in self.named_children():
                if name in model_names:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        for module in model.modules():
                            if hasattr(module, "onload"):
                                module.onload()
                    else:
                        model.to(self.device)


    def generate_noise(self, shape, seed=None, rand_device="cpu", rand_torch_dtype=torch.float32, device=None, torch_dtype=None):
        # Initialize Gaussian noise
        generator = None if seed is None else torch.Generator(rand_device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=rand_device, dtype=rand_torch_dtype)
        noise = noise.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
        return noise


    def enable_cpu_offload(self):
        warnings.warn("`enable_cpu_offload` will be deprecated. Please use `enable_vram_management`.")
        self.vram_management_enabled = True
        
        
    def get_vram(self):
        return torch.cuda.mem_get_info(self.device)[1] / (1024 ** 3)
    
    
    def freeze_except(self, model_names):
        for name, model in self.named_children():
            if name in model_names:
                model.train()
                model.requires_grad_(True)
            else:
                model.eval()
                model.requires_grad_(False)


@dataclass
class ModelConfig:
    path: Union[str, list[str]] = None
    model_id: str = None
    origin_file_pattern: Union[str, list[str]] = None
    download_resource: str = "ModelScope"
    offload_device: Optional[Union[str, torch.device]] = None
    offload_dtype: Optional[torch.dtype] = None
    skip_download: bool = False

    def download_if_necessary(self, local_model_path="./models", skip_download=False, use_usp=False):
        if self.path is None:
            # Check model_id and origin_file_pattern
            if self.model_id is None:
                raise ValueError(f"""No valid model files. Please use `ModelConfig(path="xxx")` or `ModelConfig(model_id="xxx/yyy", origin_file_pattern="zzz")`.""")
            
            # Skip if not in rank 0
            if use_usp:
                import torch.distributed as dist
                skip_download = dist.get_rank() != 0
                
            # Check whether the origin path is a folder
            if self.origin_file_pattern is None or self.origin_file_pattern == "":
                self.origin_file_pattern = ""
                allow_file_pattern = None
                is_folder = True
            elif isinstance(self.origin_file_pattern, str) and self.origin_file_pattern.endswith("/"):
                allow_file_pattern = self.origin_file_pattern + "*"
                is_folder = True
            else:
                allow_file_pattern = self.origin_file_pattern
                is_folder = False
            
            # Download
            skip_download = skip_download or self.skip_download
            if not skip_download:
                downloaded_files = glob.glob(self.origin_file_pattern, root_dir=os.path.join(local_model_path, self.model_id))
                snapshot_download(
                    self.model_id,
                    local_dir=os.path.join(local_model_path, self.model_id),
                    allow_file_pattern=allow_file_pattern,
                    ignore_file_pattern=downloaded_files,
                    local_files_only=False
                )
            
            # Let rank 1, 2, ... wait for rank 0
            if use_usp:
                import torch.distributed as dist
                dist.barrier(device_ids=[dist.get_rank()])
                
            # Return downloaded files
            if is_folder:
                self.path = os.path.join(local_model_path, self.model_id, self.origin_file_pattern)
            else:
                self.path = glob.glob(os.path.join(local_model_path, self.model_id, self.origin_file_pattern))
            if isinstance(self.path, list) and len(self.path) == 1:
                self.path = self.path[0]

def gate_from_timestep(t, t_first, t_last, beta=1.0, descending=True, min_val=0.0, max_val=1.0):
    """
    t: 当前整数时间步（或张量）
    t_first: 序列第一个步（通常高噪声，比如 999）
    t_last:  序列最后一个步（通常低噪声，比如 0）
    descending: timesteps 是否降序（True 表示从高噪声→低噪声）
    beta: 曲率，>1更陡，<1更平
    [min_val, max_val]: 输出范围
    """
    eps = 1e-8
    if descending:
        # t 越靠前（高噪声）→ 归一化 x 越大（接近 1）
        x = (t - t_last) / (t_first - t_last + eps)
    else:
        x = (t - t_first) / (t_last - t_first + eps)

    x = x.clamp(0, 1)
    gate01 = (1.0 - x).pow(beta)            # 映射到 [0,1]
    gate = min_val + (max_val - min_val) * gate01
    return gate


class WanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=None):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.motion_controller: WanMotionControllerModel = None
        self.vace: VaceWanModel = None
        self.in_iteration_models = ("dit", "motion_controller", "vace")
        self.unit_runner = PipelineUnitRunner()
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_ImageEmbedder(),
            WanVideoUnit_FunControl(),
            WanVideoUnit_FunReference(),
            WanVideoUnit_FunCameraControl(),
            WanVideoUnit_SpeedControl(),
            WanVideoUnit_VACE(),
            WanVideoUnit_UnifiedSequenceParallel(),
            WanVideoUnit_TeaCache(),
            WanVideoUnit_CfgMerger(),
            WanVideoUnit_Subject(),
        ]
        self.model_fn = model_fn_wan_video
        self.qwen = None
        self.qwen_tokenizer = None
        self.qwen_processor = None
        self.dino = None
        self.dino_processor = None
        self.clip = None
        self.clip_processor = None
        
    def encode_subject_image(self, subject_image):
        self.load_models_to_device(["vae"])
        
        images = []
        for image in subject_image:
            image = image.resize(self.weight, self.height)
            print(image.size)
            image = (
            torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
            )  # [3, H, W]
            print(image.shape)
            image = (
                image.unsqueeze(1).unsqueeze(0).to(dtype=self.torch_dtype)
            )  # [B, 3, 1, H, W]
            image = image * 2 - 1
        images.append(image)
        images = torch.cat(images, dim=2)
        subject_image_latent = self.vae.encode(images, device=self.device, tiled=False)
        return subject_image_latent

    def load_lora(self, module, path, alpha=1):
        loader = GeneralLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
        lora = load_state_dict(path, torch_dtype=self.torch_dtype, device=self.device)
        loader.load(module, lora, alpha=alpha)
    # def load_lora_mlp(self, module, path, alpha=1.0):
    #     """
    #     module: 一般传 self.dit
    #     path:   保存了 LoRA + (可选) input_mlp/output_mlp 权重的 ckpt 路径或目录
    #     """
    #     # 1) 读权重
    #     loader = GeneralLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
    #     sd = load_state_dict(path, torch_dtype=self.torch_dtype, device="cpu")  # 先放 CPU

    #     # 2) 拆分 LoRA vs MLP
    #     def is_mlp_key(k: str) -> bool:
    #         return k.startswith("input_mlp.") or k.startswith("output_mlp.")

    #     lora_sd = {k: v for k, v in sd.items() if not is_mlp_key(k)}
    #     mlp_sd  = {k: v for k, v in sd.items() if is_mlp_key(k)}

    #     # 3) 先加载 LoRA
    #     if lora_sd:
    #         loader.load(module, lora_sd, alpha=alpha)

    #     # 4) 处理 input_mlp / output_mlp
    #     if mlp_sd:
    #         dit = module  # 这里一般就是 self.dit

    #         def ensure_and_load_mlp(mlp_name: str):
    #             prefix = f"{mlp_name}."
    #             sub = {k[len(prefix):]: v for k, v in mlp_sd.items() if k.startswith(prefix)}
    #             if not sub:
    #                 return

    #             # 从固定键名直接拿首末两层的 Conv3d 权重
    #             def pick_weights(subdict):
    #                 # 优先使用 net.0 / net.3 的权重（这是你现在的命名）
    #                 k_first = "net.0.weight"
    #                 k_last  = "net.3.weight"
    #                 if k_first in subdict and k_last in subdict:
    #                     return subdict[k_first], subdict[k_last]
    #                 # 兜底：找所有 5D 的 weight，按键名排序取首末
    #                 cand = [(k, v) for k, v in subdict.items() if k.endswith("weight") and hasattr(v, "dim") and v.dim() == 5]
    #                 if len(cand) >= 2:
    #                     cand.sort(key=lambda x: x[0])
    #                     return cand[0][1], cand[-1][1]
    #                 raise RuntimeError(f"{mlp_name}: 未找到 5D Conv3d 权重（例如 net.0.weight / net.3.weight）。")

    #             w1, w2 = pick_weights(sub)  # w1: [hidden, in, 1,1,1], w2: [out, hidden, 1,1,1]
    #             in_channels     = int(w1.shape[1])
    #             hidden_channels = int(w1.shape[0])
    #             out_channels    = int(w2.shape[0])

    #             # 如 dit 上没有该模块则创建
    #             if not hasattr(dit, mlp_name) or getattr(dit, mlp_name) is None:
    #                 mlp = InputMLP3D(
    #                     in_channels=in_channels,
    #                     hidden_channels=hidden_channels,
    #                     out_channels=out_channels,
    #                     dropout=0.0
    #                 )
    #                 setattr(dit, mlp_name, mlp)

    #             # 移到目标 device / dtype
    #             getattr(dit, mlp_name).to(device=self.device, dtype=self.torch_dtype)

    #             # 加载权重（strict=False 以兼容 Dropout/Identity 差异）
    #             res = getattr(dit, mlp_name).load_state_dict(sub, strict=False)
    #             if getattr(res, "missing_keys", None):
    #                 print(f"[load_lora] {mlp_name} 缺失参数: {res.missing_keys}")
    #             if getattr(res, "unexpected_keys", None):
    #                 print(f"[load_lora] {mlp_name} 多余参数: {res.unexpected_keys}")

    #         ensure_and_load_mlp("input_mlp")
    #         ensure_and_load_mlp("output_mlp")

    #     print("[load_lora] LoRA 与附带的 input_mlp/output_mlp 权重加载完成。")
    
    def load_lora_mlp(self, module, path, alpha=1.0):
        """
        module: 一般传 self.dit
        path  : LoRA ckpt 路径 (里面可能含 input_mlp/output_mlp 权重)
        """
        import re, torch

        def is_lin(w):   return isinstance(w, torch.Tensor) and w.dim() == 2  # [out,in]
        def is_conv1(w): return isinstance(w, torch.Tensor) and w.dim() == 5 and tuple(w.shape[-3:]) == (1,1,1)
        def to_conv1(w): return w.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) if is_lin(w) else w

        loader = GeneralLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
        sd = load_state_dict(path, torch_dtype=self.torch_dtype, device="cpu")

        # 先加载 LoRA（排除 input/output_mlp 前缀）
        lora_sd = {k: v for k, v in sd.items() if not (k.startswith("input_mlp.") or k.startswith("output_mlp."))}
        if lora_sd:
            loader.load(module, lora_sd, alpha=alpha)

        # 处理 input_mlp / output_mlp
        for mlp_name in ["input_mlp", "output_mlp"]:
            prefix = f"{mlp_name}."
            sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            if not sub:
                continue

            # 收集所有 net.<idx>.weight，按 idx 排序
            cands = []
            for k, v in sub.items():
                if not k.endswith(".weight"):
                    continue
                if is_lin(v) or is_conv1(v):
                    m = re.search(r"net\.(\d+)\.weight$", k)
                    idx = int(m.group(1)) if m else 10**9
                    cands.append((idx, v))
            if len(cands) < 2:
                raise RuntimeError(f"{mlp_name} 权重不足，至少需要两层 (找到 {len(cands)})")

            cands.sort(key=lambda x: x[0])
            w_first, w_last = to_conv1(cands[0][1]), to_conv1(cands[-1][1])

            in_c, hid_c, out_c = int(w_first.shape[1]), int(w_first.shape[0]), int(w_last.shape[0])
            num_layers = max(1, len(cands) - 1)

            # 如果没有该模块，就创建；有的话直接用
            if not hasattr(module, mlp_name):
                setattr(module, mlp_name, InputMLP3D(
                    in_channels=in_c, hidden_channels=hid_c,
                    out_channels=out_c, dropout=0.0, num_layers=num_layers
                ))

            mlp = getattr(module, mlp_name).to(device=self.device, dtype=self.torch_dtype)

            # Linear 权重转换成 Conv3d 权重
            sub_conv = {k: to_conv1(v) if (k.endswith(".weight") and is_lin(v)) else v for k, v in sub.items()}
            mlp.load_state_dict(sub_conv, strict=False)

        print("[load_lora_mlp] LoRA + input/output MLP 权重加载完成。")


    




        
    def training_loss(self, **inputs):
        # self.scheduler.training = True
        
        timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)
        gate_for_dino = gate_from_timestep(timestep, 0, 1000, beta=3.0)
        inputs["latents"] = self.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep)
        training_target = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)
        
        noise_pred, align_loss1, align_loss2 = self.model_fn(**inputs, timestep=timestep)
        
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        # print("loss",loss,"qwen_loss", qwen_loss)

        loss = loss * self.scheduler.training_weight(timestep)
        loss_mse = loss
        loss = loss + gate_for_dino*0.2*align_loss1+ 1.0*align_loss2
        # loss = losss
        # print(qwen_loss)
        # print(timestep, gate_for_dino)
        return loss,loss_mse,gate_for_dino*0.2*align_loss1, 1.0*align_loss2


    def enable_vram_management(self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5):
        self.vram_management_enabled = True
        if num_persistent_param_in_dit is not None:
            vram_limit = None
        else:
            if vram_limit is None:
                vram_limit = self.get_vram()
            vram_limit = vram_limit - vram_buffer
        if self.text_encoder is not None:
            dtype = next(iter(self.text_encoder.parameters())).dtype
            enable_vram_management(
                self.text_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    T5RelativeEmbedding: AutoWrappedModule,
                    T5LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit is not None:
            dtype = next(iter(self.dit.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.vae is not None:
            dtype = next(iter(self.vae.parameters())).dtype
            enable_vram_management(
                self.vae,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    RMS_norm: AutoWrappedModule,
                    CausalConv3d: AutoWrappedModule,
                    Upsample: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.Dropout: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.motion_controller is not None:
            dtype = next(iter(self.motion_controller.parameters())).dtype
            enable_vram_management(
                self.motion_controller,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.vace is not None:
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.vace,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    RMSNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
            
            
    def initialize_usp(self):
        import torch.distributed as dist
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
        dist.init_process_group(backend="nccl", init_method="env://")
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )
        torch.cuda.set_device(dist.get_rank())
            
            
    def enable_usp(self):
        from xfuser.core.distributed import get_sequence_parallel_world_size
        from ..distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True


    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="google/*"),
    #     tokenizer_config: ModelConfig = ModelConfig(
    #     path="/ssd3/vis/zhangxinyao/video/DiffSynthStudio/models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl",
    #     skip_download=True
    # ),
        local_model_path: str = "/root/paddlejob/workspace/env_run/wanglei/DiffSynthStudio/models",
        skip_download: bool = True,
        redirect_common_files: bool = True,
        use_usp=False,
        dinov3_model_id: str = "/root/paddlejob/workspace/env_run/wanglei/dinov3_weights",
        # qwen_model_id: str = "/root/paddlejob/workspace/env_run/wanglei/Qwen2.5VL",
    ):
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": "Wan-AI/Wan2.1-T2V-14B",
                "Wan2.1_VAE.pth": "Wan-AI/Wan2.1-T2V-14B",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "Wan-AI/Wan2.1-I2V-14B-480P",
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to ({redirect_dict[model_config.origin_file_pattern]}, {model_config.origin_file_pattern}). You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern]
        
        # Initialize pipeline
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        if use_usp: pipe.initialize_usp()
        
        # Download and load models
        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.download_if_necessary(local_model_path, skip_download=skip_download, use_usp=use_usp)
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype
            )
        
        # Load models
        pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")
        pipe.dit = model_manager.fetch_model("wan_video_dit")
        pipe.vae = model_manager.fetch_model("wan_video_vae")
        pipe.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        pipe.motion_controller = model_manager.fetch_model("wan_video_motion_controller")
        pipe.vace = model_manager.fetch_model("wan_video_vace")

        # Initialize tokenizer
        tokenizer_config.download_if_necessary(local_model_path, skip_download=skip_download)
        # if tokenizer_config.path is None:
        #     tokenizer_config.download_if_necessary(local_model_path, skip_download=True)
        pipe.prompter.fetch_models(pipe.text_encoder)
        # print(tokenizer_config.path)
        pipe.prompter.fetch_tokenizer(tokenizer_config.path)
        # print("Loading Qwen")
        # pipe.qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #     qwen_model_id,
        #     torch_dtype=torch.bfloat16
        #     )
        # # pipe.qwen.to(device=pipe.device, dtype=torch.bfloat16)
        # pipe.qwen.eval()
        # for p in pipe.qwen.parameters():
        #     p.requires_grad_(False)
        # pipe.qwen_processor = AutoProcessor.from_pretrained(qwen_model_id)
        # pipe.qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_id)
        # print("Loaded")
        pipe.dino = AutoModel.from_pretrained(
            dinov3_model_id, 
            torch_dtype=torch_dtype
        )
        pipe.dino_processor = AutoProcessor.from_pretrained(
            dinov3_model_id,do_resize=False,             # 关闭 resize
            do_center_crop=False,        # 关闭裁剪
            default_to_square=False      # 不强制方形
        )
        pipe.clip = Siglip2VisionModel.from_pretrained("/root/paddlejob/workspace/env_run/wanglei/siglip2-base-patch16-naflex").eval()
        pipe.clip_processor = AutoImageProcessor.from_pretrained("/root/paddlejob/workspace/env_run/wanglei/siglip2-base-patch16-naflex",
                                                         do_resize=False,             # 关闭 resize
                                                         do_center_crop=False,        # 关闭裁剪
                                                         )

        
        # Unified Sequence Parallel
        if use_usp: pipe.enable_usp()
        return pipe


    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: list[str],
        negative_prompt: Optional[str] = "",
        # Image-to-video
        input_image: Optional[Image.Image] = None,
        # First-last-frame-to-video
        end_image: Optional[Image.Image] = None,
        # Video-to-video
        input_video: Optional[list[list[Image.Image]]] = None,
        denoising_strength: Optional[float] = 1.0,
        # ControlNet
        control_video: Optional[list[Image.Image]] = None,
        reference_image: Optional[Image.Image] = None,
        # Camera control
        camera_control_direction: Optional[Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown"]] = None,
        camera_control_speed: Optional[float] = 1/54,
        camera_control_origin: Optional[tuple] = (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        # VACE
        vace_video: Optional[list[Image.Image]] = None,
        vace_video_mask: Optional[Image.Image] = None,
        vace_reference_image: Optional[Image.Image] = None,
        vace_scale: Optional[float] = 1.0,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # Speed control
        motion_bucket_id: Optional[int] = None,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        # progress_bar
        progress_bar_cmd=tqdm,
        # multi-subject
        subject_image:Optional[list[list[Image.Image]]] = None,
    ):
        # if subject_image is not None:
            # subject_image = self.encode_subject_image(subject_image)
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "input_image": input_image,
            "end_image": end_image,
            "input_video": input_video, "denoising_strength": denoising_strength,
            "control_video": control_video, "reference_image": reference_image,
            "camera_control_direction": camera_control_direction, "camera_control_speed": camera_control_speed, "camera_control_origin": camera_control_origin,
            "vace_video": vace_video, "vace_video_mask": vace_video_mask, "vace_reference_image": vace_reference_image, "vace_scale": vace_scale,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale, "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
            "subject_image": subject_image,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            # Inference
            noise_pred_posi,_, _ = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)#i t
            # inputs_shared["subject_latents"] = None
            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    neg_inputs = dict(inputs_shared)  # 深拷贝一份
                    neg_inputs["subject_latents"] = None
                    inputs_posi_noref = dict(inputs_posi)
                    inputs_posi_noref["prompt"] = None
                    noise_pred_nega,_ ,_= self.model_fn(**models, **neg_inputs, **inputs_nega, timestep=timestep)# null
                    noise_pred_posi_noref,_,_ = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)#i
                    # noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                # noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi_noref - noise_pred_nega) + 7.5 * (noise_pred_posi - noise_pred_posi_noref)
                # noise_pred = noise_pred_posi + cfg_scale * (noise_pred_posi_noref - noise_pred_nega) + 7.5 * (noise_pred_posi - noise_pred_posi_noref)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
        
        # VACE (TODO: remove it)
        if vace_reference_image is not None:
            inputs_shared["latents"] = inputs_shared["latents"][:, :, 1:]

        # Decode
        self.load_models_to_device(['vae'])
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video = self.vae_output_to_video(video)
        self.load_models_to_device([])

        return video



class PipelineUnit:
    def __init__(
        self,
        seperate_cfg: bool = False,
        take_over: bool = False,
        input_params: tuple[str] = None,
        input_params_posi: dict[str, str] = None,
        input_params_nega: dict[str, str] = None,
        onload_model_names: tuple[str] = None
    ):
        self.seperate_cfg = seperate_cfg
        self.take_over = take_over
        self.input_params = input_params
        self.input_params_posi = input_params_posi
        self.input_params_nega = input_params_nega
        self.onload_model_names = onload_model_names


    def process(self, pipe: WanVideoPipeline, inputs: dict, positive=True, **kwargs) -> dict:
        raise NotImplementedError("`process` is not implemented.")



class PipelineUnitRunner:
    def __init__(self):
        pass

    def __call__(self, unit: PipelineUnit, pipe: WanVideoPipeline, inputs_shared: dict, inputs_posi: dict, inputs_nega: dict) -> tuple[dict, dict]:
        if unit.take_over:
            # Let the pipeline unit take over this function.
            inputs_shared, inputs_posi, inputs_nega = unit.process(pipe, inputs_shared=inputs_shared, inputs_posi=inputs_posi, inputs_nega=inputs_nega)
        elif unit.seperate_cfg:
            # Positive side
            processor_inputs = {name: inputs_posi.get(name_) for name, name_ in unit.input_params_posi.items()}
            if unit.input_params is not None:
                for name in unit.input_params:
                    processor_inputs[name] = inputs_shared.get(name)
            processor_outputs = unit.process(pipe, **processor_inputs)
            inputs_posi.update(processor_outputs)
            # Negative side
            if inputs_shared["cfg_scale"] != 1:
                processor_inputs = {name: inputs_nega.get(name_) for name, name_ in unit.input_params_nega.items()}
                if unit.input_params is not None:
                    for name in unit.input_params:
                        processor_inputs[name] = inputs_shared.get(name)
                processor_outputs = unit.process(pipe, **processor_inputs)
                inputs_nega.update(processor_outputs)
            else:
                inputs_nega.update(processor_outputs)
        else:
            processor_inputs = {name: inputs_shared.get(name) for name in unit.input_params}
            processor_outputs = unit.process(pipe, **processor_inputs)
            inputs_shared.update(processor_outputs)
        return inputs_shared, inputs_posi, inputs_nega



class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}



class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames", "seed", "rand_device", "vace_reference_image"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, seed, rand_device, vace_reference_image):
        # subject_image = self.encode_subject_image(subject_image)
        # print(subject_image.shape)
        length = (num_frames - 1) // 4 + 1
        if vace_reference_image is not None:
            length += 1
        noise = pipe.generate_noise((4, 16, length, height//8, width//8), seed=seed, rand_device=rand_device)
        if vace_reference_image is not None:
            noise = torch.concat((noise[:, :, -1:], noise[:, :, :-1]), dim=2)
        return {"noise": noise}
    


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride", "vace_reference_image"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, noise, tiled, tile_size, tile_stride, vace_reference_image):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(["vae"])
        input_video = pipe.preprocess_video(input_video)
        input_latents = pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        if vace_reference_image is not None:
            vace_reference_image = pipe.preprocess_video([vace_reference_image])
            vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
            input_latents = torch.concat([vace_reference_latents, input_latents], dim=2)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}



class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            onload_model_names=("text_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = pipe.prompter.encode_prompt(prompt, positive=positive, device=pipe.device)
        return {"context": prompt_emb}



class WanVideoUnit_ImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("image_encoder", "vae")
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device) # 1, 3, 128, 128
        clip_context = pipe.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device) # 1, 9, 128, 128
        msk[:, 1:] = 0 # 1, 9, 128, 128
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image([end_image])], dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1) # 3, 9, 128, 128

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1) #1， 12,128,128
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8) # 1,3,4，64,64
        msk = msk.transpose(1, 2)[0] # 4，3,64,64
        
        y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device) # -,3,64，,64
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context, "y": y}

        
 
class WanVideoUnit_FunControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("control_video", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride", "clip_feature", "y"),
            onload_model_names=("vae")
        )

    def process(self, pipe: WanVideoPipeline, control_video, num_frames, height, width, tiled, tile_size, tile_stride, clip_feature, y):
        if control_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        control_video = pipe.preprocess_video(control_video)
        control_latents = pipe.vae.encode(control_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        control_latents = control_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        if clip_feature is None or y is None:
            clip_feature = torch.zeros((1, 257, 1280), dtype=pipe.torch_dtype, device=pipe.device)
            y = torch.zeros((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), dtype=pipe.torch_dtype, device=pipe.device)
        else:
            y = y[:, -16:]
        y = torch.concat([control_latents, y], dim=1)
        return {"clip_feature": clip_feature, "y": y}
    


class WanVideoUnit_FunReference(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("reference_image", "height", "width", "reference_image"),
            onload_model_names=("vae")
        )

    def process(self, pipe: WanVideoPipeline, reference_image, height, width):
        if reference_image is None:
            return {}
        pipe.load_models_to_device(["vae"])
        reference_image = reference_image.resize((width, height))
        reference_latents = pipe.preprocess_video([reference_image])
        reference_latents = pipe.vae.encode(reference_latents, device=pipe.device)
        clip_feature = pipe.preprocess_image(reference_image)
        clip_feature = pipe.image_encoder.encode_image([clip_feature])
        return {"reference_latents": reference_latents, "clip_feature": clip_feature}


import matplotlib.pyplot as plt
import numpy as np

def visualize_qwen_energy(ref_qwen_hw_nchw, pil_image, save_path, alpha=0.15):
    """
    ref_qwen_hw_nchw: [B, Cq, Hm, Wm] (已是 30x52)
    pil_image: 当前循环里的原始 PIL.Image（已 resize 到 width×height）
    """
    with torch.no_grad():
        # 1) 通道聚合成能量图: [B, Hm, Wm]
        energy = torch.norm(ref_qwen_hw_nchw[0], p=2, dim=0, keepdim=False)  # [Hm, Wm]
        # 2) 归一化
        e = energy
        e = (e - e.min()) / (e.max() - e.min() + 1e-6)                       # [Hm, Wm]
        # 3) 上采样到原图大小（H, W）
        H, W = pil_image.size[1], pil_image.size[0]
        e = e.detach().to(device="cpu", dtype=torch.float32)
        e_up = F.interpolate(e.unsqueeze(0).unsqueeze(0), size=(H, W),
                             mode="bilinear", align_corners=False)[0,0].cpu().numpy()

        # 4) 可视化叠加
        plt.figure(figsize=(6, 4))
        plt.imshow(pil_image)
        plt.imshow(e_up, cmap='jet', alpha=alpha)  # 半透明热力图
        plt.axis('off'); plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

def gaussian_weights(num_layers=12, mu=9, sigma=2):
    layers = torch.arange(1, num_layers + 1, dtype=torch.float32)
    weights = torch.exp(-0.5 * ((layers - mu) / sigma) ** 2)
    weights = weights / weights.sum()
    return weights

def fuse_dino_gaussian(hidden_states, mu=9, sigma=2):
    w = gaussian_weights(len(hidden_states), mu, sigma).to(hidden_states[0].device)
    fused = sum(w[i] * hidden_states[i] for i in range(len(hidden_states)))
    return fused  # [B, L, D]

# class WanVideoUnit_Subject(PipelineUnit):
#     def __init__(self):
#         super().__init__(
#             input_params=("subject_image", "height", "width", "subject_image"),
#             onload_model_names=("vae", "dino")
#         )

#     def process(self, pipe: WanVideoPipeline, subject_image, height, width):
#         if subject_image is None:
#             return {}
#         pipe.load_models_to_device(["vae"])
#         images = [image.resize((width, height)) for image in subject_image]
#         ref_list = []
#         # ref_qwen_list = []
#         ref_qwen_vis_list = []
#         ref_qwen = None
#         ref_qwen_vis = None
#         for image in images:
#             # print(image)
#             reference_latents = pipe.preprocess_video([image])
#             # qproc = pipe.qwen_processor
#             qdev = next(pipe.dino.parameters()).device
#             # merge = getattr(qproc.image_processor, "merge_size", 2)
#             use_dino = bool(getattr(pipe.scheduler, "training", False) and pipe.training)
#             # use_dino = True
#             # print(use_qwen)
#             dino_model = pipe.dino
#             processor = pipe.dino_processor
            
#             if use_dino:
#                 dino_features = []
#                 inputs = processor(images=image, return_tensors="pt")
#                 inputs = inputs.to(qdev)
#                 with torch.inference_mode():
#                     outputs = dino_model(**inputs, output_hidden_states=True)
#                     hidden_states = outputs.hidden_states[-1][:, 5:].view(1, 30, 52, 768).permute(0, 3, 1, 2).contiguous().unsqueeze(2)
#                 # for h in hidden_states:
#                 #     feat = h[:, 5:].view(1, 30, 52, 768).permute(0, 3, 1, 2).contiguous().unsqueeze(2).unsqueeze(1)  # [1, 1, 768, 1, 30, 52]
#                 #     dino_features.append(feat)
#                 # dino_feature = 0.95*fuse_dino_gaussian(dino_features, mu=1.0, sigma=2.0) + 0.05*fuse_dino_gaussian(dino_features, mu=9.0, sigma=1.0)
                
#                 dino_feature = hidden_states

#                 ref_qwen_vis_list.append(dino_feature)
               
#             reference_latents = pipe.vae.encode(reference_latents, device=pipe.device)
#             ref_list.append(reference_latents)
#         reference_latents = torch.concat(ref_list, dim=2)
#         if use_dino:
#             ref_qwen_vis = torch.concat(ref_qwen_vis_list, dim=2)
#             ref_qwen_vis = ref_qwen_vis.detach()
#             # print(ref_qwen_vis.shape)
#             # assert 2==1
#         return {"subject_latents": reference_latents, "ref_qwen_latents": ref_qwen_vis}


class WanVideoUnit_Subject(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("subject_image", "height", "width", "subject_image"),
            onload_model_names=("vae", "dino")
        )

    def process(self, pipe: WanVideoPipeline, subject_image, height, width):
        """
        subject_image 支持两种形式：
          1) 单样本：List[PIL.Image]
          2) Batch：List[List[PIL.Image]]，外层是 batch，内层是该样本的多张参考图
        返回:
          subject_latents: [B, C, N_ref, H, W]
          ref_qwen_latents: [B, C_dino, N_ref, H_dino, W_dino] 或 None
        """
        if subject_image is None:
            return {}

        pipe.load_models_to_device(["vae"])

        # -------- 1. 统一成 batch: List[List[Image]] --------
        # 如果是单样本 List[Image]，包一层变成 batch_size=1
        if isinstance(subject_image[0], Image.Image):
            batch_subject_images = [subject_image]
        else:
            # 已经是 List[List[Image]]
            batch_subject_images = subject_image

        use_dino = bool(getattr(pipe.scheduler, "training", False) and pipe.training)
        if use_dino:
            pipe.load_models_to_device(["dino"])
            dino_model = pipe.dino
            processor = pipe.dino_processor
            qdev = next(dino_model.parameters()).device

        batch_ref_latents = []
        batch_ref_qwen_vis = [] if use_dino else None

        # -------- 2. 遍历 batch 中的每个样本 --------
        for images in batch_subject_images:
            # 每个样本里的一组参考图（你前面 collate 已经 pad 到 3 张）
            # 统一 resize 到 (width, height)
            images = [im.resize((width, height)) for im in images]

            ref_list = []
            ref_qwen_vis_list = [] if use_dino else None

            for image in images:
                # (a) VAE 参考 latent
                # preprocess_video 接受 List[Image]，单帧视频 -> [1, C, 1, H, W]
                reference_latents = pipe.preprocess_video([image])
                reference_latents = pipe.vae.encode(reference_latents, device=pipe.device)
                ref_list.append(reference_latents)  # 每张: [1, C, 1, H, W]

                # (b) DINO 视觉特征（可选）
                if use_dino:
                    inputs = processor(images=image, return_tensors="pt").to(qdev)
                    with torch.inference_mode():
                        outputs = dino_model(**inputs, output_hidden_states=True)
                        # hidden_states: 取最后一层，去掉 CLS 等前 5 个 token
                        hidden_states = outputs.hidden_states[-1][:, 5:]   # [1, N, C]
                        # 你原来的 reshape: [1, 30, 52, 768] -> [1, 768, 30, 52] -> [1, 768, 1, 30, 52]
                        hidden_states = (
                            hidden_states
                            .view(1, 30, 52, 768)
                            .permute(0, 3, 1, 2)
                            .contiguous()
                            .unsqueeze(2)  # 在 "N_ref" 维度上占 1
                        )  # [1, 768, 1, 30, 52]
                    ref_qwen_vis_list.append(hidden_states)

            # (c) 把同一条样本的多张 subject_image 拼到 N_ref 维上
            # ref_list 里每个: [1, C, 1, H, W] -> cat dim=2 得到 [1, C, N_ref, H, W]
            reference_latents_sample = torch.cat(ref_list, dim=2)
            batch_ref_latents.append(reference_latents_sample)

            if use_dino:
                # ref_qwen_vis_list 里每个: [1, 768, 1, 30, 52] -> [1, 768, N_ref, 30, 52]
                ref_qwen_vis_sample = torch.cat(ref_qwen_vis_list, dim=2).detach()
                batch_ref_qwen_vis.append(ref_qwen_vis_sample)

        # -------- 3. 再把多个样本在 batch 维 cat --------
        # 每个样本都是 [1, C, N_ref, H, W]，cat dim=0 -> [B, C, N_ref, H, W]
        subject_latents = torch.cat(batch_ref_latents, dim=0)

        if use_dino:
            # 每个样本 [1, C_dino, N_ref, H_dino, W_dino] -> [B, C_dino, N_ref, H_dino, W_dino]
            ref_qwen_vis = torch.cat(batch_ref_qwen_vis, dim=0)
        else:
            ref_qwen_vis = None

        return {
            "subject_latents": subject_latents,   # [B, C, N_ref, H, W]
            "ref_qwen_latents": ref_qwen_vis,     # [B, C_dino, N_ref, H_dino, W_dino] or None
        }


# class WanVideoUnit_Subject(PipelineUnit):
#     def __init__(self):
#         super().__init__(
#             input_params=("subject_image", "height", "width", "subject_image"),
#             onload_model_names=("vae", "clip")
#         )

#     def process(self, pipe: WanVideoPipeline, subject_image, height, width):
#         if subject_image is None:
#             return {}
#         pipe.load_models_to_device(["vae"])
#         images = [image.resize((width, height)) for image in subject_image]
#         ref_list = []
#         # ref_qwen_list = []
#         ref_qwen_vis_list = []
#         ref_qwen = None
#         ref_qwen_vis = None
#         for image in images:
#             # print(image)
#             reference_latents = pipe.preprocess_video([image])
#             # qproc = pipe.qwen_processor
#             qdev = next(pipe.clip.parameters()).device
#             # merge = getattr(qproc.image_processor, "merge_size", 2)
#             # use_clip = bool(getattr(pipe.scheduler, "training", False) and pipe.training)
#             use_clip = False
#             # print(use_qwen)
#             clip_model = pipe.clip
#             processor = pipe.clip_processor
            
#             if use_clip:
#                 clip_features = []
#                 proc = processor(images=image, return_tensors="pt", max_num_patches=1560)
#                 pixel_values = proc["pixel_values"].to(qdev)
#                 pixel_attention_mask = proc.get("pixel_attention_mask").to(qdev)
#                 spatial_shapes = proc.get("spatial_shapes").to(qdev)
#                 # inputs = inputs
#                 with torch.inference_mode():
#                     outputs = clip_model(pixel_values=pixel_values, output_hidden_states=True,
#                     pixel_attention_mask = pixel_attention_mask,
#                     spatial_shapes = spatial_shapes)
#                     hidden_states = outputs.hidden_states[-1].view(1, 30, 52, 768).permute(0, 3, 1, 2).contiguous().unsqueeze(2)
#                 # print(hidden_states.shape)
#                 # assert 2==1
#                     # hidden_states = outputs.hidden_states[-1][:, 5:].view(1, 30, 52, 768).permute(0, 3, 1, 2).contiguous().unsqueeze(2)
#                 # for h in hidden_states:
#                     # feat = h[:, 5:].view(1, 30, 52, 768).permute(0, 3, 1, 2).contiguous().unsqueeze(2)  # [1, 768, 1, 30, 52]
#                     # dino_features.append(feat)
#                 dino_feature = hidden_states
#                 # dino_feature = 0.95*fuse_dino_gaussian(dino_features, mu=1.0, sigma=2.0) + 0.05*fuse_dino_gaussian(dino_features, mu=9.0, sigma=1.0)
                
#                 # dino_feature = torch.cat(dino_features, dim=0).contiguous().mean(dim=0)

#                 ref_qwen_vis_list.append(dino_feature)
               
#             reference_latents = pipe.vae.encode(reference_latents, device=pipe.device)
#             ref_list.append(reference_latents)
#         reference_latents = torch.concat(ref_list, dim=2)
#         if use_clip:
#             ref_qwen_vis = torch.concat(ref_qwen_vis_list, dim=2)
#             ref_qwen_vis = ref_qwen_vis.detach()
#             # print(ref_qwen_vis.shape)
#             # assert 2==1
#         return {"subject_latents": reference_latents, "ref_qwen_latents": ref_qwen_vis}

# class WanVideoUnit_Subject(PipelineUnit):
#     def __init__(self):
#         super().__init__(
#             input_params=("subject_image", "height", "width", "subject_image", "prompt"),
#             onload_model_names=("vae", "qwen")
#         )

#     def process(self, pipe: WanVideoPipeline, prompt, subject_image, height, width):
#         if subject_image is None:
#             return {}
#         pipe.load_models_to_device(["vae", "qwen"])
#         images = [image.resize((width, height)) for image in subject_image]
#         ref_list = []
#         # ref_qwen_list = []
#         ref_qwen_vis_list = []
#         ref_qwen = None
#         ref_qwen_vis = None
#         for image in images:
#             # print(image)
#             reference_latents = pipe.preprocess_video([image])
#             qproc = pipe.qwen_processor
#             qdev = next(pipe.qwen.parameters()).device
#             merge = getattr(qproc.image_processor, "merge_size", 2)
#             use_qwen = bool(getattr(pipe.scheduler, "training", False) and pipe.training)
#             # use_qwen = True
#             # print(use_qwen)
#             if use_qwen:
#                 # user
#                 # messages = [
#                 #     {
#                 #         "role": "user",
#                 #         "content": [
#                 #             {"type": "text", "text": "Only describe the subject's intrinsic identity features, ignore transient conditions like lighting, angle, or pose."},
#                 #             {"type": "image", "image": image}
#                 #         ]
#                 #     }
#                 # ]
#                 #sys
#                 messages = [
#                     {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
#                 ]

#                 text = pipe.qwen_processor.apply_chat_template(
#                     messages, tokenize=False, add_generation_prompt=True
#                 )
#                 image_inputs, video_inputs = process_vision_info(messages)
#                 qwen_inputs = pipe.qwen_processor(
#                     text=[text],
#                     images=image_inputs,
#                     videos=video_inputs,
#                     return_tensors="pt",
#                     padding=True
#                 )
#                 qwen_inputs = {k: (v.to(qdev) if hasattr(v, "to") else v) for k, v in qwen_inputs.items()}

#                 with torch.no_grad():
#                     outputs = pipe.qwen(**qwen_inputs, output_hidden_states=True)
#                     # hidden_states[-1] = 最后一层视觉/多模态embedding
#                     # vision_embeds = outputs.hidden_states[-1].unsqueeze(1)
#                     # seq = outputs.hidden_states[-1]
#                     seq = outputs.hidden_states[-1]            # [B, L, C]
#                     # print(seq.shape)
#                 B, L, C = seq.shape

#                 # 1) 取 pre-merge 网格与 merge_size
#                 t, H, W = qwen_inputs["image_grid_thw"].view(-1, 3)[0].tolist()
#                 m = int(getattr(pipe.qwen_processor.image_processor, "merge_size", 2))
#                 Hm, Wm = H // m, W // m                     # post-merge 网格

#                 # 2) 从 tokenizer 找到 <|image_pad|> 的 id，并在 input_ids 里定位它的所有位置
#                 tok = getattr(pipe, "qwen_tokenizer", None) or pipe.qwen_processor.tokenizer
#                 img_pad_id = tok.convert_tokens_to_ids("<|image_pad|>")

#                 # 注意：input_ids 形状一般是 [B, S]；只取这一条样本
#                 input_ids = qwen_inputs["input_ids"]        # [B, S]
#                 mask_img = (input_ids == img_pad_id)        # [B, S]，True 的位置对应视觉 token
#                 # 如果你明确知道只有 1 张图，并且处理器按顺序把所有视觉 token 连续放置：
#                 sel = mask_img.nonzero(as_tuple=False)      # [N, 2], 每行 (b, s_idx)
#                 sel = sel[:, 1]                             # 取序列索引 [N]

#                 # 3) 取出对应 hidden states，应该有 N == t*Hm*Wm（这里理想是 510）
#                 vis_seq = seq.index_select(dim=1, index=sel.to(seq.device))  # [B, N, C]
#                 vis_seq = vis_seq.unsqueeze(1)
                
#                 # print(vis_seq.shape)
#                 # N = vis_seq.shape[1]
#                 # expected = t * Hm * Wm
#                 # print(expected)
#                 # print("NN",N)
#                 # if N != expected:
#                 #     # 部分版本会有极少量 pad/裁剪，做个稳妥截断
#                 #     take = min(N, expected)
#                 #     vis_seq = vis_seq[:, :take, :]
#                 #     expected = take
#                     # 也可以在这里打印提醒：print(f"[Qwen] visual tokens N={N}, expect={t*Hm*Wm}, use={take}")

#                 # 4) reshape 回 2D：B×t×Hm×Wm×C -> 单图取第0帧
#                 seq_hw = vis_seq.view(B, t, Hm, Wm, C)
#                 ref_qwen_hw = seq_hw[:, 0]                  # [B, Hm, Wm, C]
#                 # 如需 NCHW：
#                 ref_qwen_hw_nchw = ref_qwen_hw.permute(0, 3, 1, 2).contiguous()  # [B, C, Hm, Wm]
#                 ref_qwen_hw_nchw = F.interpolate(ref_qwen_hw_nchw, size=(30, 52), mode="bilinear", align_corners=False)
#                 # out_path = f"qwen_heat_{len(ref_qwen_vis_list)}.png"
#                 # visualize_qwen_energy(ref_qwen_hw_nchw, image, out_path, alpha=0.45)
#                 ref_qwen_hw_nchw = ref_qwen_hw_nchw.unsqueeze(2)
#                 # ref_qwen_hw_nchw = ref_qwen_hw_nchw.to(pipe.device)
                
#                 # ref_qwen_hw_nchw = rearrange(ref_qwen_hw_nchw, 'b c h w -> b h w c').contiguous().unsqueeze(1)
#                 ref_qwen_vis_list.append(ref_qwen_hw_nchw)
#                 # print(ref_qwen_hw_nchw.shape)
#                 # print("Qwen vision embeds:", vision_embeds.shape)
#                 # ref_qwen_list.append(vision_embeds)
#                 # print("Qwen vision embeds:", vision_embeds.shape)
#             reference_latents = pipe.vae.encode(reference_latents, device=pipe.device)
#             # print(f'{reference_latents.shape}, {reference_latents.dtype}')
#             ref_list.append(reference_latents)
#         reference_latents = torch.concat(ref_list, dim=2)
#         if use_qwen:
#             # ref_qwen = torch.concat(ref_qwen_list, dim=1)
#             ref_qwen_vis = torch.concat(ref_qwen_vis_list, dim=2)
#         # print(f'{ref_qwen.shape}, {ref_qwen.dtype}')
#         # print(f'{ref_qwen_vis.shape}, {ref_qwen_vis.dtype}')
#         # print(f'{reference_latents.shape}, {reference_latents.dtype}')
#         # try:
#         #     dev = pipe.device if hasattr(pipe, "device") else "cuda"
#         #     # subject_latents 一定有（本 unit 负责产出），也打印一下形状
#         #     ddp_debug_tensor("subject_latents(from_unit)", reference_latents, device=dev)

#         #     # ref_qwen_vis 只有 use_qwen=True 才会产出；这里打印存在性+形状
#         #     ddp_debug_tensor("ref_qwen_latents(from_unit)", ref_qwen_vis if 'ref_qwen_vis' in locals() else None, device=dev)

#         #     # 如果你希望一旦不一致就立刻失败，打开下面两行“断言一致”
#         #     # ddp_assert_all_equal_flag("has_subject_latents(from_unit)", True, device=dev)
#         #     # ddp_assert_all_equal_flag("has_ref_qwen_latents(from_unit)", ref_qwen_vis is not None if 'ref_qwen_vis' in locals() else False, device=dev)
#         # except Exception as e:
#         #     # 调试期不要中断训练
#         #     print(f"[DDP-DEBUG] exception while reporting in Subject unit: {e}", flush=True)
#             ref_qwen_vis = ref_qwen_vis.detach()
#         return {"subject_latents": reference_latents, "ref_qwen_latents": ref_qwen_vis}

# class WanVideoUnit_Subject(PipelineUnit):
#     def __init__(self):
#         super().__init__(
#             input_params=("subject_image", "height", "width", "subject_image", "prompt"),
#             onload_model_names=("vae", "qwen")
#         )

#     def process(self, pipe: WanVideoPipeline, prompt, subject_image, height, width):
#         if subject_image is None:
#             return {}
#         pipe.load_models_to_device(["vae", "qwen"])
#         images = [image.resize((width, height)) for image in subject_image]
#         ref_list = []
#         ref_qwen_vis_list = []
#         ref_qwen = None
#         ref_qwen_vis = None
#         qproc = pipe.qwen_processor
#         qdev = next(pipe.qwen.parameters()).device
#         merge = getattr(qproc.image_processor, "merge_size", 2)
#         use_qwen = bool(getattr(pipe.scheduler, "training", False) and pipe.training)
#         # use_qwen = True
#         if use_qwen:
#             contents = [
#                 {"type": "image", "image": image}
#                 for image in images
#             ]
#             contents.append({"type": "text", "text": prompt})

#             messages = [{"role": "user", "content": contents}]
#             text = pipe.qwen_processor.apply_chat_template(
#                 messages, tokenize=False, add_generation_prompt=True
#             )
#             image_inputs, video_inputs = process_vision_info(messages)
#             qwen_inputs = pipe.qwen_processor(
#                 text=[text],
#                 images=image_inputs,
#                 videos=video_inputs,
#                 return_tensors="pt",
#                 padding=True
#             )
#             qwen_inputs = {k: (v.to(qdev) if hasattr(v, "to") else v) for k, v in qwen_inputs.items()}

#             with torch.no_grad():
#                 outputs = pipe.qwen(**qwen_inputs, output_hidden_states=True)
#                 seq = outputs.hidden_states[-1]            # [B, L, C]
#             B, L, C = seq.shape
#             # print(seq.shape)
#             # 1) 取 pre-merge 网格与 merge_size
#             t, H, W = qwen_inputs["image_grid_thw"].view(-1, 3)[0].tolist()
#             # print("t",t)
#             time = len(images)
#             m = int(getattr(pipe.qwen_processor.image_processor, "merge_size", 2))
#             Hm, Wm = H // m, W // m                     # post-merge 网格

#             # 2) 从 tokenizer 找到 <|image_pad|> 的 id，并在 input_ids 里定位它的所有位置
#             tok = getattr(pipe, "qwen_tokenizer", None) or pipe.qwen_processor.tokenizer
#             img_pad_id = tok.convert_tokens_to_ids("<|image_pad|>")

#             # 注意：input_ids 形状一般是 [B, S]；只取这一条样本
#             input_ids = qwen_inputs["input_ids"]        # [B, S]
#             mask_img = (input_ids == img_pad_id)        # [B, S]，True 的位置对应视觉 token
#             # 如果你明确知道只有 1 张图，并且处理器按顺序把所有视觉 token 连续放置：
#             sel = mask_img.nonzero(as_tuple=False)      # [N, 2], 每行 (b, s_idx)
#             sel = sel[:, 1]                             # 取序列索引 [N]

#             # 3) 取出对应 hidden states，应该有 N == t*Hm*Wm（这里理想是 510）
#             vis_seq = seq.index_select(dim=1, index=sel.to(seq.device))  # [B, N, C]
#             vis_seq = vis_seq.unsqueeze(1)
#             seq_hw = vis_seq.view(B, time, Hm, Wm, C)
#             seq_nchw = seq_hw.view(B * time, Hm, Wm, C).permute(0, 3, 1, 2)

#             # 双线性插值到 30×52
#             seq_nchw_resized = F.interpolate(
#                 seq_nchw,
#                 size=(30, 52),
#                 mode="bilinear",
#                 align_corners=False,
#             )

#             # 再变回 [B, T, 30, 52, C]
#             seq_hw_resized = seq_nchw_resized.view(B, time, C, 30, 52)
#             seq_hw_resized = seq_hw_resized.permute(0, 2, 1, 3, 4).contiguous()
#             ref_qwen_vis = seq_hw_resized.detach()

#         for image in images:
#             reference_latents = pipe.preprocess_video([image])
#             reference_latents = pipe.vae.encode(reference_latents, device=pipe.device)
#             ref_list.append(reference_latents)
#         reference_latents = torch.concat(ref_list, dim=2)
#         # if use_qwen:
#         #     ref_qwen_vis = torch.concat(ref_qwen_vis_list, dim=2)
#         #     ref_qwen_vis = ref_qwen_vis.detach()
#         return {"subject_latents": reference_latents, "ref_qwen_latents": ref_qwen_vis}


class WanVideoUnit_FunCameraControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "camera_control_direction", "camera_control_speed", "camera_control_origin", "latents", "input_image")
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, camera_control_direction, camera_control_speed, camera_control_origin, latents, input_image):
        if camera_control_direction is None:
            return {}
        camera_control_plucker_embedding = pipe.dit.control_adapter.process_camera_coordinates(
            camera_control_direction, num_frames, height, width, camera_control_speed, camera_control_origin)
        
        control_camera_video = camera_control_plucker_embedding[:num_frames].permute([3, 0, 1, 2]).unsqueeze(0)
        control_camera_latents = torch.concat(
            [
                torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
                control_camera_video[:, :, 1:]
            ], dim=2
        ).transpose(1, 2)
        b, f, c, h, w = control_camera_latents.shape
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)
        control_camera_latents_input = control_camera_latents.to(device=pipe.device, dtype=pipe.torch_dtype)

        input_image = input_image.resize((width, height))
        input_latents = pipe.preprocess_video([input_image])
        input_latents = pipe.vae.encode(input_latents, device=pipe.device)
        y = torch.zeros_like(latents).to(pipe.device)
        y[:, :, :1] = input_latents
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"control_camera_latents_input": control_camera_latents_input, "y": y}



class WanVideoUnit_SpeedControl(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("motion_bucket_id",))

    def process(self, pipe: WanVideoPipeline, motion_bucket_id):
        if motion_bucket_id is None:
            return {}
        motion_bucket_id = torch.Tensor((motion_bucket_id,)).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"motion_bucket_id": motion_bucket_id}



class WanVideoUnit_VACE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("vace_video", "vace_video_mask", "vace_reference_image", "vace_scale", "height", "width", "num_frames", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        vace_video, vace_video_mask, vace_reference_image, vace_scale,
        height, width, num_frames,
        tiled, tile_size, tile_stride
    ):
        if vace_video is not None or vace_video_mask is not None or vace_reference_image is not None:
            pipe.load_models_to_device(["vae"])
            if vace_video is None:
                vace_video = torch.zeros((1, 3, num_frames, height, width), dtype=pipe.torch_dtype, device=pipe.device)
            else:
                vace_video = pipe.preprocess_video(vace_video)
            
            if vace_video_mask is None:
                vace_video_mask = torch.ones_like(vace_video)
            else:
                vace_video_mask = pipe.preprocess_video(vace_video_mask, min_value=0, max_value=1)
            
            inactive = vace_video * (1 - vace_video_mask) + 0 * vace_video_mask
            reactive = vace_video * vace_video_mask + 0 * (1 - vace_video_mask)
            inactive = pipe.vae.encode(inactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            reactive = pipe.vae.encode(reactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            vace_video_latents = torch.concat((inactive, reactive), dim=1)
            
            vace_mask_latents = rearrange(vace_video_mask[0,0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8)
            vace_mask_latents = torch.nn.functional.interpolate(vace_mask_latents, size=((vace_mask_latents.shape[2] + 3) // 4, vace_mask_latents.shape[3], vace_mask_latents.shape[4]), mode='nearest-exact')
            
            if vace_reference_image is None:
                pass
            else:
                vace_reference_image = pipe.preprocess_video([vace_reference_image])
                vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
                vace_reference_latents = torch.concat((vace_reference_latents, torch.zeros_like(vace_reference_latents)), dim=1)
                vace_video_latents = torch.concat((vace_reference_latents, vace_video_latents), dim=2)
                vace_mask_latents = torch.concat((torch.zeros_like(vace_mask_latents[:, :, :1]), vace_mask_latents), dim=2)
            
            vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)
            return {"vace_context": vace_context, "vace_scale": vace_scale}
        else:
            return {"vace_context": None, "vace_scale": vace_scale}



class WanVideoUnit_UnifiedSequenceParallel(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=())

    def process(self, pipe: WanVideoPipeline):
        if hasattr(pipe, "use_unified_sequence_parallel"):
            if pipe.use_unified_sequence_parallel:
                return {"use_unified_sequence_parallel": True}
        return {}



class WanVideoUnit_TeaCache(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            input_params_nega={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
        )

    def process(self, pipe: WanVideoPipeline, num_inference_steps, tea_cache_l1_thresh, tea_cache_model_id):
        if tea_cache_l1_thresh is None:
            return {}
        return {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id)}



class WanVideoUnit_CfgMerger(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = ["context", "clip_feature", "y", "reference_latents", "subject_latents"]

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat((tensor_shared, tensor_shared), dim=0)
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega



class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states



class TemporalTiler_BCTHW:
    def __init__(self):
        pass

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + 1) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + 1) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, T, _, _ = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        mask = repeat(t, "T -> 1 1 T 1 1")
        return mask
    
    def run(self, model_fn, sliding_window_size, sliding_window_stride, computation_device, computation_dtype, model_kwargs, tensor_names, batch_size=None):
        tensor_names = [tensor_name for tensor_name in tensor_names if model_kwargs.get(tensor_name) is not None]
        tensor_dict = {tensor_name: model_kwargs[tensor_name] for tensor_name in tensor_names}
        B, C, T, H, W = tensor_dict[tensor_names[0]].shape
        if batch_size is not None:
            B *= batch_size
        data_device, data_dtype = tensor_dict[tensor_names[0]].device, tensor_dict[tensor_names[0]].dtype
        value = torch.zeros((B, C, T, H, W), device=data_device, dtype=data_dtype)
        weight = torch.zeros((1, 1, T, 1, 1), device=data_device, dtype=data_dtype)
        for t in range(0, T, sliding_window_stride):
            if t - sliding_window_stride >= 0 and t - sliding_window_stride + sliding_window_size >= T:
                continue
            t_ = min(t + sliding_window_size, T)
            model_kwargs.update({
                tensor_name: tensor_dict[tensor_name][:, :, t: t_:, :].to(device=computation_device, dtype=computation_dtype) \
                    for tensor_name in tensor_names
            })
            model_output,_ = model_fn(**model_kwargs).to(device=data_device, dtype=data_dtype)
            mask = self.build_mask(
                model_output,
                is_bound=(t == 0, t_ == T),
                border_width=(sliding_window_size - sliding_window_stride,)
            ).to(device=data_device, dtype=data_dtype)
            value[:, :, t: t_, :, :] += model_output * mask
            weight[:, :, t: t_, :, :] += mask
        value /= weight
        model_kwargs.update(tensor_dict)
        return value


def compute_semantic_alignment_loss(ref_qwen_latents, y_ip):
    # print("y_ip",y_ip.shape)
    # print("qwen",ref_qwen_latents.shape)
    q = F.normalize(ref_qwen_latents, dim=-1)
    s = F.normalize(y_ip, dim=-1)
    # loss = F.mse_loss(s, q, reduction='mean')
    loss = (-(q * s)).sum(dim=-1).mean(dim=0) 
    # print(f"loss_cos: {loss_cos:.4f}")

    return loss

def charbonnier_dense(student, teacher, eps=1e-3):
    # student/teacher: (B, L, C)
    s = torch.nn.functional.normalize(student, dim=-1)
    t = torch.nn.functional.normalize(teacher, dim=-1)
    diff = s - t
    return torch.sqrt((diff * diff).sum(dim=-1) + eps * eps).mean()


def compute_semantic_alignment_loss_mse(
    y_ip: torch.Tensor,                 # [B, C, F, H, W]  (VAE+MLP)
    ref_qwen_latents: torch.Tensor,     # [B, C, F, H, W]  (Qwen encoder)
    normalize: bool = False,            # 如果想消掉全局尺度差异就开
    weight: torch.Tensor = None,        # 可选 [B, 1 or C, F, H, W] 主体权重/掩码
) -> torch.Tensor:
    """
    语义对齐的 MSE 版本:
      - 默认在原幅值上做 MSE（不做归一化），对幅值敏感。
      - normalize=True 时做通道范数归一化，更多关注方向/相对能量。
      - 支持空间/时序权重（如主体区域更重）。
    """
    y = y_ip
    q = ref_qwen_latents

    # 尺寸不一致时，双线性/三线性对齐到 y 的时空分辨率
    if q.shape != y.shape:
        # 假设维度都是 [B, C, F, H, W]
        mode = "trilinear" if y.dim() == 5 else "bilinear"
        align_corners = False
        q = F.interpolate(q, size=y.shape[-3:], mode=mode, align_corners=align_corners)

    if normalize:
        # 通道范数归一化，避免幅值主导
        eps = 1e-6
        y = y / (y.norm(dim=1, keepdim=True) + eps)
        q = q / (q.norm(dim=1, keepdim=True) + eps)

    if weight is None:
        loss = F.mse_loss(y, q, reduction="mean")
    else:
        # 带权 MSE（常用于只在主体处强对齐）
        # weight 形状可为 [B,1,F,H,W] 或 [B,C,F,H,W]；自动广播
        diff2 = (y - q) ** 2
        weighted = diff2 * weight
        loss = weighted.sum() / (weight.sum() + 1e-6)

    return loss


import os, math, torch, numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

@torch.no_grad()
def viz_reg_qwen_bcfhw(
    vae_feat: torch.Tensor,         # [B,C,F,H,W]
    qwen_feat: torch.Tensor,        # [B,Cq,F,H,W]
    save_dir: str = "viz_feat",
    b_idx: int = 0,
    frames: list[int] | None = None,
    max_frames: int = 8,
    proj_dim: int = 64,             # 余弦相似度用的随机正交投影维度
    eps: float = 1e-6
):
    """
    输出三类图:
      1) energy_<t>.png      : VAE/Qwen 每帧能量热力图 (通道L2)
      2) cos_<t>.png         : VAE vs Qwen 的逐点余弦相似度热力图 (随机正交投影后)
      3) energy_hist.png     : 全部已选帧的能量直方图对比
    """
    os.makedirs(save_dir, exist_ok=True)

    assert vae_feat.dim() == 5 and qwen_feat.dim() == 5, "需要 [B,C,F,H,W]"
    B, Cv, Fv, Hv, Wv = vae_feat.shape
    Bq, Cq, Fq, Hq, Wq = qwen_feat.shape
    assert b_idx < B and b_idx < Bq, "b_idx 越界"

    # 截取 batch
    V = vae_feat[b_idx].detach().float().cpu()   # [Cv,F,H,W]
    Q = qwen_feat[b_idx].detach().float().cpu()  # [Cq,F,H,W]

    # 帧对齐（取公共最小 F）
    Fmin = min(V.shape[1], Q.shape[1])
    V = V[:, :Fmin]      # [Cv,F,H,W]
    Q = Q[:, :Fmin]      # [Cq,F,H,W]

    # 空间对齐（如分辨率不同，双线性插值到一致）
    if V.shape[2:] != Q.shape[2:]:
        Q = F.interpolate(Q.unsqueeze(0), size=V.shape[2:], mode="bilinear", align_corners=False).squeeze(0)

    # 选择要画的帧
    all_frames = list(range(Fmin)) if frames is None else [f for f in frames if 0 <= f < Fmin]
    if frames is None and len(all_frames) > max_frames:
        all_frames = all_frames[:max_frames]

    # ---------- 1) 能量热力图（通道L2） ----------
    def energy_map(X):  # X: [C,F,H,W] -> [F,H,W]
        e = torch.linalg.vector_norm(X, ord=2, dim=0)  # 按通道
        # 归一化到 0-1，便于可视化
        e = (e - e.min()) / (e.max() - e.min() + eps)
        return e

    E_v = energy_map(V)   # [F,H,W]
    E_q = energy_map(Q)   # [F,H,W]

    # 画每帧：上行VAE能量，下行Qwen能量
    for t in all_frames:
        fig, axes = plt.subplots(2, 1, figsize=(6, 6))
        axes[0].imshow(E_v[t].numpy(), cmap="viridis")
        axes[0].set_title(f"VAE energy (t={t})"); axes[0].axis("off")
        axes[1].imshow(E_q[t].numpy(), cmap="magma")
        axes[1].set_title(f"Qwen energy (t={t})"); axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"energy_{t:03d}.png"), dpi=200)
        plt.close(fig)

    # ---------- 2) 余弦相似度热力图（随机正交投影后逐点 cos） ----------
    # 将 [C,F,H,W] -> [F,H,W,C] 再把 C 投影到同一维度
    def rand_orth_proj(in_dim, out_dim, seed=1234):
        rng = np.random.default_rng(seed)
        A = rng.standard_normal(size=(in_dim, out_dim)).astype(np.float32)
        # 简单QR正交化
        Qm, _ = np.linalg.qr(A)
        return torch.from_numpy(Qm[:, :out_dim])  # [in_dim, out_dim]

    Pv = rand_orth_proj(V.shape[0], proj_dim)   # [Cv, d]
    Pq = rand_orth_proj(Q.shape[0], proj_dim)   # [Cq, d]

    # [F,H,W,C]
    V_whc = V.permute(1, 2, 3, 0).contiguous().view(Fmin, Hv, Wv, V.shape[0])
    Q_whc = Q.permute(1, 2, 3, 0).contiguous().view(Fmin, Hv, Wv, Q.shape[0])

    Vp = torch.tensordot(V_whc, Pv, dims=([3],[0]))   # [F,H,W,d]
    Qp = torch.tensordot(Q_whc, Pq, dims=([3],[0]))   # [F,H,W,d]

    # 逐点余弦：<v,q>/||v||/||q||
    def cos_map(A, B):  # [F,H,W,d]
        A_n = A / (A.norm(dim=-1, keepdim=True) + eps)
        B_n = B / (B.norm(dim=-1, keepdim=True) + eps)
        return (A_n * B_n).sum(dim=-1)  # [F,H,W] in [-1,1]

    Cmap = cos_map(Vp, Qp)  # [-1,1]
    # 归一化到 [0,1] 仅用于展示
    Cmap_vis = (Cmap + 1.0) / 2.0

    for t in all_frames:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        im = ax.imshow(Cmap_vis[t].numpy(), cmap="coolwarm", vmin=0, vmax=1)
        ax.set_title(f"Cosine map after random orthogonal proj (t={t})"); ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"cos_{t:03d}.png"), dpi=200)
        plt.close(fig)

    # ---------- 3) 能量直方图（看分布差异） ----------
    # 只对已选帧累积
    vv = torch.cat([E_v[t].reshape(-1) for t in all_frames], dim=0).numpy()
    qq = torch.cat([E_q[t].reshape(-1) for t in all_frames], dim=0).numpy()

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.hist(vv, bins=50, alpha=0.6, label="VAE energy")
    ax.hist(qq, bins=50, alpha=0.6, label="Qwen energy")
    ax.set_title("Energy hist (selected frames)")
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "energy_hist.png"), dpi=200)
    plt.close(fig)

    print(f"[viz] Saved to: {os.path.abspath(save_dir)}")

from torch import Tensor

def gumbel_sigmoid(logits: Tensor, tau: float = 1, hard: bool = False, threshold: float = 0.5) -> Tensor:
    """
    Samples from the Gumbel-Sigmoid distribution and optionally discretizes.
    The discretization converts the values greater than `threshold` to 1 and the rest to 0.
    The code is adapted from the official PyTorch implementation of gumbel_softmax:
    https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized,
            but will be differentiated as if it is the soft sample in autograd
     threshold: threshold for the discretization,
                values greater than this will be set to 1 and the rest to 0

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Sigmoid distribution.
      If ``hard=True``, the returned samples are descretized according to `threshold`, otherwise they will
      be probability distributions.

    """
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0, 1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits, tau)
    y_soft = gumbels.sigmoid()

    if hard:
        # Straight through.
        # indices = (y_soft > threshold).nonzero(as_tuple=True)
        # y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        # y_hard[indices[0], indices[1]] = 1.0
        y_hard = torch.zeros_like(logits)
        y_hard[y_soft > threshold] = 1.0
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

import math
import matplotlib.pyplot as plt
import torch
import numpy as np

def _to01(x: torch.Tensor):
    x = x.float()
    x = x - x.min()
    den = (x.max() - x.min()).clamp_min(1e-6)
    return (x / den).clamp(0, 1)

@torch.no_grad()
def viz_many_frames_pairs(regular_vae, mask, frames=None, out="viz_pairs.png",
                          cols=6, energy="l2"):
    """
    regular_vae: [B, C, F, H, W]
    mask:        [B, 1, F, H, W] (二值/概率均可；这里按灰度显示)
    frames:      要展示的帧索引列表；None=全部
    cols:        一行展示多少“帧对”(左VAE、右Mask)
    """
    assert regular_vae.ndim == 5 and mask.ndim == 5, "shape 必须是 [B,C,F,H,W] / [B,1,F,H,W]"
    B, C, F, H, W = regular_vae.shape
    assert B == 1, "当前函数默认 batch=1，可按需改造"

    if frames is None:
        frames = list(range(F))
    pairs_per_row = max(1, int(cols))
    rows = 2 * math.ceil(len(frames) / pairs_per_row)
    total_cols = min(pairs_per_row, len(frames))


    fig, axes = plt.subplots(rows, total_cols, figsize=(3*total_cols, 3*rows))
    axes = np.atleast_2d(axes)  # 统一成 2D 索引

    for i, f in enumerate(frames):
        r_pair = i // pairs_per_row
        c = i % total_cols
        r_vae = 2 * r_pair
        r_msk = r_vae + 1

        vae_f = regular_vae[0, :, f]                    # [C,H,W]
        if energy == "l2":
            heat = torch.linalg.vector_norm(vae_f, dim=0)  # [H,W]
        else:
            heat = vae_f.abs().mean(0)
        heat = _to01(heat).cpu().numpy()

        m = mask[0, 0, f].float().cpu().numpy()         # [H,W]

        ax1 = axes[r_vae, c]     # 上：VAE
        ax2 = axes[r_msk, c]     # 下：Mask

        ax1.imshow(heat, cmap="viridis", interpolation="nearest")
        ax1.set_title(f"f={f} • VAE energy")
        ax1.axis("off")

        ax2.imshow(m, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        # 边界更清晰一点（可选）
        try:
            ax2.contour(m, levels=[0.5], colors="red", linewidths=1.0)
        except Exception:
            pass
        ax2.set_title(f"f={f} • Mask")
        ax2.axis("off")

    # # 把多余空格子隐藏
    # filled_cols = 2 * (len(frames) % pairs_per_row if len(frames) % pairs_per_row != 0 else pairs_per_row)
    # for r in range(rows):
    #     start = filled_cols if r == rows - 1 and len(frames) % pairs_per_row != 0 else total_cols
    #     for c in range(start, total_cols):
    #         axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[saved] {out}")


# def gate_from_timestep(t, t_first, t_last, beta=1.0, descending=True):
#     """
#     t: 当前整数时间步（或张量）
#     t_first: 序列第一个步（通常高噪声，比如 999）
#     t_last:  序列最后一个步（通常低噪声，比如 0）
#     descending: timesteps 是否降序
#     """
#     if descending:
#         # t 越靠前越高噪声 → 归一化后越大
#         x = (t - t_last) / (t_first - t_last + 1e-8)
#     else:
#         x = (t - t_first) / (t_last - t_first + 1e-8)
#     x = x.clamp(0, 1)
#     gate01 = (1.0 - x).pow(beta)     # [0,1]
#     gate = 0.5 + 0.5 * gate01        # 线性映射到 [0.1,1.0]
#     return gate





def model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents = None,
    vace_context = None,
    vace_scale = 1.0,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input = None,
    subject_latents=None,
    use_input_mlp: bool = True,
    ref_qwen_latents = None,
    # training: bool = True,
    **kwargs,
):
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            motion_controller=motion_controller,
            vace=vace,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            reference_latents=reference_latents,
            vace_context=vace_context,
            vace_scale=vace_scale,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            motion_bucket_id=motion_bucket_id,
            subject_latents=subject_latents,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video,
            sliding_window_size, sliding_window_stride,
            latents.device, latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1
        )
    
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)
    x_ip = None
    t_mod_ip = None
    cond_size = None
    image_num= None
    ####inference
    align_loss1 = 0.0
    align_loss2 = 0.0
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    # print(t_mod.shape)
    # print(t_mod.shape)
    if subject_latents is not None:
        timestep_ip = torch.zeros_like(timestep)  # [B] with 0s
        t_ip = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep_ip))
        t_mod_ip = dit.time_projection(t_ip).unflatten(1, (6, dit.dim))
        # t_mod = t_mod + t_mod_ip
        
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    
    context = dit.text_embedding(context)
    # print(context.shape)
    # assert 2==1

    x = latents
    ### add subject
    # print(x.shape)
    
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)
    
    if dit.has_image_input:
        x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
    
    # Add camera control
    x, (f, h, w) = dit.patchify(x, control_camera_latents_input)
    # print(x.shape)

    # print(x.shape)
    # Reference image
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1


    offset = 3
    freqs = (
        torch.cat(
            [
                dit.freqs[0][offset : f + offset].view(f, 1, 1, -1).expand(f, h, w, -1),
                dit.freqs[1][offset:h + offset].view(1, h, 1, -1).expand(f, h, w, -1),
                dit.freqs[2][offset:w+offset].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        )
        .reshape(f * h * w, 1, -1)
        .to(x.device)
    )
    freqs_original = freqs
    res_vae = None
    flag = False
    if subject_latents is not None:
        keep = gate_from_timestep(timestep, 999, 0)
        
        # vals = torch.unique(F.dropout(torch.ones_like(subject_latents[:, :1]), p=1 - keep.item())* keep)
        # print("unique(mask) =", vals)
        if ref_qwen_latents is not None:
            flag = True
        x_ip, (f_ip, h_ip, w_ip), y_ip = dit.patchify_ip(
            subject_latents
        )
        # print(x_ip.shape, f_ip, h_ip, w_ip)
        if use_input_mlp:
            # if not hasattr(dit, "input_mlp"):
                # in_c = y_ip.shape[1]
                # dit.input_mlp = InputMLP3D(in_channels=in_c, hidden_channels=in_c//2, out_channels=3584, dropout=0.0, num_layers=2)
                # dit.output_mlp = InputMLP3D(in_channels=3584, hidden_channels=3584//2, out_channels=in_c, dropout=0.0, num_layers=2, mask=True)
            # dit.input_mlp = dit.input_mlp.to(device=y_ip.device, dtype=y_ip.dtype)
            # dit.output_mlp = dit.output_mlp.to(device=y_ip.device, dtype=y_ip.dtype)
            # print("input_mlp.param_ids", [id(p) for p in dit.input_mlp.parameters()])
            # print("all.model.param_ids", [id(p) for p in dit.parameters()])
            # print(ref_qwen_latents.shape)
            vis_dino = ref_qwen_latents
            regular_vae = y_ip
            # y_ip = y_ip * F.dropout(torch.ones_like(y_ip[:, :1]), p=1 - keep.item())* keep
            x_ip = rearrange(y_ip, 'b c f h w -> b (f h w) c').contiguous()
            if flag:
                # print(ref_qwen_latents.shape)
                # assert 2==1
                ref_qwen_latents = rearrange(ref_qwen_latents, 'b c f h w -> (b f h w) c').contiguous()
            if ref_qwen_latents is not None:
                p = 0.1  # 10% 概率
                if random.random() < p:
                    x_ip = x_ip*0
                    ref_qwen_latents = ref_qwen_latents*0
                if random.random() < p:
                    context = context*0
            # viz_many_frames_pairs(vis_dino, vis_dino,out="grid.png", cols=6, energy="l2")
            # assert 2==1
            # viz_reg_qwen_bcfhw(regular_vae, ref_qwen_latents)
            # assert 2==1
            # res_y_ip = rearrange(y_ip, 'b c f h w -> b (f h w) c').contiguous()
            # ref_qwen_latents = ref_qwen_latents.to(device=x.device, dtype=next(dit.input_mlp.parameters()).dtype)
            # ref_qwen_latents = rearrange(ref_qwen_latents, 'b n c f h w -> b  (f h w) n c',).contiguous()
            # y_ip = (y_ip - y_ip.mean(dim=(2,3,4), keepdim=True)) / (y_ip.std(dim=(2,3,4), keepdim=True) + 1e-6)
            # x_ip = dit.input_mlp(ref_qwen_latents)
            

            # viz_reg_qwen_bcfhw(vis_dino, y_ip)
            # assert 2==1
            
            # H = y_ip.shape[3]
            # W = y_ip.shape[4]
            
            # print(y_ip.shape)
            # if ref_qwen_latents is not None:
                # ref_qwen_latents = ref_qwen_latents.to(device=y_ip.device, dtype=y_ip.dtype)
            # if ref_qwen_latents is not None:
                # ref_qwen_latents = torch.linalg.vector_norm(ref_qwen_latents, dim=1, keepdim=True)
                # align_loss1 = compute_semantic_alignment_loss_mse(y_ip, ref_qwen_latents, normalize=False)
                # align_loss2 = compute_semantic_alignment_loss(y_ip, ref_qwen_latents)
                # print(align_loss)
            
            # y_ip = rearrange(y_ip, 'b c f h w -> b f (h w) c').contiguous()
            # y_ip = rearrange(y_ip, 'b f (h w) c -> b f h w c', h=H, w=W)
            # y_ip = rearrange(y_ip, 'b f h w c -> b c f h w', h=H, w=W)
            # y_ip = y_ip.detach()
            # y_id = F.dropout3d(y_ip, p=0.1, training = training)
            # y_ip = dit.output_mlp(y_ip)
            # y_ip = torch.linalg.vector_norm(ref_qwen_latents, dim=1, keepdim=True)
            # # y_ip = torch.sqrt((y_ip ** 2).mean(dim=1, keepdim=True) + 1e-6)
            # # y_ip = torch.sqrt((ref_qwen_latents ** 2).mean(dim=1, keepdim=True) + 1e-6)
            # y_ip = gumbel_sigmoid(y_ip, tau=1.0, hard=True)
            # e = torch.linalg.vector_norm(ref_qwen_latents, dim=1, keepdim=True)
            # emin = e.amin(dim=(-2,-1), keepdim=True)
            # emax = e.amax(dim=(-2,-1), keepdim=True)
            # s = ((e - emin) / (emax - emin + 1e-6)).clamp(0,1)  # 0..1

            # p = 0.3
            # th = torch.quantile(s, 1 - p, dim=(-2,-1), keepdim=True)
            # S = torch.logit(s.clamp(1e-6, 1-1e-6))
            # use_quantile = True
            # if use_quantile:
            #     # 自适应百分位阈：保留最显著的 top-p 作为前景
            #     p_keep = 0.3  # 保留 30% 高能量区域，按需调
            #     # 量化在 (F,H,W) 上独立计算阈值
            #     S_per = S.view(S.shape[0], 1, S.shape[2], -1)              # [B,1,F,H*W]
            #     th = torch.quantile(S_per.float(), q=1.0 - p_keep, dim=3, keepdim=True)  # [B,1,F,1]
            #     th = th.to(S.dtype)
            #     th = th.unsqueeze(-1)
            # else:
            #     # 固定阈
            #     th = 0.5

            # # 4) 二值化（避免 logit 爆炸，用直接比较）
            # M_hard = (S >= 0.5)   # [B,1,F,Hq,Wq]

            # print(M_hard.shape)

            # logits = logits + (math.log(p) - math.log(1-p))

            # mask = gumbel_sigmoid(ref_qwen_latents, tau=1.0, hard=True, threshold=0.7)
            # print('mask sparsity:', (M_hard == 0).float().mean().item())  # 0 的比例
            # print('mask density:', (M_hard == 1).float().mean().item())   # 1 的比例

            # y_ip = mask
            # print(y_ip.shape)
            # y_ip_vis = y_ip
            # viz_many_frames_pairs(vis_dino, mask,out="grid.png", cols=6, energy="l2")
            # assert 2==1
            # y_ip = y_ip.detach()
            # y_ip[:, :, :, :, :] = 1
            # y_ip = y_ip * regular_vae
            # if ref_qwen_latents is not None:
                # align_loss2 = compute_semantic_alignment_loss_mse(y_ip, regular_vae)
                # align_loss = align_loss1 + align_loss2
            # print("align_loss",align_loss,"align_loss2",align_loss2)
            # y_ip = M_hard*regular_vae
            
            # y_ip = x_ip * (1-y_ip)
            # res_vae = y_ip
            # y_ip = y_ip.detach()
            cond_size = x_ip.shape[1]
            # print("context shape:", context.shape)
            # print("sub shape", x_ip.shape)
            # x = torch.concat([x_ip, x], dim=1)
            freqs_ip = (
                        torch.cat(
                            [
                                dit.freqs[0][offset-f_ip:offset].view(f_ip, 1, 1, -1).expand(f_ip, h_ip, w_ip, -1),
                                dit.freqs[1][ h+ offset: h+ offset+h_ip]
                                .view(1, h_ip, 1, -1)
                                .expand(f_ip, h_ip, w_ip, -1),
                                dit.freqs[2][w+offset:w+offset+w_ip]
                                .view(1, 1, w_ip, -1)
                                .expand(f_ip, h_ip, w_ip, -1),
                            ],
                            dim=-1,
                        )
                        .reshape(f_ip * h_ip * w_ip, 1, -1)
                        .to(x_ip.device)
                    )
            freqs = torch.cat([freqs_ip,freqs], dim=0)
        # else:
            # x = torch.concat([x_ip, x], dim=1)
        # print(x.shape)
        # f+=f_ip
        image_num = f_ip
    # if subject_latents is not None:
        # offset = subject_latents.shape[2]
        # print("Subject Latents Shape: ", subject_latents.shape)


    
    
    ### spatial-temporal decoupling
    # cond_size = None
    # if subject_latents is not None:
    #     # print(subject_latents.shape) #torch.Size([1, 16, 4, 60, 104])
    #     x_ip, (f_ip, h_ip, w_ip) = dit.patchify(
    #         subject_latents
    #     )# x_ip [1,6240, 1536]BND  f_ip = 4  h_ip = 30  w_ip = 52
    #     print(x_ip.shape, f_ip, h_ip, w_ip)
    #     # print(x_ip.shape)
    #     assert offset == f_ip
    #     freqs_ip = (
    #         torch.cat(
    #             [
    #                 dit.freqs[0][0:offset].view(f_ip, 1, 1, -1).expand(f_ip, h_ip, w_ip, -1),
    #                 dit.freqs[1][h+offset : h+offset + h_ip]
    #                 .view(1, h_ip, 1, -1)
    #                 .expand(f_ip, h_ip, w_ip, -1),
    #                 dit.freqs[2][w+offset  : w+offset  + w_ip]
    #                 .view(1, 1, w_ip, -1)
    #                 .expand(f_ip, h_ip, w_ip, -1),
    #             ],
    #             dim=-1,
    #         )
    #         .reshape(f_ip * h_ip * w_ip, 1, -1)
    #         .to(x_ip.device)
    #     )
    #     # freqs_original = freqs
    #     # print(freqs.shape)
    #     # print(freqs_ip.shape)
    #     freqs = torch.cat([freqs_ip,freqs], dim=0)
    #     cond_size = x_ip.shape[1]
        # x = torch.cat([x,x_ip], dim=1)
        # f += f_ip
    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
        
    if vace_context is not None:
        vace_hints = vace(x, vace_context, context, t_mod, freqs)
    
    if flag:
        align_loss = []
    # blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        for block_id, block in enumerate(dit.blocks):
            is_first_15 = block_id < 30
            this_t_mod_ip = t_mod_ip if is_first_15 else None
            this_freqs = freqs if is_first_15 else freqs_original
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x,x_ip = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, this_freqs,x_ip, cond_size, image_num,this_t_mod_ip,timestep,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                x,x_ip = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, this_freqs, x_ip, cond_size, image_num,this_t_mod_ip,timestep,
                    use_reentrant=False,
                )
            else:
                x,x_ip = block(x, context, t_mod, this_freqs, x_ip, cond_size, image_num,this_t_mod_ip,timestep)
            if block_id < 30 and flag:
                x_ip_proj = rearrange(x_ip, 'b l c -> (b l) c').contiguous()
                x_ip_proj = dit.input_mlp(x_ip_proj)
                loss_align_layer_mean = charbonnier_dense(ref_qwen_latents, x_ip_proj)
                align_loss.append(loss_align_layer_mean)
                # x_ip = res_vae
                # x_ipy_ip
                # x_ip_vis = x_ip.view(1, 1536, image_num, 30, 52)
                # if block_id < 8:
                    # viz_many_frames_pairs(x_ip_vis, mask,out=f"grid_{block_id}.png", cols=6, energy="l2")
        if flag:
            align_loss = torch.stack(align_loss).mean()
            if vace_context is not None and block_id in vace.vace_layers_mapping:
                current_vace_hint = vace_hints[vace.vace_layers_mapping[block_id]]
                if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
                    current_vace_hint = torch.chunk(current_vace_hint, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
                x = x + current_vace_hint * vace_scale
        if tea_cache is not None:
            tea_cache.store(x)
            
    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
    
    # if subject_latents is not None:
        # x = x[:, x_ip.shape[1]:]
        # f -= f_ip
    # Remove reference latents
    if reference_latents is not None:
        x = x[:, reference_latents.shape[1]:]
        f -= 1
    x = dit.unpatchify(x, (f, h, w))
    # if ref_qwen_latents is not None and use_input_mlp:
        # return x, align_loss1, 0.1*align_loss2
    # else:
    if flag:
        return x, align_loss, 0
    return x,0,0
