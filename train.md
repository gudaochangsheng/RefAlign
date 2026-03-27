# Wan2.1 T2V 训练完整指南

## 📋 目录
- [项目概述](#项目概述)
- [环境要求](#环境要求)
- [环境配置](#环境配置)
- [模型下载](#模型下载)
- [训练代码](#训练代码)
- [训练配置](#训练配置)
- [数据集准备](#数据集准备)
- [训练流程](#训练流程)
- [参考资料](#参考资料)

## 🎯 项目概述

**Wan2.1 T2V (Text-to-Video)** 是由阿里巴巴通义实验室开源的文本到视频生成模型，支持将文本描述转换为高质量视频内容。

**主要特性：**
- 🎬 文本到视频生成
- 🖼️ 支持多种分辨率 (480p, 720p, 1080p)
- 🚀 高效训练框架
- 💾 支持全量训练和 LoRA 微调
- 🔧 灵活的配置选项

## 💻 环境要求

### 硬件要求
- **GPU**: NVIDIA GPU with CUDA support
  - **1.3B 模型**: 最低 8GB VRAM
  - **14B 模型**: 最低 24GB VRAM (推荐 40GB+)
- **内存**: 最低 16GB RAM
- **存储**: 至少 50GB 可用空间

### 软件要求
- **操作系统**: Linux (推荐 Ubuntu 20.04+) / Windows 11 / macOS
- **Python**: 3.10.11
- **CUDA**: 12.4
- **PyTorch**: 2.5.1

## 🛠️ 环境配置

### 1. 基础环境安装

```bash
# 克隆项目
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio

# 创建虚拟环境
conda create -n wan_training python=3.10
conda activate wan_training

# 安装 PyTorch (根据你的 CUDA 版本选择)
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装 DiffSynth-Studio
pip install -e .
```

### 2. 依赖包安装

```bash
# 安装必要的依赖
pip install accelerate transformers diffusers
pip install safetensors pillow opencv-python
pip install wandb tensorboard

# 安装 accelerate 配置
accelerate config
```

### 3. 环境验证

```bash
# 验证 PyTorch 安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# 验证 DiffSynth-Studio 安装
python -c "from diffsynth.pipelines.wan_video_new import WanVideoPipeline; print('DiffSynth-Studio 安装成功!')"
```

## 📥 模型下载

### 模型地址

| 模型名称 | ModelScope | HuggingFace | 大小 | 说明 |
|---------|------------|-------------|------|------|
| **Wan2.1-T2V-1.3B** | [下载链接](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B) | [下载链接](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) | ~2.6GB | 轻量级模型，适合入门 |
| **Wan2.1-T2V-14B** | [下载链接](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-14B) | [下载链接](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) | ~28GB | 高质量模型，需要更多显存 |

### 模型文件结构

```
Wan2.1-T2V-1.3B/
├── diffusion_pytorch_model.safetensors    # 扩散模型权重
├── models_t5_umt5-xxl-enc-bf16.pth       # T5 文本编码器
├── Wan2.1_VAE.pth                        # VAE 编码器
├── config.json                            # 模型配置
└── tokenizer/                             # 分词器
```

### 下载命令

```bash
# 使用 git-lfs 下载 (推荐)
git lfs install
git clone https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B

# 或使用 modelscope-cli
pip install modelscope
modelscope download Wan-AI/Wan2.1-T2V-1.3B
```

## 💻 训练代码

### 核心训练脚本

**文件位置**: `examples/wanvideo/model_training/train.py`

**主要类**: `WanTrainingModule`

```python
class WanTrainingModule(DiffusionTrainingModule):
    def __init__(self, model_paths=None, model_id_with_origin_paths=None, 
                 trainable_models=None, lora_base_model=None, 
                 lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32,
                 use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
                 extra_inputs=None, max_timestep_boundary=1.0, min_timestep_boundary=0.0):
        # 初始化训练模块
        super().__init__()
        
        # 加载模型配置
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1]) for i in model_id_with_origin_paths]
        
        # 创建管道
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16, 
            device="cpu", 
            model_configs=model_configs
        )
        
        # 设置训练调度器
        self.pipe.scheduler.set_timesteps(1000, training=True)
        
        # 冻结不可训练模型
        self.pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        
        # 添加 LoRA (如果指定)
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(self.pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank
            )
            setattr(self.pipe, lora_base_model, model)
```

### 前向传播处理

```python
def forward_preprocess(self, data):
    # CFG 敏感参数
    inputs_posi = {"prompt": data["prompt"]}
    inputs_nega = {}
    
    # CFG 不敏感参数
    inputs_shared = {
        "input_video": data["video"],
        "height": data["video"][0].size[1],
        "width": data["video"][0].size[0],
        "num_frames": len(data["video"]),
        "cfg_scale": 1,
        "tiled": False,
        "rand_device": self.pipe.device,
        "use_gradient_checkpointing": self.use_gradient_checkpointing,
        "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        "cfg_merge": False,
        "vace_scale": 1,
        "max_timestep_boundary": self.max_timestep_boundary,
        "min_timestep_boundary": self.min_timestep_boundary,
    }
    
    # 处理额外输入
    for extra_input in self.extra_inputs:
        if extra_input == "input_image":
            inputs_shared["input_image"] = data["video"][0]
        elif extra_input == "end_image":
            inputs_shared["end_image"] = data["video"][-1]
        # ... 其他输入处理
    
    return {**inputs_shared, **inputs_posi}
```

## ⚙️ 训练配置

### 1.3B 模型训练脚本

**文件**: `examples/wanvideo/model_training/full/Wan2.1-T2V-1.3B.sh`

```bash
## full+本地模型路径
accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path data/example_video_dataset \
  --dataset_metadata_path data/example_video_dataset/metadata.csv \
  --height 480 \
  --width 832 \
  --dataset_repeat 100 \
  --model_paths '[
      [
          "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
      ],
      "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
      "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
  ]' \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-T2V-1.3B_full" \
  --trainable_models "dit"


#--model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth" \
```

### 14B 模型训练脚本

**文件**: `examples/wanvideo/model_training/full/Wan2.1-T2V-14B.sh`

```bash
##本地模型路径
accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path data/example_video_dataset \
  --dataset_metadata_path data/example_video_dataset/metadata.csv \
  --height 480 \
  --width 832 \
  --dataset_repeat 100 \
  --model_paths '[
      [
          "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
      ],
      "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
      "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
  ]' \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-T2V-14B_full" \
  --trainable_models "dit"

## --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-14B:Wan2.1_VAE.pth" \
```

### Accelerate 配置 (14B 模型)

**文件**: `examples/wanvideo/model_training/full/accelerate_config_14B.yaml`

```yaml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_accumulation_steps: 1
  offload_optimizer_device: cpu
  offload_param_device: cpu
  zero3_init_flag: false
  zero_stage: 2
distributed_type: DEEPSPEED
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

### 训练参数说明

| 参数 | 说明 | 默认值 | 建议值 |
|------|------|--------|--------|
| `--dataset_base_path` | 数据集根目录 | - | 必需 |
| `--dataset_metadata_path` | 数据集元数据文件 | - | 必需 |
| `--height` | 视频高度 | - | 480/720/1080 |
| `--width` | 视频宽度 | - | 832/1280/1920 |
| `--dataset_repeat` | 数据集重复次数 | - | 100 |
| `--learning_rate` | 学习率 | - | 1e-5 |
| `--num_epochs` | 训练轮数 | - | 2-10 |
| `--trainable_models` | 可训练模型组件 | - | "dit" |
| `--output_path` | 输出路径 | - | 必需 |

## 📊 数据集准备

### 数据集结构

```
data/example_video_dataset/
├── metadata.csv                    # 数据集元数据
├── video_001.mp4                  # 视频文件目录
├── video_002.mp4                       
├── ...
```

### 元数据格式

**CSV 文件示例**:

```csv
video,prompt
train_000000.mp4,A ruined land
train_000001.mp4,Mexico city - circa 1973: food stand in the street in 1973 in mexico city
train_000002.mp4,Airplane landing chepstow
```

**字段说明**:
- `prompt`: 视频描述文本
- `video_path`: 视频文件相对路径

<!-- ### 视频要求

- **格式**: MP4, AVI, MOV 等常见格式
- **分辨率**: 建议与训练配置一致 (480p, 720p, 1080p)
- **帧率**: 15-30 FPS
- **时长**: 建议 3-10 秒
- **质量**: 清晰、无压缩伪影 -->

## 🚀 训练流程

### 1. 环境准备

```bash
# 激活环境
conda activate wan_training

# 进入项目目录
cd DiffSynth-Studio
```

### 2. 数据集准备

```bash
# 创建数据集目录
mkdir -p data/example_video_dataset/videos
mkdir -p data/example_video_dataset/images

# 复制视频文件到 videos 目录
# 创建 metadata.csv 文件
```

### 3. 模型下载

```bash
# 下载模型文件
git clone https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B
```

### 4. 开始训练

```bash
# 1.3B 模型训练
bash examples/wanvideo/model_training/full/Wan2.1-T2V-1.3B.sh

# 14B 模型训练
bash examples/wanvideo/model_training/full/Wan2.1-T2V-14B.sh
```

### 5. 监控训练

```bash
# 查看训练日志
tail -f models/train/Wan2.1-T2V-1.3B_full/logs.txt

# 启动 TensorBoard
tensorboard --logdir models/train/Wan2.1-T2V-1.3B_full
```

## 🔧 训练优化

### 显存优化

```bash
# 启用梯度检查点
--use_gradient_checkpointing

# 启用梯度检查点卸载
--use_gradient_checkpointing_offload

# 使用 DeepSpeed ZeRO-2
# 在 accelerate_config_14B.yaml 中配置
```

### 训练策略

```bash
# 降低学习率
--learning_rate 5e-6

# 增加数据集重复
--dataset_repeat 200

# 调整训练轮数
--num_epochs 5
```



## 📚 参考资料

### 官方文档
- [DiffSynth-Studio GitHub](https://github.com/modelscope/DiffSynth-Studio)
- [Wan 模型文档](https://github.com/modelscope/DiffSynth-Studio/tree/main/examples/wanvideo)

### 模型地址
- [ModelScope](https://www.modelscope.cn/models?search=Wan2.1)
- [HuggingFace](https://huggingface.co/Wan-AI)

### 相关论文
- [Wan 2.1 技术报告](https://arxiv.org/abs/2401.13658)
- [DiffSynth 框架论文](https://arxiv.org/abs/2308.03463)

### 社区资源
- [魔搭社区](https://www.modelscope.cn/)
- [通义实验室](https://github.com/alibaba)

## 📝 更新日志

- **2025-01-XX**: 创建初始版本
- **2025-01-XX**: 添加训练配置说明
- **2025-01-XX**: 完善环境配置指南

---

**注意**: 本指南基于 DiffSynth-Studio 最新版本编写，如有更新请参考官方文档。

**贡献**: 欢迎提交 Issue 和 Pull Request 来改进本指南。

**许可证**: 本指南遵循 Apache 2.0 许可证。
