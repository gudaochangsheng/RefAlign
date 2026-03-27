import torch, os, json
from diffsynth.pipelines.wan_video_new6 import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils_os2v import DiffusionTrainingModule, VideoDataset, ModelLogger, launch_training_task, wan_parser
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
from collections import defaultdict, Counter
import os
import torch.nn as nn
from diffsynth.models.set_dual_LoRA import set_unsharedLoRA
from diffsynth.models.set_input_mlp_independent import set_inputmlp



class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
    ):
        super().__init__()
        # Load models
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1]) for i in model_id_with_origin_paths]
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)
        
        # Reset training scheduler
        self.pipe.scheduler.set_timesteps(1000, training=True)
        
        # Freeze untrainable models
        self.pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        
        # Add LoRA to the base models
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(self.pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank
            )
            setattr(self.pipe, lora_base_model, model)
            
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        
        
    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        # if isinstance(data["video"][0], list):  # batch: List[List[Image]]
        #     first_frame = data["video"][0][0]
        #     video_frames = data["video"][0]
        # elif isinstance(data["video"][0], Image.Image):  # 单条: List[Image]
        #     first_frame = data["video"][0]
        #     video_frames = data["video"]
        # else:
        #     raise TypeError(f"Unexpected video type: {type(data['video'][0])}")

        # height = first_frame.size[1]
        # width = first_frame.size[0]
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
        }
        
        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss




def dump_param_indices(model):
    # 只在 rank0 打印一次即可
    import torch.distributed as dist
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    print("\n[PARAM-INDEX MAP]")
    for i, (n, p) in enumerate(model.named_parameters()):
        print(f"{i:04d} | {n} | {tuple(p.shape)} | requires_grad={p.requires_grad}", flush=True)

# 创建 model 和把 MLP 挂到 dit 上之后，立刻调用



if __name__ == "__main__":
    import os

    parser = wan_parser()
    args = parser.parse_args()
    dataset = VideoDataset(args=args)
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
    )
    # set_unsharedLoRA(model.pipe, train=True)
    ## if stage2
    set_inputmlp(model.pipe, train=True, model_path="stage1.safetenors", strict=True)
    # set_inputmlp(model.pipe, train=True)
    # for block in model.pipe.dit.blocks:
    #     block.init_inputmlp(train=True)

    
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        save_every_steps=500,
    )
    # print(model.trainable_modules)
    # for n, p in model.named_parameters():
    #     if p.requires_grad:
    #         print("TRAIN:", n, p.shape)
    # print(model.trainable_modules)
    # assert 2==1 32112
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    launch_training_task(
        dataset, model, model_logger, optimizer, scheduler,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_samples_per_epoch=3000
    )
