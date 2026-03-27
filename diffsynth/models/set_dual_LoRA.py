import torch
from safetensors.torch import load_file as safe_load_file

def set_unsharedLoRA(pipe, train=False, model_path=None):
    for block in pipe.dit.blocks:
        block.self_attn.init_lora(train)
    if model_path is not None:
        print(f"Loading Dual LoRA weights from: {model_path}")
        load_lora_weights_into_pipe(pipe, model_path)


def load_lora_weights_into_pipe(pipe, ckpt_path, strict=False):
    ckpt = safe_load_file(ckpt_path)                # 直接读取 .safetensors
    state_dict = ckpt.get("state_dict", ckpt)       # 兼容包含 state_dict 的情况


    model = {}
    for i, block in enumerate(pipe.dit.blocks):
        prefix = f"blocks.{i}.self_attn."
        attn = block.self_attn
        for name in ["q_loras", "k_loras", "v_loras"]:
            for sub in ["down", "up"]:
                key = f"{prefix}{name}.{sub}.weight"
                if hasattr(getattr(attn, name), sub):
                    model[key] = getattr(getattr(attn, name), sub).weight
                    print(f'loaded {key}')
                else:
                    if strict:
                        raise KeyError(f"Missing module: {key}")

    for k, param in state_dict.items():
        if k in model:
            if model[k].shape != param.shape:
                if strict:
                    raise ValueError(
                        f"Shape mismatch: {k} | {model[k].shape} vs {param.shape}"
                    )
                else:
                    continue
            model[k].data.copy_(param)
        else:
            if strict:
                raise KeyError(f"Unexpected key in ckpt: {k}")