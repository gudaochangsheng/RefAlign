import os
import torch
from typing import Optional, Dict

def _read_state_dict(path: str, map_location="cpu") -> Dict[str, torch.Tensor]:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".safetensors", ".sft"]:
        from safetensors.torch import load_file
        # safetensors 通常就是扁平的 tensor 字典，不会再包一层 "state_dict"
        return load_file(path, device="cpu")  # 再按需 to(...)
    else:
        ckpt = torch.load(path, map_location=map_location)
        # 兼容有/无 "state_dict" 包装
        return ckpt.get("state_dict", ckpt)

def set_hyper(pipe, train: bool = False, model_path: Optional[str] = None, strict: bool = True):
    for blk in pipe.dit.blocks:
        if hasattr(blk, "self_attn") and hasattr(blk.self_attn, "init_hypermlpnet"):
            blk.self_attn.init_hypermlpnet(train=train)
    if model_path is not None:
        print(f"[Hyper] Loading HyperMLP weights from: {model_path}")
        load_hyper_weights_into_pipe(pipe, model_path, strict=strict)

def load_hyper_weights_into_pipe(pipe, path: str, strict: bool = True, map_location="cpu"):
    state_dict = _read_state_dict(path, map_location=map_location)

    loaded_keys, missing_keys, mismatches = [], [], []

    def find_key(fullname: str):
        # 多前缀候选
        cands = [
            fullname,
            f"module.{fullname}",
            f"dit.{fullname}",
            f"model.{fullname}",
            f"model.pipe.{fullname}",
        ]
        for k in cands:
            if k in state_dict:
                return k
        return None

    for i, blk in enumerate(pipe.dit.blocks):
        if not hasattr(blk, "self_attn") or not hasattr(blk.self_attn, "hyper"):
            continue
        hyper = blk.self_attn.hyper
        tgt_sd = hyper.state_dict()
        prefix = f"blocks.{i}.self_attn.hyper."

        for leaf, tgt in tgt_sd.items():
            expect = prefix + leaf

            # 直接匹配 / 多前缀 / 裸 key / 末尾匹配兜底
            src_key = expect if expect in state_dict else find_key(expect)
            if src_key is None and leaf in state_dict:
                src_key = leaf
            if src_key is None:
                for k in state_dict.keys():
                    if k.endswith(leaf):
                        src_key = k
                        break

            if src_key is None:
                missing_keys.append(expect)
                continue

            src = state_dict[src_key]
            if tuple(src.shape) != tuple(tgt.shape):
                mismatches.append((expect, tuple(tgt.shape), tuple(src.shape)))
                continue

            with torch.no_grad():
                tgt.copy_(src.to(tgt.device, dtype=tgt.dtype))
            loaded_keys.append(expect)

    if strict:
        if missing_keys:
            prev = ", ".join(missing_keys[:10])
            more = " ..." if len(missing_keys) > 10 else ""
            raise KeyError(f"[Hyper] Missing keys: {prev}{more}")
        if mismatches:
            prev = ", ".join([f"{k} (exp {e}, got {g})" for k, e, g in mismatches[:6]])
            more = " ..." if len(mismatches) > 6 else ""
            raise ValueError(f"[Hyper] Shape mismatch: {prev}{more}")

    print(f"[Hyper] Loaded={len(loaded_keys)}, Missing={len(missing_keys)}, "
          f"Mismatch={len(mismatches)} from '{path}' (strict={strict}).")

# 可选：保存为 .safetensors，方便下次直接加载
def dump_hyper_state_dict(pipe) -> Dict[str, torch.Tensor]:
    out = {}
    for i, blk in enumerate(pipe.dit.blocks):
        if hasattr(blk, "self_attn") and hasattr(blk.self_attn, "hyper"):
            for leaf, tensor in blk.self_attn.hyper.state_dict().items():
                out[f"blocks.{i}.self_attn.hyper.{leaf}"] = tensor.detach().cpu()
    return out

def save_hyper_safetensors(pipe, path: str):
    from safetensors.torch import save_file
    sd = dump_hyper_state_dict(pipe)
    save_file(sd, path)
    print(f"[Hyper] Saved safetensors to {path}")
