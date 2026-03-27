import math, types, os, torch, torch.nn as nn
from typing import Optional, Dict


# === 通用 ckpt 读取（沿用你已有的）===
def _read_state_dict(path: str, map_location="cpu") -> Dict[str, torch.Tensor]:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".safetensors", ".sft"]:
        from safetensors.torch import load_file
        return load_file(path, device="cpu")
    else:
        ckpt = torch.load(path, map_location=map_location)
        return ckpt.get("state_dict", ckpt)

# === 把 init_inputmlp 方法“挂载”到 dit 上，并执行 ===
def set_inputmlp(
    pipe,
    train: bool = False,
    model_path: Optional[str] = None,
    strict: bool = True
):
    pipe.dit.init_inputmlp(train=train)
    if model_path is not None:
        print(f"[InputMLP] Loading weights from: {model_path}")
        load_inputmlp_weights_into_pipe(pipe, model_path, strict=strict)

# === 只加载 input_mlp 的权重（自动匹配多种前缀）===
def load_inputmlp_weights_into_pipe(pipe, path: str, strict: bool = True, map_location="cpu"):
    sd = _read_state_dict(path, map_location=map_location)
    dit = pipe.dit
    if not hasattr(dit, "input_mlp"):
        raise RuntimeError("[InputMLP] input_mlp not found. Call set_inputmlp(...)/init_inputmlp() first.")

    tgt = dit.input_mlp.state_dict()

    loaded, missing, mism = [], [], []

    def _find(src_keys, want):
        # 精确、加前缀、裸 key、后缀兜底
        cands = [want, f"module.{want}", f"dit.{want}", f"model.{want}", f"model.pipe.{want}"]
        for k in cands:
            if k in src_keys: return k
        if want in src_keys: return want
        for k in src_keys:
            if k.endswith(want): return k
        return None

    for leaf, tgt_tensor in tgt.items():
        want = f"input_mlp.{leaf}"
        src_key = _find(sd.keys(), want)
        if src_key is None:
            missing.append(want); continue
        src = sd[src_key]
        if tuple(src.shape) != tuple(tgt_tensor.shape):
            mism.append((want, tuple(tgt_tensor.shape), tuple(src.shape))); continue
        with torch.no_grad():
            tgt_tensor.copy_(src.to(tgt_tensor.device, dtype=tgt_tensor.dtype))
        loaded.append(want)
    
    

    if strict:
        if missing:
            prev = ", ".join(missing[:10]); more = " ..." if len(missing) > 10 else ""
            raise KeyError(f"[InputMLP] Missing keys: {prev}{more}")
        if mism:
            prev = ", ".join([f"{k} (exp {e}, got {g})" for k,e,g in mism[:6]])
            more = " ..." if len(mism) > 6 else ""
            raise ValueError(f"[InputMLP] Shape mismatch: {prev}{more}")

    print(f"[InputMLP] Loaded={len(loaded)}, Missing={len(missing)}, Mismatch={len(mism)} from '{path}' (strict={strict}).")

# def load_inputmlp_weights_into_pipe(pipe, path: str, strict: bool = True, map_location="cpu"):
#     sd = _read_state_dict(path, map_location=map_location)
#     dit = pipe.dit
#     if not hasattr(dit, "input_mlp") or not hasattr(dit, "output_mlp"):
#         raise RuntimeError("[InputMLP] call init_inputmlp() first.")

#     loaded, missing, mism = [], [], []

#     def _find(src_keys, want):
#         cands = [want, f"module.{want}", f"dit.{want}", f"model.{want}", f"model.pipe.{want}"]
#         for k in cands:
#             if k in src_keys: return k
#         for k in src_keys:
#             if k.endswith(want): return k
#         return None

#     def _load_one_module(mod, prefix: str):
#         nonlocal loaded, missing, mism
#         # 用 named_parameters + buffers，这样 copy_ 到的是“参数本体”
#         named_tensors = dict(mod.named_parameters())
#         named_tensors.update({fbuf: buf for fbuf, buf in mod.named_buffers()})

#         for leaf, tgt_tensor in named_tensors.items():
#             want = f"{prefix}.{leaf}"
#             src_key = _find(sd.keys(), want)
#             if src_key is None:
#                 missing.append(want); continue
#             src = sd[src_key]
#             if tuple(src.shape) != tuple(tgt_tensor.shape):
#                 mism.append((want, tuple(tgt_tensor.shape), tuple(src.shape))); continue
#             with torch.no_grad():
#                 tgt_tensor.copy_(src.to(tgt_tensor.device, dtype=tgt_tensor.dtype))
#             loaded.append(want)

#     # 分开加载，避免前缀错配
#     _load_one_module(dit.input_mlp,  "input_mlp")
#     _load_one_module(dit.output_mlp, "output_mlp")

#     if strict:
#         if missing:
#             prev = ", ".join(missing[:10]); more = " ..." if len(missing) > 10 else ""
#             raise KeyError(f"[InputMLP] Missing keys: {prev}{more}")
#         if mism:
#             prev = ", ".join([f"{k} (exp {e}, got {g})" for k,e,g in mism[:6]])
#             more = " ..." if len(mism) > 6 else ""
#             raise ValueError(f"[InputMLP] Shape mismatch: {prev}{more}")

#     print(f"[InputMLP] Loaded={len(loaded)}, Missing={len(missing)}, Mismatch={len(mism)} from '{path}' (strict={strict}).")
