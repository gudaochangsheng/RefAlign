import os, torch
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

# === 每个 block init_inputmlp，并可选加载权重 ===
def set_inputmlp(
    pipe,
    train: bool = False,
    model_path: Optional[str] = None,
    strict: bool = True
):
    dit = pipe.dit
    if not hasattr(dit, "blocks"):
        raise RuntimeError("[InputMLP] dit.blocks not found. This version requires per-block input_mlp.")

    for i, blk in enumerate(dit.blocks[:9]):
    # for i, blk in enumerate(dit.blocks):
        # print("block",i)
        if not hasattr(blk, "init_inputmlp"):
            raise RuntimeError(f"[InputMLP] dit.blocks[{i}] has no init_inputmlp().")
        blk.init_inputmlp(train=train)

    if model_path is not None:
        print(f"[InputMLP] Loading weights from: {model_path}")
        load_inputmlp_weights_into_pipe(pipe, model_path, strict=strict)

# === 只加载每个 block 的 input_mlp 权重（必须是 blocks.{i}.input_mlp.*）===
def load_inputmlp_weights_into_pipe(pipe, path: str, strict: bool = True, map_location="cpu"):
    sd = _read_state_dict(path, map_location=map_location)
    dit = pipe.dit

    if not hasattr(dit, "blocks"):
        raise RuntimeError("[InputMLP] dit.blocks not found. This version requires per-block input_mlp.")

    loaded, missing, mism = [], [], []

    def _find(src_keys, want):
        # 精确、加前缀、后缀兜底
        cands = [
            want,
            f"module.{want}",
            f"dit.{want}",
            f"model.{want}",
            f"model.pipe.{want}",
        ]
        for k in cands:
            if k in src_keys:
                return k
        for k in src_keys:
            if k.endswith(want):
                return k
        return None

    for i, blk in enumerate(dit.blocks[:9]):
        if not hasattr(blk, "input_mlp"):
            raise RuntimeError(
                f"[InputMLP] blocks[{i}].input_mlp not found. "
                f"Call set_inputmlp(...)/block.init_inputmlp() first."
            )

        tgt = blk.input_mlp.state_dict()
        for leaf, tgt_tensor in tgt.items():
            want = f"blocks.{i}.input_mlp.{leaf}"
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
