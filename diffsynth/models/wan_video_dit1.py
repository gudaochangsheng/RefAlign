import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
from .utils import hash_state_dict_keys
from .wan_video_camera_controller import SimpleAdapter
try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
    
    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x,tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


# def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
#     # 3d rope precompute
#     f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
#     h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
#     w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
#     return f_freqs_cis, h_freqs_cis, w_freqs_cis


# def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
#     # 1d rope precompute
#     freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
#                    [: (dim // 2)].double() / dim))
#     freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
#     freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
#     return freqs_cis

def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end + 3, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    ######################################################     add  f = -3
    positions = torch.arange(-3, end, device=freqs.device)
    freqs = torch.outer(positions, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    ######################################################
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


# class Hypermlpnet(nn.Module):
#     def __init__(
#         self,
#         in_features: int,
#         out_features: int,
#         hidden: int,
#         rank: int = 128,
#         factor: int = 256,
#         device="cuda",
#         dtype: Optional[torch.dtype] = torch.float32,
#     ):
#         super().__init__()
#         new_in = in_features // factor
#         new_out = out_features // factor
#         self.net = nn.Sequential(
#             nn.Linear(in_features, hidden, bias=False, device=device, dtype=dtype),
#             nn.GELU(),
#             nn.Linear(hidden, hidden, bias=False, device=device, dtype=dtype),
#             nn.GELU(),
#             nn.Linear(hidden, hidden, bias=False, device=device, dtype=dtype)
#         )
#         self.head_A = nn.Linear(hidden, in_features * rank, bias=False, device=device, dtype=dtype)
#         self.head_B = nn.Linear(hidden, rank * out_features, bias=False, device=device, dtype=dtype)
#         self.in_features = in_features
#         self.out_features = out_features
#         self.rank = rank
#         self.new_in = new_in
#         self.new_out = new_out
#         self.reset_parameters()

#     def reset_parameters(self):
#         # Xavier（Glorot）初始化；GELU 可用 gain≈sqrt(2)
#         gain = math.sqrt(2.0)
#         def _init(m):
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight, gain=gain)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#         self.net.apply(_init)
#         # nn.init.xavier_uniform_(self.head_A.weight, gain=gain)
#         # nn.init.xavier_uniform_(self.head_B.weight, gain=gain)
#         nn.init.zeros_(self.head_A.weight)
#         nn.init.zeros_(self.head_B.weight)

#     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         orig_dtype = hidden_states.dtype
#         dtype = self.net[0].weight.dtype
#         hidden_states = hidden_states.to(dtype)
#         # print(hidden_states.shape)
#         hidden_states = hidden_states.transpose(1,2)
#         hidden_states = F.adaptive_avg_pool1d(hidden_states, 1)
#         # print(hidden_states.shape)
#         # assert 2==1
#         hidden_states = hidden_states.squeeze(-1)
#         weight = self.net(hidden_states)
#         weight = weight.squeeze(0)
#         # a_len = self.out_features * self.rank
#         # A = weight[:a_len].view(self.out_features, self.rank)
#         weight_A = self.head_A(weight).view(self.in_features, self.rank) #in_c rank
#         weight_B = self.head_B(weight).view(self.rank, self.out_features) #
        
#         return weight_A.to(orig_dtype), weight_B.to(orig_dtype)
    


# class SelfAttention(nn.Module):
#     def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads

#         self.q = nn.Linear(dim, dim)
#         self.k = nn.Linear(dim, dim)
#         self.v = nn.Linear(dim, dim)
#         self.o = nn.Linear(dim, dim)
#         self.norm_q = RMSNorm(dim, eps=eps)
#         self.norm_k = RMSNorm(dim, eps=eps)
        
#         self.attn = AttentionModule(self.num_heads)
#         self.cond_size = None
#         self.image_num = None
    
#     def init_hypermlpnet(self, train=False):
#         dim = self.dim
#         self.hyper = Hypermlpnet(dim, dim, hidden= 128,rank=8, factor=32)

#         for p in self.hyper.parameters():
#             p.requires_grad = bool(train)

#     def forward(self, x, freqs):
#         if self.cond_size is not None:
#             x_ip, x_main = x[:, : self.cond_size,:], x[:, self.cond_size:,:]
#             subject_token = self.cond_size // self.image_num
#             subjects = []
#             for i in range(self.image_num):
#                 start = i * subject_token
#                 end = (i+1)* subject_token
#                 subject = x_ip[:,start:end,:]
#                 # print(subject.shape, self.hyper(subject).shape)
#                 # assert 2==1
#                 A, B = self.hyper(subject)
#                 subject = subject@A@B
#                 subjects.append(subject)
#             subject = torch.cat(subjects, dim=1)
#             subject = subject + x_ip
#             x = torch.cat([subject, x_main], dim=1)
        
#         q = self.norm_q(self.q(x))
#         k = self.norm_k(self.k(x))
#         v = self.v(x)
#         q = rope_apply(q, freqs, self.num_heads)
#         k = rope_apply(k, freqs, self.num_heads)
#         x = self.attn(q, k, v)
#         return self.o(x)

class LoRALinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 128,
        device="cuda",
        dtype: Optional[torch.dtype] = torch.float32,
    ):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)
        return up_hidden_states.to(orig_dtype)

def gate_from_timestep(t, t_first, t_last, beta=3.5, descending=True):
    """
    t: 当前整数时间步（或张量）
    t_first: 序列第一个步（通常高噪声，比如 999）
    t_last:  序列最后一个步（通常低噪声，比如 0）
    descending: timesteps 是否降序
    """
    if descending:
        # t 越靠前越高噪声 → 归一化后越大
        x = (t - t_last) / (t_first - t_last + 1e-8)
    else:
        x = (t - t_first) / (t_last - t_first + 1e-8)
    x = x.clamp(0, 1)
    gate01 = (1.0 - x).pow(beta)     # [0,1]
    gate = 0.1 + 0.9 * gate01        # 线性映射到 [0.1,1.0]
    return gate


# class SelfAttention(nn.Module):
#     def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads

#         self.q = nn.Linear(dim, dim)
#         self.k = nn.Linear(dim, dim)
#         self.v = nn.Linear(dim, dim)
#         self.o = nn.Linear(dim, dim)
#         self.norm_q = RMSNorm(dim, eps=eps)
#         self.norm_k = RMSNorm(dim, eps=eps)
        
#         self.attn = AttentionModule(self.num_heads)
#         self.cond_size = None
#         self.image_num = None

#     def init_lora(self, train=False):
#         dim = self.dim
#         self.q_loras = LoRALinearLayer(dim, dim, rank=512)
#         self.k_loras = LoRALinearLayer(dim, dim, rank=512)
#         self.v_loras = LoRALinearLayer(dim, dim, rank=512)
#         # self.q_video = LoRALinearLayer(dim, dim, rank=512)
#         # self.k_video = LoRALinearLayer(dim, dim, rank=512)
#         # self.v_video = LoRALinearLayer(dim, dim, rank=512)

#         requires_grad = train
#         for lora in [self.q_loras, self.k_loras, self.v_loras]:
#             for param in lora.parameters():
#                 param.requires_grad = requires_grad


#     def forward(self, x, freqs, timestep):
#         cond_size = self.cond_size
#         # print(timestep)
#         # gate = gate_from_timestep(timestep, 999, 0)
#         gate = 1.0
#         # gate = gate*0.5
#         # print(gate)
#         if cond_size is not None:
#             x_ip, x_main = x[:, : cond_size,:], x[:, cond_size:,:]
#             # x_ip = x_ip.detach()
#             # x_main = x_main.detach()
#             freqs_ip, freqs_main = freqs[: cond_size], freqs[cond_size:]
#             q_main = self.norm_q(self.q(x_main))
#             k_main = self.norm_k(self.k(x_main))
#             v_main = self.v(x_main)

#             q_main = rope_apply(q_main, freqs_main, self.num_heads)
#             k_main = rope_apply(k_main, freqs_main, self.num_heads)

            # q_ip = self.norm_q(self.q(x_ip)+ gate*self.q_loras(x_ip))
            # k_ip = self.norm_k(self.k(x_ip)+ gate*self.k_loras(x_ip))
            # v_ip = self.v(x_ip) + gate*self.v_loras(x_ip)

#             q_ip = rope_apply(q_ip, freqs_ip, self.num_heads)
#             k_ip = rope_apply(k_ip, freqs_ip, self.num_heads)
#             # k_ip = k_ip.detach()
#             # v_ip = v_ip.detach()
#             # q = torch.cat([q_ip, q_main], dim=1)
#             k_all = torch.cat([k_ip, k_main], dim=1)
#             v_all = torch.cat([v_ip, v_main], dim=1)
#             main = self.attn(q_main, k_all, v_all)
#             subject = self.attn(q_ip, k_ip, v_ip)
#             # x = self.attn(q, k, v)
#             # print(self.q_loras.up.weight)
#             # assert 2==1
#             x = torch.cat([subject, main], dim=1)
#             return self.o(x)
#         else:
#             q = self.norm_q(self.q(x))
#             k = self.norm_k(self.k(x))
#             v = self.v(x)
#             q = rope_apply(q, freqs, self.num_heads)
#             k = rope_apply(k, freqs, self.num_heads)
#             x = self.attn(q, k, v)
#             return self.o(x)

class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)
        self.cond_size = None
        self.image_num = None

    def init_lora(self, train=False):
        dim = self.dim
        self.q_loras = LoRALinearLayer(dim, dim, rank=256)
        self.k_loras = LoRALinearLayer(dim, dim, rank=256)
        self.v_loras = LoRALinearLayer(dim, dim, rank=256)
        # self.q_video = LoRALinearLayer(dim, dim, rank=512)
        # self.k_video = LoRALinearLayer(dim, dim, rank=512)
        # self.v_video = LoRALinearLayer(dim, dim, rank=512)

        requires_grad = train
        for lora in [self.q_loras, self.k_loras, self.v_loras]:
            for param in lora.parameters():
                param.requires_grad = requires_grad


    def forward(self, x, freqs, timestep):
        cond_size = self.cond_size
        # print(timestep)
        # gate = gate_from_timestep(timestep, 999, 0)
        gate = 1.0
        # gate = gate*0.5
        # print(gate)
        if cond_size is not None:
            x_ip, x_main = x[:, : cond_size,:], x[:, cond_size:,:]
            # x_ip = x_ip.detach()
            # x_main = x_main.detach()
            freqs_ip, freqs_main = freqs[: cond_size], freqs[cond_size:]
            q_main = self.norm_q(self.q(x_main))
            k_main = self.norm_k(self.k(x_main))
            v_main = self.v(x_main)

            q_main = rope_apply(q_main, freqs_main, self.num_heads)
            k_main = rope_apply(k_main, freqs_main, self.num_heads)

            q_ip = self.norm_q(self.q(x_ip)+ gate*self.q_loras(x_ip))
            k_ip = self.norm_k(self.k(x_ip)+ gate*self.k_loras(x_ip))
            v_ip = self.v(x_ip) + gate*self.v_loras(x_ip)

            q_ip = rope_apply(q_ip, freqs_ip, self.num_heads)
            k_ip = rope_apply(k_ip, freqs_ip, self.num_heads)
            # k_ip = k_ip.detach()
            # v_ip = v_ip.detach()
            # q = torch.cat([q_ip, q_main], dim=1)
            k_all = torch.cat([k_ip, k_main], dim=1)
            v_all = torch.cat([v_ip, v_main], dim=1)
            main = self.attn(q_main, k_all, v_all)
            subject = self.attn(q_ip, k_ip, v_ip)
            # x = self.attn(q, k, v)
            # print(self.q_loras.up.weight)
            # assert 2==1
            x = torch.cat([subject, main], dim=1)
            return self.o(x)
        else:
            q = self.norm_q(self.q(x))
            k = self.norm_k(self.k(x))
            v = self.v(x)
            q = rope_apply(q, freqs, self.num_heads)
            k = rope_apply(k, freqs, self.num_heads)
            x = self.attn(q, k, v)
            return self.o(x)

class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def init_inputmlp(self, train=False):
        dim = self.dim
        self.input_mlp = MLP3(dim, dim // 2,768, has_pos_emb=False)
        for p in self.input_mlp.parameters():
            p.requires_grad = bool(train)

    def forward(self, x, context, t_mod, freqs, x_ip, cond_size, image_num,t_mod_ip, timestep):
        if t_mod_ip is None: 
            self.self_attn.cond_size = cond_size
            # self.self_attn.image_num = image_num
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
            input_x = modulate(self.norm1(x), shift_msa, scale_msa)
            x = self.gate(x, gate_msa, self.self_attn(input_x, freqs, timestep))
            x = x + self.cross_attn(self.norm3(x), context)
            input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
            x = self.gate(x, gate_mlp, self.ffn(input_x))
            return x,x_ip
        else:
            self.self_attn.cond_size = cond_size
            # x = x[:,cond_size:]
            # x_ip = x[:, :cond_size]
            (
                shift_msa_ip,
                scale_msa_ip,
                gate_msa_ip,
                shift_mlp_ip,
                scale_mlp_ip,
                gate_mlp_ip,
            ) = (
                self.modulation.to(dtype=t_mod_ip.dtype, device=t_mod_ip.device)
                + t_mod_ip
            ).chunk(6, dim=1)
            input_x_ip = modulate(
                self.norm1(x_ip), shift_msa_ip, scale_msa_ip
            )
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
            input_x = modulate(self.norm1(x), shift_msa, scale_msa)
            input_x = torch.concat([input_x_ip,input_x], dim=1)
            attn_out = self.self_attn(input_x, freqs, timestep)
            attn_out, attn_out_ip = (attn_out[:,cond_size:],attn_out[:, :cond_size])

            x = self.gate(x, gate_msa, attn_out)
            x = x + self.cross_attn(self.norm3(x), context)
            input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
            x = self.gate(x, gate_mlp, self.ffn(input_x))
            x_ip = self.gate(x_ip, gate_msa_ip, attn_out_ip)
            input_x_ip = modulate(self.norm2(x_ip), shift_mlp_ip, scale_mlp_ip)
            x_ip = self.gate(x_ip, gate_mlp_ip, self.ffn(input_x_ip))
            return x,x_ip

# class DiTBlock(nn.Module):
#     def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.ffn_dim = ffn_dim

#         self.self_attn = SelfAttention(dim, num_heads, eps)
#         self.cross_attn = CrossAttention(
#             dim, num_heads, eps, has_image_input=has_image_input)
#         self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
#         self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
#         self.norm3 = nn.LayerNorm(dim, eps=eps)
#         self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
#             approximate='tanh'), nn.Linear(ffn_dim, dim))
#         self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
#         self.gate = GateModule()

#     def forward(self, x, context, t_mod, freqs, x_ip, cond_size, image_num,t_mod_ip, timestep):
#         if t_mod_ip is None: 
#             # self.self_attn.cond_size = cond_size
#             # self.self_attn.image_num = image_num
#             shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
#                 self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
#             input_x = modulate(self.norm1(x), shift_msa, scale_msa)
#             x = self.gate(x, gate_msa, self.self_attn(input_x, freqs, timestep))
#             x_ip = x[:,: cond_size]
#             x_video = x[:, cond_size:]
#             cross_video = self.cross_attn(self.norm3(x_video), context)
#             x_video = x_video + cross_video
#             x = torch.concat([x_ip,x_video], dim=1)
#             input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
#             x = self.gate(x, gate_mlp, self.ffn(input_x))
#             x_ip = x[:,: cond_size]
#             return x,x_ip
        


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)

class MLP3(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim,  out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim, bias=False)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)

class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


def _ensure_on(mod: nn.Module, device: torch.device, dtype: torch.dtype):
    # 若发现有任意参数/缓冲区不在目标 device/dtype，就整体迁移一次
    need_move = False
    for t in list(mod.parameters()) + list(mod.buffers()):
        if t is None or t.numel() == 0:
            continue
        if t.device != device or t.dtype != dtype:
            need_move = True
            break
    if need_move:
        mod.to(device=device, dtype=dtype)


class DenseHead(nn.Module):
    def __init__(self, dim, n_layers, depth, n_heads, dim_dit, device="cuda",
        dtype: Optional[torch.dtype] = torch.bfloat16):
        super().__init__()
        self.n_layers = n_layers
        self.dim = dim
        self.pos = nn.Parameter(torch.zeros(1, n_layers, dim, device=device, dtype=dtype))
        nn.init.normal_(self.pos, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim*4, batch_first=True, norm_first=True, activation="gelu", device=device,                # <---
            dtype=dtype
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        # self.encoder.to(device=device, dtype=dtype)
        self.reduce = nn.Linear(n_layers, 1, bias=False, device=device, dtype=dtype)
        with torch.no_grad():
            self.reduce.weight.fill_(1.0 / n_layers)
        self.proj = MLP(dim, dim_dit, has_pos_emb=False)
        self.proj.to(device=device, dtype=dtype)
    def forward(self, x):
        orig_dtype = x.dtype
        target_dtype = self.pos.dtype
        target_device = self.pos.device
        _ensure_on(self.encoder, target_device, target_dtype)
        _ensure_on(self.reduce,  target_device, target_dtype)
        _ensure_on(self.proj,    target_device, target_dtype)
        B, L, N, D = x.shape
        # print(x.shape)
        x = x.view(B*L, N, D).to(device=target_device, dtype=target_dtype)
        x = x + self.pos
        x = self.encoder(x)
        x_dn = x.transpose(1,2)
        fused = self.reduce(x_dn).squeeze(-1)
        fuse = fused.view(B, L, D)
        x = self.proj(fuse)
        return x.to(orig_dtype)



class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.control_adapter = None

    def init_inputmlp(self, train=False):
        dim = self.dim
        # self.input_mlp = DenseHead(768, 12,4, 4, dim)
        self.input_mlp = MLP3(dim, dim // 2,768, has_pos_emb=False)
        # self.output_mlp = InputMLP3D(in_channels=3584, hidden_channels=3584//2, out_channels=dim,dropout=0.0, num_layers=1)
        for p in self.input_mlp.parameters():
            p.requires_grad = bool(train)
        # for p in self.output_mlp.parameters():
        #     p.requires_grad = bool(train)

    def patchify(self, x: torch.Tensor,control_camera_latents_input: torch.Tensor = None):
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        grid_size = x.shape[2:]
        # y = rearrange(x, 'b c f h w -> b f (h w) c').contiguous()
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)
    
    def patchify_ip(self, x: torch.Tensor,control_camera_latents_input: torch.Tensor = None):
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        # x = self.vae2qwen(x)
        grid_size = x.shape[2:]
        # y = rearrange(x, 'b c f h w -> b f (h w) c').contiguous()
        y = x
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size, y  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        
        x, (f, h, w) = self.patchify(x)
        
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, t_mod, freqs)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()
    
    
class WanModelStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = {
            "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
            "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
            "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
            "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
            "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
            "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
            "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
            "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
            "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
            "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
            "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
            "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
            "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
            "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
            "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
            "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
            "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
            "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
            "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
            "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
            "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
            "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
            "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
            "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
            "blocks.0.norm2.bias": "blocks.0.norm3.bias",
            "blocks.0.norm2.weight": "blocks.0.norm3.weight",
            "blocks.0.scale_shift_table": "blocks.0.modulation",
            "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
            "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
            "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
            "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
            "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
            "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
            "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
            "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
            "condition_embedder.time_proj.bias": "time_projection.1.bias",
            "condition_embedder.time_proj.weight": "time_projection.1.weight",
            "patch_embedding.bias": "patch_embedding.bias",
            "patch_embedding.weight": "patch_embedding.weight",
            "scale_shift_table": "head.modulation",
            "proj_out.bias": "head.head.bias",
            "proj_out.weight": "head.head.weight",
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            else:
                name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
                if name_ in rename_dict:
                    name_ = rename_dict[name_]
                    name_ = ".".join(name_.split(".")[:1] + [name.split(".")[1]] + name_.split(".")[2:])
                    state_dict_[name_] = param
        if hash_state_dict_keys(state_dict) == "cb104773c6c2cb6df4f9529ad5c60d0b":
            config = {
                "model_type": "t2v",
                "patch_size": (1, 2, 2),
                "text_len": 512,
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "window_size": (-1, -1),
                "qk_norm": True,
                "cross_attn_norm": True,
                "eps": 1e-6,
            }
        else:
            config = {}
        return state_dict_, config
    
    def from_civitai(self, state_dict):
        state_dict = {name: param for name, param in state_dict.items() if not name.startswith("vace")}
        if hash_state_dict_keys(state_dict) == "9269f8db9040a9d860eaca435be61814":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "aafcfd9672c3a2456dc46e1cb6e52c70":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6d6ccde6845b95ad9114ab993d917893":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "349723183fc063b2bfc10bb2835cf677":
            # 1.3B PAI control
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "efa44cddf936c70abd0ea28b6cbe946c":
            # 14B PAI control
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "3ef3b1f8e1dab83d5b71fd7b617f859f":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "has_image_pos_emb": True
            }
        elif hash_state_dict_keys(state_dict) == "70ddad9d3a133785da5ea371aae09504":
            # 1.3B PAI control v1.1
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6,
                "has_ref_conv": True
            }
        elif hash_state_dict_keys(state_dict) == "26bde73488a92e64cc20b0a7485b9e5b":
            # 14B PAI control v1.1
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "has_ref_conv": True
            }
        elif hash_state_dict_keys(state_dict) == "ac6a5aa74f4a0aab6f64eb9a72f19901":
            # 1.3B PAI control-camera v1.1
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 32,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6,
                "has_ref_conv": False,
                "add_control_adapter": True,
                "in_dim_control_adapter": 24,
            }
        elif hash_state_dict_keys(state_dict) == "b61c605c2adbd23124d152ed28e049ae":
            # 14B PAI control-camera v1.1
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 32,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "has_ref_conv": False,
                "add_control_adapter": True,
                "in_dim_control_adapter": 24,
            }
        else:
            config = {}
        return state_dict, config
