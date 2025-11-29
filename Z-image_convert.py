import safetensors.torch
import torch
import sys
import re

# Usage: python Z-image_convert.py output.safetensors input.safetensors
# 将训练输出的 LoRA 转换为 ComfyUI 格式

cast_to = None
if "fp8_e4m3fn" in sys.argv[1]:
    cast_to = torch.float8_e4m3fn
elif "fp16" in sys.argv[1]:
    cast_to = torch.float16
elif "bf16" in sys.argv[1]:
    cast_to = torch.bfloat16


def convert_key(k):
    """
    转换 key 格式：
    lora_unet_layers_0_attention_to_out_0.lora_down.weight
    -> diffusion_model.layers.0.attention.out.lora_down.weight
    """
    # 分离 lora 后缀 (.lora_down.weight, .lora_up.weight, .alpha)
    lora_suffix = ""
    for suffix in [".lora_down.weight", ".lora_up.weight", ".alpha"]:
        if k.endswith(suffix):
            lora_suffix = suffix
            k = k[:-len(suffix)]
            break
    
    # 移除 lora_unet_ 前缀
    if k.startswith("lora_unet_"):
        k = k[len("lora_unet_"):]
    
    # 先处理特殊的命名模式（在数字边界处理之前）
    k = k.replace("attention_to_out_0", "attention.out")  # 移除多余的 _0
    k = k.replace("attention_to_out", "attention.out")
    k = k.replace("attention_to_q", "attention.to_q")
    k = k.replace("attention_to_k", "attention.to_k")
    k = k.replace("attention_to_v", "attention.to_v")
    k = k.replace("attention_norm_q", "attention.q_norm")
    k = k.replace("attention_norm_k", "attention.k_norm")
    k = k.replace("feed_forward_w1", "feed_forward.w1")
    k = k.replace("feed_forward_w2", "feed_forward.w2")
    k = k.replace("feed_forward_w3", "feed_forward.w3")
    
    # 然后处理数字边界
    k = re.sub(r'_(\d+)_', r'.\1.', k)  # layers_0_xxx -> layers.0.xxx
    k = re.sub(r'_(\d+)$', r'.\1', k)   # xxx_0 -> xxx.0
    
    # 剩余的下划线转点号（在数字边界）
    # 但要小心不要破坏 context_refiner 等
    parts = k.split(".")
    new_parts = []
    for part in parts:
        # 对于每个部分，智能转换下划线
        # 保留: context_refiner, noise_refiner, feed_forward, adaLN_modulation, lora_down, lora_up
        if part in ["context_refiner", "noise_refiner", "feed_forward", "adaLN_modulation", "lora_down", "lora_up"]:
            new_parts.append(part)
        elif "_" in part and not any(x in part for x in ["refiner", "forward", "modulation", "lora"]):
            # 尝试在数字边界分割
            sub = re.sub(r'_(\d+)', r'.\1', part)
            new_parts.append(sub)
        else:
            new_parts.append(part)
    
    k = ".".join(new_parts)
    
    # 清理多余的点号
    k = re.sub(r'\.+', '.', k)
    
    # 添加 diffusion_model 前缀
    k_out = "diffusion_model." + k + lora_suffix
    
    return k_out


out_sd = {}
for f in sys.argv[2:]:
    sd = safetensors.torch.load_file(f)
    for k in sd:
        w = sd[k]

        if cast_to is not None:
            w = w.to(cast_to)
        
        k_out = convert_key(k)
        out_sd[k_out] = w
        
print(f"Converted {len(out_sd)} keys")
# 打印几个示例
for i, k in enumerate(list(out_sd.keys())[:5]):
    print(f"  {k}")

safetensors.torch.save_file(out_sd, sys.argv[1])
print(f"Saved to {sys.argv[1]}")