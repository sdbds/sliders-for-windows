# Z-Image model loading utilities
# ref: https://github.com/JerryWu-code/diffusers/blob/1048d0a94d553978b35bec0942c408c57bb12c9e/src/diffusers/pipelines/z_image/pipeline_z_image.py

from typing import Tuple, Optional
import os
import torch
from safetensors.torch import load_file as load_safetensors
from transformers import AutoTokenizer, PreTrainedModel, AutoModel
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers import ZImageTransformer2DModel


DIFFUSERS_CACHE_DIR = None

# Default base model for loading tokenizer/text_encoder when using single file
DEFAULT_ZIMAGE_BASE_MODEL = "Tongyi-MAI/Z-Image-Turbo"


def is_safetensors(path: str) -> bool:
    """Check if path is a safetensors file."""
    return os.path.splitext(path)[1].lower() == ".safetensors"


def is_single_file(path: str) -> bool:
    """Check if path is a single checkpoint file (.safetensors or .ckpt)."""
    return path.endswith(".safetensors") or path.endswith(".ckpt")


def is_diffusers_directory(path: str) -> bool:
    """
    Check if path is a diffusers format model directory.
    Diffusers format has model_index.json in the root directory.
    """
    if not os.path.isdir(path):
        return False
    return os.path.isfile(os.path.join(path, "model_index.json"))


def get_model_type(path: str) -> str:
    """
    Detect the type of model path.
    
    Returns:
        "single_file" - Single safetensors/ckpt file
        "diffusers" - Diffusers format directory with subfolders
        "unknown" - Unknown format (will try as HuggingFace model ID)
    """
    if is_single_file(path):
        return "single_file"
    
    if os.path.isdir(path) and is_diffusers_directory(path):
        return "diffusers"
    
    return "unknown"


def print_directory_info(path: str) -> None:
    """Print information about a model path for debugging."""
    model_type = get_model_type(path)
    print(f"Model path: {path}")
    print(f"Model type: {model_type}")
    
    if model_type == "diffusers":
        print("Diffusers format detected. Subfolders:")
        for subfolder in ["transformer", "vae", "text_encoder", "tokenizer", "scheduler"]:
            subfolder_path = os.path.join(path, subfolder)
            if os.path.isdir(subfolder_path):
                files = os.listdir(subfolder_path)
                print(f"  {subfolder}/: {files}")
    elif model_type == "single_file":
        print(f"Single file: {os.path.basename(path)}")


def load_tokenizer(model_path: str) -> AutoTokenizer:
    """Load Z-Image tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        subfolder="tokenizer",
        cache_dir=DIFFUSERS_CACHE_DIR,
        trust_remote_code=True,
    )
    return tokenizer


def load_text_encoder(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> PreTrainedModel:
    """Load Z-Image text encoder directly to GPU."""
    text_encoder = AutoModel.from_pretrained(
        model_path,
        subfolder="text_encoder",
        dtype=dtype,
        cache_dir=DIFFUSERS_CACHE_DIR,
        trust_remote_code=True,
        device_map=device,
    )
    return text_encoder


def load_transformer(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> ZImageTransformer2DModel:
    """Load Z-Image transformer directly to GPU."""
    # Z-Image uses a transformer similar to Flux
    transformer = ZImageTransformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=dtype,
        cache_dir=DIFFUSERS_CACHE_DIR,
        trust_remote_code=True,
        device_map=device,
    )
    return transformer


def load_transformer_from_single_file(
    checkpoint_path: str,
    dtype: torch.dtype = torch.bfloat16,
) -> ZImageTransformer2DModel:
    """Load Z-Image transformer from a single safetensors/ckpt file."""
    transformer = ZImageTransformer2DModel.from_single_file(
        checkpoint_path,
        torch_dtype=dtype,
        cache_dir=DIFFUSERS_CACHE_DIR,
    )
    return transformer


def load_vae(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
) -> AutoencoderKL:
    """Load Z-Image VAE."""
    vae = AutoencoderKL.from_pretrained(
        model_path,
        subfolder="vae",
        dtype=dtype,
        cache_dir=DIFFUSERS_CACHE_DIR,
    )
    return vae


def load_scheduler(model_path: str) -> FlowMatchEulerDiscreteScheduler:
    """Load Z-Image scheduler (FlowMatchEulerDiscreteScheduler)."""
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_path,
        subfolder="scheduler",
        cache_dir=DIFFUSERS_CACHE_DIR,
    )
    return scheduler


def create_default_scheduler() -> FlowMatchEulerDiscreteScheduler:
    """Create default FlowMatchEulerDiscreteScheduler for Z-Image."""
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1.0,
    )
    return scheduler


def find_safetensors_in_subfolder(base_path: str, subfolder: str) -> Optional[str]:
    """Find a safetensors file in a subfolder."""
    subfolder_path = os.path.join(base_path, subfolder)
    if not os.path.isdir(subfolder_path):
        return None
    
    for f in os.listdir(subfolder_path):
        if f.endswith(".safetensors"):
            return os.path.join(subfolder_path, f)
    return None


# Default Z-Image Transformer config (no HuggingFace download needed)
DEFAULT_ZIMAGE_TRANSFORMER_CONFIG = {
    "_class_name": "ZImageTransformer2DModel",
    "_diffusers_version": "0.36.0.dev0",
    "all_f_patch_size": [1],
    "all_patch_size": [2],
    "axes_dims": [32, 48, 48],
    "axes_lens": [1536, 512, 512],
    "cap_feat_dim": 2560,
    "dim": 3840,
    "in_channels": 16,
    "n_heads": 30,
    "n_kv_heads": 30,
    "n_layers": 30,
    "n_refiner_layers": 2,
    "norm_eps": 1e-05,
    "qk_norm": True,
    "rope_theta": 256.0,
    "t_scale": 1000.0,
}


def load_transformer_from_local_safetensors(
    safetensors_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> ZImageTransformer2DModel:
    """
    Load transformer directly from a local safetensors file to GPU.
    Uses hardcoded config, no HuggingFace download needed.
    
    Args:
        safetensors_path: Path to the .safetensors file
        dtype: Model dtype
        device: Target device (default: cuda)
    """
    import json
    print(f"Loading transformer from: {safetensors_path}")
    
    # Load state dict directly to GPU
    state_dict = load_safetensors(safetensors_path, device=device)
    
    # Try to load config from the same directory first
    config_dir = os.path.dirname(safetensors_path)
    config_file = os.path.join(config_dir, "config.json")
    
    if os.path.exists(config_file):
        # Use local config
        print("Using local config.json")
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        # Use hardcoded default config
        print("Using default Z-Image transformer config")
        config = DEFAULT_ZIMAGE_TRANSFORMER_CONFIG
    
    # Create model directly on GPU with correct dtype
    with torch.device(device):
        transformer = ZImageTransformer2DModel.from_config(config)
        transformer = transformer.to(dtype)
    
    # Load weights (already on GPU)
    transformer.load_state_dict(state_dict, strict=False)
    
    return transformer


# Default Z-Image Text Encoder config (Qwen3-4B, no HuggingFace download needed)
DEFAULT_ZIMAGE_TEXT_ENCODER_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 2560,
    "initializer_range": 0.02,
    "intermediate_size": 9728,
    "max_position_embeddings": 40960,
    "max_window_layers": 36,
    "model_type": "qwen3",
    "num_attention_heads": 32,
    "num_hidden_layers": 36,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_scaling": None,
    "rope_theta": 1000000,
    "sliding_window": None,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.51.0",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
}


def load_text_encoder_from_local_safetensors(
    safetensors_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> PreTrainedModel:
    """
    Load text encoder directly from a local safetensors file to GPU.
    Uses hardcoded config, no HuggingFace download needed.
    """
    from transformers import AutoConfig
    import json
    print(f"Loading text encoder from: {safetensors_path}")
    
    # Load state dict directly to GPU
    state_dict = load_safetensors(safetensors_path, device=device)
    
    # Try to load config from local directory first
    config_dir = os.path.dirname(safetensors_path)
    config_file = os.path.join(config_dir, "config.json")
    
    if os.path.exists(config_file):
        # Use local config
        print("Using local config.json")
        config = AutoConfig.from_pretrained(config_dir, trust_remote_code=True)
    else:
        # Use hardcoded default config
        print("Using default Z-Image text encoder config (Qwen3-4B)")
        # Remove model_type from kwargs since it's the first positional arg
        config_kwargs = {k: v for k, v in DEFAULT_ZIMAGE_TEXT_ENCODER_CONFIG.items() if k != "model_type"}
        config = AutoConfig.for_model("qwen3", **config_kwargs)
    
    # Create model directly on GPU with correct dtype
    with torch.device(device):
        text_encoder = AutoModel.from_config(config, trust_remote_code=True, torch_dtype=dtype)
    
    # Load weights (already on GPU)
    text_encoder.load_state_dict(state_dict, strict=False)
    
    return text_encoder


# Default Z-Image VAE config (no HuggingFace download needed)
DEFAULT_ZIMAGE_VAE_CONFIG = {
    "_class_name": "AutoencoderKL",
    "_diffusers_version": "0.36.0.dev0",
    "act_fn": "silu",
    "block_out_channels": [128, 256, 512, 512],
    "down_block_types": [
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
    ],
    "force_upcast": True,
    "in_channels": 3,
    "latent_channels": 16,
    "latents_mean": None,
    "latents_std": None,
    "layers_per_block": 2,
    "mid_block_add_attention": True,
    "norm_num_groups": 32,
    "out_channels": 3,
    "sample_size": 1024,
    "scaling_factor": 0.3611,
    "shift_factor": 0.1159,
    "up_block_types": [
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    ],
    "use_post_quant_conv": False,
    "use_quant_conv": False,
}


def load_vae_from_local_safetensors(
    safetensors_path: str,
    dtype: torch.dtype = torch.bfloat16,
) -> AutoencoderKL:
    """
    Load VAE directly from a local safetensors file.
    Uses hardcoded config, no HuggingFace download needed.
    """
    import json
    print(f"Loading VAE from: {safetensors_path}")
    
    # Load state dict
    state_dict = load_safetensors(safetensors_path)
    
    # Try to load config from local directory
    config_file = os.path.join(os.path.dirname(safetensors_path), "config.json")
    if os.path.exists(config_file):
        print("Using local config.json")
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
    else:
        # Use hardcoded default config
        print("Using default Z-Image VAE config")
        config_dict = DEFAULT_ZIMAGE_VAE_CONFIG
    
    vae = AutoencoderKL.from_config(config_dict)
    vae = vae.to(dtype)
    
    # Load weights
    vae.load_state_dict(state_dict, strict=False)
    
    return vae


def load_tokenizer_and_text_encoder(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    text_encoder_path: Optional[str] = None,
) -> Tuple[AutoTokenizer, PreTrainedModel]:
    """
    Load only tokenizer and text encoder for prompt encoding.
    Used for staged loading to save VRAM.
    
    Args:
        model_path: Main model path (diffusers dir, single file, or HF repo)
        dtype: Model dtype
        text_encoder_path: Optional separate path for text_encoder (diffusers dir, single file, or HF repo)
    """
    # Use separate text_encoder_path if provided
    te_path = text_encoder_path or model_path
    te_model_type = get_model_type(te_path)
    
    if te_model_type == "single_file":
        # Single safetensors file for text encoder
        print(f"Loading text encoder from single file: {te_path}")
        text_encoder = load_text_encoder_from_local_safetensors(te_path, dtype)
        # Tokenizer must come from default base model
        tokenizer = load_tokenizer(DEFAULT_ZIMAGE_BASE_MODEL)
    elif te_model_type == "diffusers" and os.path.isdir(te_path):
        # Check if local safetensors
        text_encoder_file = find_safetensors_in_subfolder(te_path, "text_encoder")
        if text_encoder_file:
            print(f"Loading text encoder from local safetensors: {text_encoder_file}")
            text_encoder = load_text_encoder_from_local_safetensors(text_encoder_file, dtype)
            tokenizer = load_tokenizer(te_path)
        else:
            tokenizer = load_tokenizer(te_path)
            text_encoder = load_text_encoder(te_path, dtype)
    elif te_path != model_path:
        # Separate path provided, treat as HF repo
        print(f"Loading text encoder from HF repo: {te_path}")
        tokenizer = load_tokenizer(te_path)
        text_encoder = load_text_encoder(te_path, dtype)
    else:
        # Use default base model
        tokenizer = load_tokenizer(DEFAULT_ZIMAGE_BASE_MODEL)
        text_encoder = load_text_encoder(DEFAULT_ZIMAGE_BASE_MODEL, dtype)
    
    return tokenizer, text_encoder


def load_transformer_and_scheduler(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[ZImageTransformer2DModel, FlowMatchEulerDiscreteScheduler]:
    """
    Load only transformer and scheduler for training.
    Used for staged loading to save VRAM.
    """
    model_type = get_model_type(model_path)
    
    if model_type == "diffusers" and os.path.isdir(model_path):
        # Check if local safetensors
        transformer_file = find_safetensors_in_subfolder(model_path, "transformer")
        if transformer_file:
            print(f"Loading transformer from local safetensors: {transformer_file}")
            transformer = load_transformer_from_local_safetensors(transformer_file, dtype)
            scheduler = create_default_scheduler()
        else:
            transformer = load_transformer(model_path, dtype)
            scheduler = load_scheduler(model_path)
    elif model_type == "single_file":
        # ZImageTransformer2DModel doesn't support from_single_file, use local loader
        print(f"Loading transformer from single file: {model_path}")
        transformer = load_transformer_from_local_safetensors(model_path, dtype)
        scheduler = create_default_scheduler()
    else:
        transformer = load_transformer(DEFAULT_ZIMAGE_BASE_MODEL, dtype)
        scheduler = create_default_scheduler()
    
    return transformer, scheduler


def load_local_safetensors_model(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[ZImageTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler, AutoTokenizer, PreTrainedModel]:
    """
    Load Z-Image model from a directory containing safetensors files in subfolders.
    
    Expected structure:
        model_path/
        ├── transformer/*.safetensors
        ├── text_encoder/*.safetensors  
        └── vae/*.safetensors
    
    Tokenizer and scheduler are loaded from the default base model.
    """
    print(f"Loading local safetensors model from: {model_path}")
    
    # Find safetensors files in each subfolder
    transformer_file = find_safetensors_in_subfolder(model_path, "transformer")
    text_encoder_file = find_safetensors_in_subfolder(model_path, "text_encoder")
    vae_file = find_safetensors_in_subfolder(model_path, "vae")
    
    if not transformer_file:
        raise ValueError(f"No transformer safetensors found in {model_path}/transformer/")
    if not text_encoder_file:
        raise ValueError(f"No text_encoder safetensors found in {model_path}/text_encoder/")
    if not vae_file:
        raise ValueError(f"No vae safetensors found in {model_path}/vae/")
    
    # Load components from local safetensors
    transformer = load_transformer_from_local_safetensors(transformer_file, dtype)
    text_encoder = load_text_encoder_from_local_safetensors(text_encoder_file, dtype)
    vae = load_vae_from_local_safetensors(vae_file, dtype)
    
    # Load tokenizer from default base model
    tokenizer = load_tokenizer(DEFAULT_ZIMAGE_BASE_MODEL)
    
    # Create default scheduler
    scheduler = create_default_scheduler()
    
    return transformer, vae, scheduler, tokenizer, text_encoder


def load_diffusers_model(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[ZImageTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler, AutoTokenizer, PreTrainedModel]:
    """
    Load Z-Image model from diffusers format (directory with subfolders).
    
    Returns:
        transformer, vae, scheduler, tokenizer, text_encoder
    """
    tokenizer = load_tokenizer(model_path)
    text_encoder = load_text_encoder(model_path, dtype)
    transformer = load_transformer(model_path, dtype)
    vae = load_vae(model_path, dtype)
    scheduler = load_scheduler(model_path)
    
    return transformer, vae, scheduler, tokenizer, text_encoder


def load_single_file_model(
    checkpoint_path: str,
    dtype: torch.dtype = torch.bfloat16,
    base_model_path: Optional[str] = None,
) -> Tuple[ZImageTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler, AutoTokenizer, PreTrainedModel]:
    """
    Load Z-Image model from a single safetensors/ckpt file.
    
    Note: Single file only contains the transformer weights. 
    Tokenizer, text_encoder, VAE, and scheduler are loaded from base model.
    
    Args:
        checkpoint_path: Path to the .safetensors or .ckpt file
        dtype: Model dtype
        base_model_path: Path to base model for loading other components (default: Z-a-o/Z-Image-Turbo)
    
    Returns:
        transformer, vae, scheduler, tokenizer, text_encoder
    """
    if base_model_path is None:
        base_model_path = DEFAULT_ZIMAGE_BASE_MODEL
    
    # Load transformer from single file
    transformer = load_transformer_from_single_file(checkpoint_path, dtype)
    
    # Load other components from base model
    tokenizer = load_tokenizer(base_model_path)
    text_encoder = load_text_encoder(base_model_path, dtype)
    vae = load_vae(base_model_path, dtype)
    
    # Try to load scheduler from base model, fallback to default
    try:
        scheduler = load_scheduler(base_model_path)
    except Exception:
        scheduler = create_default_scheduler()
    
    return transformer, vae, scheduler, tokenizer, text_encoder


def load_model(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    base_model_path: Optional[str] = None,
    use_local_safetensors: bool = True,
) -> Tuple[ZImageTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler, AutoTokenizer, PreTrainedModel]:
    """
    Load all Z-Image model components.
    
    Supports:
    - Local directory with safetensors (transformer/, vae/, text_encoder/ subfolders)
    - Single file (.safetensors or .ckpt)
    - Diffusers format from HuggingFace Hub
    
    Args:
        model_path: Path to model directory, single checkpoint file, or HuggingFace model ID
        dtype: Model dtype (default: bfloat16)
        base_model_path: For single file loading, path to base model for tokenizer/text_encoder/VAE
        use_local_safetensors: If True, load local safetensors directly instead of using diffusers
    
    Returns:
        transformer, vae, scheduler, tokenizer, text_encoder
    """
    model_type = get_model_type(model_path)
    print_directory_info(model_path)
    
    if model_type == "single_file":
        return load_single_file_model(model_path, dtype, base_model_path)
    elif model_type == "diffusers":
        if use_local_safetensors and os.path.isdir(model_path):
            # Load directly from local safetensors files
            print("Loading from local safetensors files...")
            return load_local_safetensors_model(model_path, dtype)
        else:
            # Use diffusers from_pretrained (for HuggingFace Hub)
            return load_diffusers_model(model_path, dtype)
    else:
        # Try as HuggingFace model ID
        print(f"Trying to load as HuggingFace model: {model_path}")
        return load_diffusers_model(model_path, dtype)


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """
    Calculate shift value for timestep scheduling.
    Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def encode_prompt_zimage(
    tokenizer: AutoTokenizer,
    text_encoder: PreTrainedModel,
    prompt: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    max_sequence_length: int = 512,
) -> torch.FloatTensor:
    """
    Encode a single prompt using Z-Image's chat template format.
    
    Returns:
        prompt_embeds: [seq_len, hidden_dim] - variable length embeddings
    """
    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    prompt_formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    
    # Tokenize
    text_inputs = tokenizer(
        prompt_formatted,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    
    text_input_ids = text_inputs.input_ids.to(device)
    prompt_mask = text_inputs.attention_mask.to(device).bool()
    
    # Encode
    with torch.no_grad():
        outputs = text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_mask,
            output_hidden_states=True,
        )
        prompt_embeds = outputs.hidden_states[-2]  # penultimate layer
    
    # Extract only non-padded embeddings
    # Return shape: [1, seq_len, hidden_dim] where seq_len is variable
    embeddings = prompt_embeds[0][prompt_mask[0]].unsqueeze(0)
    
    return embeddings, prompt_mask


def encode_prompts_zimage(
    tokenizer: AutoTokenizer,
    text_encoder: PreTrainedModel,
    prompts: list[str],
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    max_sequence_length: int = 512,
    num_images_per_prompt: int = 1,
) -> Tuple[list[torch.FloatTensor], list[torch.BoolTensor]]:
    """
    Encode multiple prompts using Z-Image's format.
    
    Returns:
        prompt_embeds_list: list of [seq_len, hidden_dim] tensors
        prompt_masks_list: list of attention masks
    """
    prompt_embeds_list = []
    prompt_masks_list = []
    
    for prompt in prompts:
        embeds, mask = encode_prompt_zimage(
            tokenizer,
            text_encoder,
            prompt,
            device,
            dtype,
            max_sequence_length,
        )
        prompt_embeds_list.append(embeds)
        prompt_masks_list.append(mask)
    
    return prompt_embeds_list, prompt_masks_list
