from typing import Optional, Union
import inspect
import torch

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, SchedulerMixin

from model_util import SDXL_TEXT_ENCODER_TYPE
from hunyuan_models import HunYuanDiT
from hunyuan_utils import calc_rope

from tqdm import tqdm

UNET_IN_CHANNELS = 4  # Stable Diffusion の in_channels は 4 で固定。XLも同じ。
VAE_SCALE_FACTOR = 8  # 2 ** (len(vae.config.block_out_channels) - 1) = 8

UNET_ATTENTION_TIME_EMBED_DIM = 256  # XL
TEXT_ENCODER_2_PROJECTION_DIM = 1280
UNET_PROJECTION_CLASS_EMBEDDING_INPUT_DIM = 2816


def get_random_noise(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    generator: torch.Generator = None,
) -> torch.Tensor:
    return torch.randn(
        (
            batch_size,
            UNET_IN_CHANNELS,
            height// VAE_SCALE_FACTOR,  # 縦と横これであってるのかわからないけど、どっちにしろ大きな問題は発生しないのでこれでいいや
            width // VAE_SCALE_FACTOR,
        ),
        device=device,
        generator=generator,
    )


# https://www.crosslabs.org/blog/diffusion-with-offset-noise
def apply_noise_offset(latents: torch.FloatTensor, noise_offset: float):
    latents = latents + noise_offset * torch.randn(
        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
    )
    return latents

# https://arxiv.org/abs/2305.08891
def fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler):
    # fix beta: zero terminal SNR

    def enforce_zero_terminal_snr(betas):
        # Convert betas to alphas_bar_sqrt
        alphas = 1 - betas
        alphas_bar = alphas.cumprod(0)
        alphas_bar_sqrt = alphas_bar.sqrt()

        # Store old values.
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
        # Shift so last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T
        # Scale so first timestep is back to old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

        # Convert alphas_bar_sqrt to betas
        alphas_bar = alphas_bar_sqrt**2
        alphas = alphas_bar[1:] / alphas_bar[:-1]
        alphas = torch.cat([alphas_bar[0:1], alphas])
        betas = 1 - alphas
        return betas

    betas = noise_scheduler.betas
    betas = enforce_zero_terminal_snr(betas)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # logger.info(f"original: {noise_scheduler.betas}")
    # logger.info(f"fixed: {betas}")

    noise_scheduler.betas = betas
    noise_scheduler.alphas = alphas
    noise_scheduler.alphas_cumprod = alphas_cumprod


def get_initial_latents(
    scheduler: SchedulerMixin,
    n_imgs: int,
    height: int,
    width: int,
    n_prompts: int,
    device: torch.device,
    generator=None,
) -> torch.Tensor:
    noise = get_random_noise(n_imgs, height, width, device, generator=generator).repeat(
        n_prompts, 1, 1, 1
    )

    latents = noise * torch.tensor(scheduler.init_noise_sigma).to(device)

    return latents


def prepare_extra_step_kwargs(scheduler, generator, eta):
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]

    accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    # check if the scheduler accepts generator
    accepts_generator = "generator" in set(
        inspect.signature(scheduler.step).parameters.keys()
    )
    if accepts_generator:
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs


def text_tokenize(
    tokenizer: CLIPTokenizer,  # 普通ならひとつ、XLならふたつ！
    prompts: list[str],
):
    return tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids


def text_encode(text_encoder: CLIPTextModel, tokens):
    return text_encoder(tokens.to(text_encoder.device))[0]


def encode_prompts(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTokenizer,
    prompts: list[str],
):

    text_tokens = text_tokenize(tokenizer, prompts)
    text_embeddings = text_encode(text_encoder, text_tokens)

    return text_embeddings


# https://github.com/huggingface/diffusers/blob/78922ed7c7e66c20aa95159c7b7a6057ba7d590d/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L334-L348
def text_encode_xl(
    text_encoder: SDXL_TEXT_ENCODER_TYPE,
    tokens: torch.FloatTensor,
    num_images_per_prompt: int = 1,
):
    prompt_embeds = text_encoder(
        tokens.to(text_encoder.device), output_hidden_states=True
    )
    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]  # always penultimate layer

    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


def encode_prompts_xl(
    tokenizers: list[CLIPTokenizer],
    text_encoders: list[SDXL_TEXT_ENCODER_TYPE],
    prompts: list[str],
    num_images_per_prompt: int = 1,
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    # text_encoder and text_encoder_2's penuultimate layer's output
    text_embeds_list = []
    pooled_text_embeds = None  # always text_encoder_2's pool

    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_tokens_input_ids = text_tokenize(tokenizer, prompts)
        text_embeds, pooled_text_embeds = text_encode_xl(
            text_encoder, text_tokens_input_ids, num_images_per_prompt
        )

        text_embeds_list.append(text_embeds)

    bs_embed = pooled_text_embeds.shape[0]
    pooled_text_embeds = pooled_text_embeds.repeat(1, num_images_per_prompt).view(
        bs_embed * num_images_per_prompt, -1
    )

    return torch.concat(text_embeds_list, dim=-1), pooled_text_embeds


# https://github.com/huggingface/diffusers/blob/00d8d46e23b8b741a451db5e711969c46af127fd/src/diffusers/pipelines/hunyuandit/pipeline_hunyuandit.py#L242-L402
def encode_prompts_hunyuan(
    tokenizers,
    text_encoders,
    prompt: list[str],
    num_images_per_prompt: int = 1,
):
    prompt_embeds_list = []
    prompt_attention_masks = []

    dtype = text_encoders[1].torch_dtype
    device = text_encoders[0].device
    max_lengths = [227, 256]

    for tokenizer, text_encoder, max_length in zip(
        tokenizers, text_encoders, max_lengths
    ):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = tokenizer.batch_decode(
                untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
            )

        prompt_attention_mask = text_inputs.attention_mask.to(device)
        prompt_embeds = text_encoder(
            text_input_ids.to(device),
            attention_mask=prompt_attention_mask,
        )
        prompt_embeds = prompt_embeds[0]
        prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1).view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        prompt_embeds_list.append(prompt_embeds)
        prompt_attention_masks.append(prompt_attention_mask)

    return prompt_embeds_list, prompt_attention_masks


def concat_embeddings(
    unconditional: torch.FloatTensor,
    conditional: torch.FloatTensor,
    n_imgs: int,
):
    return torch.cat([unconditional, conditional]).repeat_interleave(n_imgs, dim=0)


def concat_embeddings_tuple(
    unconditional: tuple[torch.FloatTensor, torch.FloatTensor],
    conditional: tuple[torch.FloatTensor, torch.FloatTensor],
    n_imgs: int,
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    # 解包 tuple
    unconditional_1, unconditional_2 = unconditional
    conditional_1, conditional_2 = conditional

    # 拼接和重复操作
    concatenated_1 = torch.cat([unconditional_1, conditional_1]).repeat_interleave(
        n_imgs, dim=0
    )
    concatenated_2 = torch.cat([unconditional_2, conditional_2]).repeat_interleave(
        n_imgs, dim=0
    )

    return concatenated_1, concatenated_2


# ref: https://github.com/huggingface/diffusers/blob/0bab447670f47c28df60fbd2f6a0f833f75a16f5/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L721
def predict_noise(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    timestep: int,  # 現在のタイムステップ
    latents: torch.FloatTensor,
    text_embeddings: torch.FloatTensor,  # uncond な text embed と cond な text embed を結合したもの
    guidance_scale=7.5,
) -> torch.FloatTensor:
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

    # predict the noise residual
    noise_pred = unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=text_embeddings,
    ).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    guided_target = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )

    return guided_target


# ref: https://github.com/huggingface/diffusers/blob/0bab447670f47c28df60fbd2f6a0f833f75a16f5/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L746
@torch.no_grad()
def diffusion(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    latents: torch.FloatTensor,  # ただのノイズだけのlatents
    text_embeddings: torch.FloatTensor,
    total_timesteps: int = 1000,
    start_timesteps=0,
    **kwargs,
):
    # latents_steps = []

    for timestep in tqdm(scheduler.timesteps[start_timesteps:total_timesteps]):
        noise_pred = predict_noise(
            unet, scheduler, timestep, latents, text_embeddings, **kwargs
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    # return latents_steps
    return latents


def rescale_noise_cfg(
    noise_cfg: torch.FloatTensor, noise_pred_text, guidance_rescale=0.0
):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True
    )
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = (
        guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    )

    return noise_cfg


def predict_noise_xl(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    timestep: int,  # 現在のタイムステップ
    latents: torch.FloatTensor,
    text_embeddings: torch.FloatTensor,  # uncond な text embed と cond な text embed を結合したもの
    add_text_embeddings: torch.FloatTensor,  # pooled なやつ
    add_time_ids: torch.FloatTensor,
    guidance_scale=7.5,
    guidance_rescale=0.7,
) -> torch.FloatTensor:
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

    added_cond_kwargs = {
        "text_embeds": add_text_embeddings,
        "time_ids": add_time_ids,
    }

    # predict the noise residual
    noise_pred = unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=text_embeddings,
        added_cond_kwargs=added_cond_kwargs,
    ).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    guided_target = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )

    # https://github.com/huggingface/diffusers/blob/7a91ea6c2b53f94da930a61ed571364022b21044/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L775
    noise_pred = rescale_noise_cfg(
        guided_target, noise_pred_text, guidance_rescale=guidance_rescale
    )

    return guided_target


@torch.no_grad()
def diffusion_xl(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    latents: torch.FloatTensor,  # ただのノイズだけのlatents
    text_embeddings: tuple[torch.FloatTensor, torch.FloatTensor],
    add_text_embeddings: torch.FloatTensor,  # pooled なやつ
    add_time_ids: torch.FloatTensor,
    guidance_scale: float = 1.0,
    total_timesteps: int = 1000,
    start_timesteps=0,
):
    # latents_steps = []

    for timestep in tqdm(scheduler.timesteps[start_timesteps:total_timesteps]):
        noise_pred = predict_noise_xl(
            unet,
            scheduler,
            timestep,
            latents,
            text_embeddings,
            add_text_embeddings,
            add_time_ids,
            guidance_scale=guidance_scale,
            guidance_rescale=0.7,
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    # return latents_steps
    return latents


def predict_noise_hunyuan(
    unet: HunYuanDiT,
    scheduler: SchedulerMixin,
    timestep: int,  # 現在のタイムステップ
    latents: torch.FloatTensor,
    text_embeddings: tuple[
        torch.FloatTensor, torch.FloatTensor
    ],  # uncond な text embed と cond な text embed を結合したもの
    prompt_attention_masks: tuple[torch.FloatTensor, torch.FloatTensor],
    guidance_scale=5.0,
    guidance_rescale=0.7,
) -> torch.FloatTensor:
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

    # expand scalar t to 1-D tensor to match the 1st dim of latent_model_input
    t_expand = torch.tensor([timestep] * latent_model_input.shape[0], device=latent_model_input.device).to(
        dtype=latent_model_input.dtype
    )

    B, C, H, W = latents.shape
    freqs_cis_img = calc_rope(H * 8, W * 8, 2, 88)

    # predict the noise residual
    noise_pred = unet(
        latent_model_input,
        t_expand,
        encoder_hidden_states=text_embeddings[0],
        text_embedding_mask=prompt_attention_masks[0],
        encoder_hidden_states_t5=text_embeddings[1],
        text_embedding_mask_t5=prompt_attention_masks[1],
        image_meta_size=None,
        style=None,
        cos_cis_img=freqs_cis_img[0],
        sin_cis_img=freqs_cis_img[1],
    )

    # dim / 2
    noise_pred, _ = noise_pred.chunk(2, dim=1)

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    guided_target = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )

    if guidance_rescale > 0.0:
        # https://github.com/huggingface/diffusers/blob/7a91ea6c2b53f94da930a61ed571364022b21044/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L775
        rescale_target = rescale_noise_cfg(
            guided_target, noise_pred_text, guidance_rescale=guidance_rescale
        )

    return rescale_target


@torch.no_grad()
def diffusion_hunyuan(
    unet: HunYuanDiT,
    scheduler: SchedulerMixin,
    latents: torch.FloatTensor,  # ただのノイズだけのlatents
    text_embeddings: tuple[torch.FloatTensor, torch.FloatTensor],
    prompt_attention_masks: tuple[torch.FloatTensor, torch.FloatTensor],
    guidance_scale: float = 1.0,
    total_timesteps: int = 1000,
    start_timesteps=0,
):

    extra_step_kwargs = prepare_extra_step_kwargs(
        scheduler,
        None,
        0.0 if scheduler.__class__.__name__ == "DDIMScheduler" else None,
    )

    for timestep in tqdm(scheduler.timesteps[start_timesteps:total_timesteps]):
        noise_pred = predict_noise_hunyuan(
            unet,
            scheduler,
            timestep,
            latents,
            text_embeddings,
            prompt_attention_masks,
            guidance_scale=guidance_scale,
            guidance_rescale=0.7,
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(
            noise_pred, timestep, latents, **extra_step_kwargs, return_dict=False
        )[0]

    return latents


# for XL
def get_add_time_ids(
    height: int,
    width: int,
    dynamic_crops: bool = False,
    dtype: torch.dtype = torch.float32,
):
    if dynamic_crops:
        # random float scale between 1 and 3
        random_scale = torch.rand(1).item() * 2 + 1
        original_size = (int(height * random_scale), int(width * random_scale))
        # random position
        crops_coords_top_left = (
            torch.randint(0, original_size[0] - height, (1,)).item(),
            torch.randint(0, original_size[1] - width, (1,)).item(),
        )
        target_size = (height, width)
    else:
        original_size = (height, width)
        crops_coords_top_left = (0, 0)
        target_size = (height, width)

    # this is expected as 6
    add_time_ids = list(original_size + crops_coords_top_left + target_size)

    # this is expected as 2816
    passed_add_embed_dim = (
        UNET_ATTENTION_TIME_EMBED_DIM * len(add_time_ids)  # 256 * 6
        + TEXT_ENCODER_2_PROJECTION_DIM  # + 1280
    )
    if passed_add_embed_dim != UNET_PROJECTION_CLASS_EMBEDDING_INPUT_DIM:
        raise ValueError(
            f"Model expects an added time embedding vector of length {UNET_PROJECTION_CLASS_EMBEDDING_INPUT_DIM}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        )

    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    return add_time_ids


def get_optimizer(name: str):
    name = name.lower()

    if name.startswith("dadapt"):
        import dadaptation

        if name == "dadaptadam":
            return dadaptation.DAdaptAdam
        elif name == "dadaptlion":
            return dadaptation.DAdaptLion
        else:
            raise ValueError("DAdapt optimizer must be dadaptadam or dadaptlion")

    elif name.endswith("8bit"):  # 検証してない
        import bitsandbytes as bnb

        if name == "adam8bit":
            return bnb.optim.Adam8bit
        elif name == "lion8bit":
            return bnb.optim.Lion8bit
        elif name == "ademamix8bit":
            return bnb.optim.AdEMAMix8bit
        else:
            raise ValueError("8bit optimizer must be adam8bit or lion8bit or ademamix8bit")

    else:
        if name == "adam":
            return torch.optim.Adam
        elif name == "adamw":
            return torch.optim.AdamW
        elif name == "lion":
            from lion_pytorch import Lion

            return Lion
        elif name == "prodigy":
            import prodigyopt

            return prodigyopt.Prodigy
        else:
            raise ValueError("Optimizer must be adam, adamw, lion or Prodigy")


def get_lr_scheduler(
    name: Optional[str],
    optimizer: torch.optim.Optimizer,
    max_iterations: Optional[int],
    lr_min: Optional[float],
    **kwargs,
):
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_iterations, eta_min=lr_min, **kwargs
        )
    elif name == "cosine_with_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max_iterations // 10, T_mult=2, eta_min=lr_min, **kwargs
        )
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max_iterations // 100, gamma=0.999, **kwargs
        )
    elif name == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, **kwargs)
    elif name == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer, factor=0.5, total_iters=max_iterations // 100, **kwargs
        )
    else:
        raise ValueError(
            "Scheduler must be cosine, cosine_with_restarts, step, linear or constant"
        )


def get_random_resolution_in_bucket(bucket_resolution: int = 512) -> tuple[int, int]:
    max_resolution = bucket_resolution
    min_resolution = bucket_resolution // 2

    step = 64

    min_step = min_resolution // step
    max_step = max_resolution // step

    # +1 to include max_resolution (randint upper bound is exclusive)
    height = torch.randint(min_step, max_step + 1, (1,)).item() * step
    width = torch.randint(min_step, max_step + 1, (1,)).item() * step

    return height, width


# ==================== Z-Image Functions ====================

ZIMAGE_VAE_SCALE_FACTOR = 16  # vae_scale_factor * 2 for Z-Image
ZIMAGE_IN_CHANNELS = 16  # Z-Image transformer in_channels


def get_random_noise_zimage(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """Get random noise for Z-Image latent space."""
    return torch.randn(
        (
            batch_size,
            ZIMAGE_IN_CHANNELS,
            height // ZIMAGE_VAE_SCALE_FACTOR,
            width // ZIMAGE_VAE_SCALE_FACTOR,
        ),
        device=device,
        generator=generator,
        dtype=torch.float32,  # Z-Image uses float32 for latents
    )


def get_initial_latents_zimage(
    n_imgs: int,
    height: int,
    width: int,
    n_prompts: int,
    device: torch.device,
    generator=None,
) -> torch.Tensor:
    """Get initial latents for Z-Image (no scheduler scaling needed for flow matching)."""
    noise = get_random_noise_zimage(n_imgs, height, width, device, generator=generator)
    latents = noise.repeat(n_prompts, 1, 1, 1)
    return latents


def calculate_shift_zimage(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Calculate shift value for Z-Image timestep scheduling."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def predict_noise_zimage(
    transformer,
    scheduler: SchedulerMixin,
    timestep: torch.Tensor,  # normalized timestep (1000 - t) / 1000
    latents: torch.FloatTensor,
    prompt_embeds_list: list[torch.FloatTensor],  # list of variable-length embeddings
    guidance_scale: float = 0.0,
) -> torch.FloatTensor:
    """
    Predict noise for Z-Image using flow matching.
    
    Args:
        transformer: Z-Image transformer model
        scheduler: FlowMatchEulerDiscreteScheduler
        timestep: normalized timestep tensor
        latents: input latents [B, C, H, W]
        prompt_embeds_list: list of prompt embeddings for each batch item
        guidance_scale: CFG scale (0 for no CFG)
    
    Returns:
        noise prediction
    """
    dtype = latents.dtype
    batch_size = latents.shape[0]
    
    do_cfg = guidance_scale > 0
    
    if do_cfg:
        # For CFG, we need to double the batch
        latent_model_input = latents.repeat(2, 1, 1, 1)
        timestep_input = timestep.repeat(2)
        # prompt_embeds should be [positive, negative] for each item
        # Assuming prompt_embeds_list contains [pos_0, ..., pos_n, neg_0, ..., neg_n]
    else:
        latent_model_input = latents
        timestep_input = timestep
    
    # Z-Image expects latents with extra dim: [B, C, 1, H, W] -> list of [C, 1, H, W]
    latent_model_input = latent_model_input.unsqueeze(2)
    latent_model_input_list = list(latent_model_input.unbind(dim=0))
    
    # Forward pass through transformer
    model_out_list = transformer(
        latent_model_input_list,
        timestep_input,
        prompt_embeds_list,
    )[0]
    
    if do_cfg:
        # Perform CFG
        pos_out = model_out_list[:batch_size]
        neg_out = model_out_list[batch_size:]
        
        noise_pred = []
        for j in range(batch_size):
            pos = pos_out[j].float()
            neg = neg_out[j].float()
            pred = pos + guidance_scale * (pos - neg)
            noise_pred.append(pred)
        
        noise_pred = torch.stack(noise_pred, dim=0)
    else:
        noise_pred = torch.stack([t.float() for t in model_out_list], dim=0)
    
    noise_pred = noise_pred.squeeze(2)
    # Z-Image requires negating the noise prediction
    noise_pred = -noise_pred
    
    return noise_pred


@torch.no_grad()
def diffusion_zimage(
    transformer,
    scheduler: SchedulerMixin,
    latents: torch.FloatTensor,
    prompt_embeds_list: list[torch.FloatTensor],
    negative_prompt_embeds_list: list[torch.FloatTensor] = None,
    num_inference_steps: int = 30,
    guidance_scale: float = 0.0,
    start_step: int = 0,
    total_steps: int = None,
) -> torch.FloatTensor:
    """
    Run Z-Image diffusion process using flow matching.
    
    Args:
        transformer: Z-Image transformer model
        scheduler: FlowMatchEulerDiscreteScheduler
        latents: initial noise latents
        prompt_embeds_list: list of prompt embeddings
        negative_prompt_embeds_list: list of negative prompt embeddings (for CFG)
        num_inference_steps: number of denoising steps
        guidance_scale: CFG scale
        start_step: starting step index
        total_steps: ending step index
    
    Returns:
        denoised latents
    """
    batch_size = latents.shape[0]
    device = latents.device
    dtype = latents.dtype
    
    if total_steps is None:
        total_steps = num_inference_steps
    
    # Calculate mu for timestep scheduling
    image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
    mu = calculate_shift_zimage(
        image_seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    
    # Set timesteps
    scheduler.sigma_min = 0.0
    scheduler.set_timesteps(num_inference_steps, device=device, mu=mu)
    timesteps = scheduler.timesteps
    
    do_cfg = guidance_scale > 0 and negative_prompt_embeds_list is not None
    
    for i, t in enumerate(timesteps[start_step:total_steps]):
        # Normalized timestep for Z-Image: (1000 - t) / 1000
        timestep = t.expand(batch_size)
        timestep_normalized = (1000 - timestep) / 1000
        
        if do_cfg:
            combined_embeds = prompt_embeds_list + negative_prompt_embeds_list
        else:
            combined_embeds = prompt_embeds_list
        
        noise_pred = predict_noise_zimage(
            transformer,
            scheduler,
            timestep_normalized,
            latents if latents.dtype == dtype else latents.to(dtype),
            combined_embeds,
            guidance_scale=guidance_scale if do_cfg else 0.0,
        )
        
        # Scheduler step
        latents = scheduler.step(noise_pred.to(torch.float32), t, latents, return_dict=False)[0]
    
    return latents


def concat_embeddings_zimage(
    unconditional_embeds: torch.FloatTensor,
    conditional_embeds: torch.FloatTensor,
    n_imgs: int,
) -> list[torch.FloatTensor]:
    """
    Concatenate unconditional and conditional embeddings for Z-Image CFG.
    Returns a list suitable for Z-Image transformer input.
    """
    # For Z-Image, we return a list: [cond_0, ..., cond_n, uncond_0, ..., uncond_n]
    result = []
    for _ in range(n_imgs):
        result.append(conditional_embeds)
    for _ in range(n_imgs):
        result.append(unconditional_embeds)
    return result
