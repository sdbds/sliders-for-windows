# ref:
# - https://github.com/JerryWu-code/diffusers/blob/1048d0a94d553978b35bec0942c408c57bb12c9e/src/diffusers/pipelines/z_image/pipeline_z_image.py
# - https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion/blob/main/train.py

import argparse
import ast
from pathlib import Path
import gc

import torch
from tqdm import tqdm


from lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV
from sai_model_spec import build_metadata
import time
import train_util
import zimage_utils
import prompt_util
from prompt_util import (
    PromptEmbedsCache,
    PromptEmbedsPair,
    PromptSettings,
    PromptEmbedsZImage,
)
import debug_util
import config_util
from config_util import RootConfig

import wandb

NUM_IMAGES_PER_PROMPT = 1


def flush():
    torch.cuda.empty_cache()
    gc.collect()


def train(
    config: RootConfig,
    prompts: list[PromptSettings],
    device,
):
    # Enable TF32 for faster training on Ampere+ GPUs
    # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    metadata = {
        "prompts": ",".join([prompt.json() for prompt in prompts]),
        "config": config.json(),
    }
    save_path = Path(config.save.path)

    modules = DEFAULT_TARGET_REPLACE
    if config.network.type == "c3lier":
        modules += UNET_TARGET_REPLACE_MODULE_CONV

    if config.logging.verbose:
        print(metadata)

    if config.logging.use_wandb:
        wandb.init(project=f"LECO_{config.save.name}", config=metadata)

    metadata.update(
        build_metadata(
            v2=config.pretrained_model.v2,
            v_parameterization=config.pretrained_model.v_pred,
            sdxl=False,
            timestamp=time.time(),
            title="textsliders",
            zimage=config.pretrained_model.zimage or "turbo",
        )
    )

    weight_dtype = config_util.parse_precision(config.train.precision)
    
    # Text encoder always uses bfloat16 (more stable for embeddings)
    text_encoder_dtype = torch.bfloat16
    
    # Transformer dtype: use config precision, but convert fp16 to bf16
    transformer_dtype = weight_dtype
    if transformer_dtype == torch.float16:
        transformer_dtype = torch.bfloat16
    
    # Save weights in bfloat16 (FP8 not suitable for LoRA weights)
    save_weight_dtype = torch.bfloat16
    
    print(f"Text encoder dtype: {text_encoder_dtype}")
    print(f"Transformer dtype: {transformer_dtype}")
    print(f"Save dtype: {save_weight_dtype}")

    # Z-Image specific parameters
    max_sequence_length = config.train.max_sequence_length
    model_path = config.pretrained_model.name_or_path
    text_encoder_path = config.pretrained_model.text_encoder_path
    vae_path = config.pretrained_model.vae_path

    # ========== Stage 1: Load text encoder, encode prompts, then free memory ==========
    print("Stage 1: Loading tokenizer and text encoder...")
    tokenizer, text_encoder = zimage_utils.load_tokenizer_and_text_encoder(
        model_path=model_path,
        dtype=text_encoder_dtype,
        text_encoder_path=text_encoder_path,
    )

    # Move text encoder to device
    text_encoder.to(device, dtype=text_encoder_dtype)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    print("Prompts")
    for settings in prompts:
        print(settings)

    # Cache all prompt embeddings
    cache = PromptEmbedsCache()
    prompt_pairs: list[PromptEmbedsPair] = []
    criteria = torch.nn.MSELoss()

    with torch.no_grad():
        for settings in prompts:
            print(settings)
            for prompt in [
                settings.target,
                settings.positive,
                settings.neutral,
                settings.unconditional,
            ]:
                if cache[prompt] is None:
                    prompt_embeds, prompt_attention_mask = zimage_utils.encode_prompt_zimage(
                        tokenizer,
                        text_encoder,
                        prompt,
                        device=device,
                        dtype=transformer_dtype,
                        max_sequence_length=max_sequence_length,
                    )
                    cache[prompt] = PromptEmbedsZImage(prompt_embeds, prompt_attention_mask)

            prompt_pairs.append(
                PromptEmbedsPair(
                    criteria,
                    cache[settings.target],
                    cache[settings.positive],
                    cache[settings.unconditional],
                    cache[settings.neutral],
                    settings,
                )
            )

    # Free text encoder memory
    print("Freeing text encoder memory...")
    del tokenizer, text_encoder
    flush()

    # ========== Stage 2: Load transformer and start training ==========
    print("Stage 2: Loading transformer and scheduler...")
    transformer, scheduler = zimage_utils.load_transformer_and_scheduler(
        model_path=model_path,
        dtype=transformer_dtype,
    )

    # Move transformer to device
    transformer.to(device, dtype=transformer_dtype)
    if config.other.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    transformer.requires_grad_(False)
    transformer.eval()

    # Compile transformer for faster inference
    if config.other.torch_compile:
        # Check if any prompt uses dynamic_resolution
        use_dynamic = any(p.dynamic_resolution for p in prompts)
        print(f"Compiling transformer with mode: {config.other.torch_compile_mode}, dynamic={use_dynamic}")
        transformer = torch.compile(
            transformer,
            mode=config.other.torch_compile_mode,
            fullgraph=False,  # LoRA hooks prevent fullgraph
            dynamic=use_dynamic,  # Enable dynamic shapes if any prompt uses dynamic_resolution
        )

    # Create LoRA network for transformer
    network = LoRANetwork(
        transformer,
        rank=config.network.rank,
        multiplier=1.0,
        alpha=config.network.alpha,
        train_method=config.network.training_method,
    ).to(device, dtype=transformer_dtype)

    optimizer_module = train_util.get_optimizer(config.train.optimizer)
    # optimizer_args
    optimizer_kwargs = {}
    if config.train.optimizer_args is not None and len(config.train.optimizer_args) > 0:
        for arg in config.train.optimizer_args.split(" "):
            key, value = arg.split("=")
            value = ast.literal_eval(value)
            optimizer_kwargs[key] = value

    optimizer = optimizer_module(
        network.prepare_optimizer_params(), lr=config.train.lr, **optimizer_kwargs
    )
    lr_scheduler = train_util.get_lr_scheduler(
        config.train.lr_scheduler,
        optimizer,
        max_iterations=config.train.iterations,
        lr_min=config.train.lr / 100,
    )

    # debug
    debug_util.check_requires_grad(network)
    debug_util.check_training_mode(network)

    pbar = tqdm(range(config.train.iterations))

    loss = None

    for i in pbar:
        with torch.no_grad():
            optimizer.zero_grad()

            prompt_pair: PromptEmbedsPair = prompt_pairs[
                torch.randint(0, len(prompt_pairs), (1,)).item()
            ]

            # Random timestep selection (1 ~ max_denoising_steps-1)
            timesteps_to = torch.randint(
                1, config.train.max_denoising_steps, (1,)
            ).item()

            height, width = prompt_pair.resolution, prompt_pair.resolution
            if prompt_pair.dynamic_resolution:
                height, width = train_util.get_random_resolution_in_bucket(
                    prompt_pair.resolution
                )

            if config.logging.verbose:
                print("guidance_scale:", prompt_pair.guidance_scale)
                print("resolution:", prompt_pair.resolution)
                print("dynamic_resolution:", prompt_pair.dynamic_resolution)
                if prompt_pair.dynamic_resolution:
                    print("bucketed resolution:", (height, width))
                print("batch_size:", prompt_pair.batch_size)

            # Get initial latents for Z-Image (uses float32)
            latents = train_util.get_initial_latents_zimage(
                prompt_pair.batch_size, height, width, 1, device=device
            )

            # Calculate mu for timestep scheduling
            image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
            mu = train_util.calculate_shift_zimage(
                image_seq_len,
                scheduler.config.get("base_image_seq_len", 256),
                scheduler.config.get("max_image_seq_len", 4096),
                scheduler.config.get("base_shift", 0.5),
                scheduler.config.get("max_shift", 1.15),
            )

            # Set timesteps
            scheduler.sigma_min = 0.0
            scheduler.set_timesteps(config.train.max_denoising_steps, device=device, mu=mu)

            with network:
                # Denoise with LoRA enabled (target prompt)
                denoised_latents = diffusion_zimage_partial(
                    transformer,
                    scheduler,
                    latents,
                    prompt_pair.target.prompt_embeds,
                    start_step=0,
                    total_steps=timesteps_to,
                    batch_size=prompt_pair.batch_size,
                    device=device,
                    dtype=transformer_dtype,
                )

            # Get current timestep for noise prediction
            current_timestep_idx = timesteps_to
            current_timestep = scheduler.timesteps[current_timestep_idx]
            timestep_normalized = (1000 - current_timestep.expand(prompt_pair.batch_size)) / 1000

            # Predict noise without LoRA for different prompts
            positive_latents = predict_noise_zimage_single(
                transformer,
                timestep_normalized,
                denoised_latents,
                prompt_pair.positive.prompt_embeds,
                prompt_pair.batch_size,
                device,
                transformer_dtype,
            )

            neutral_latents = predict_noise_zimage_single(
                transformer,
                timestep_normalized,
                denoised_latents,
                prompt_pair.neutral.prompt_embeds,
                prompt_pair.batch_size,
                device,
                transformer_dtype,
            )

            unconditional_latents = predict_noise_zimage_single(
                transformer,
                timestep_normalized,
                denoised_latents,
                prompt_pair.unconditional.prompt_embeds,
                prompt_pair.batch_size,
                device,
                transformer_dtype,
            )

            if config.logging.verbose:
                print("positive_latents:", positive_latents[0, 0, :5, :5])
                print("neutral_latents:", neutral_latents[0, 0, :5, :5])
                print("unconditional_latents:", unconditional_latents[0, 0, :5, :5])

        # Predict noise with LoRA enabled (target prompt)
        with network:
            target_latents = predict_noise_zimage_single(
                transformer,
                timestep_normalized,
                denoised_latents,
                prompt_pair.target.prompt_embeds,
                prompt_pair.batch_size,
                device,
                transformer_dtype,
            )

            if config.logging.verbose:
                print("target_latents:", target_latents[0, 0, :5, :5])

        positive_latents.requires_grad = False
        neutral_latents.requires_grad = False
        unconditional_latents.requires_grad = False

        loss = prompt_pair.loss(
            target_latents=target_latents,
            positive_latents=positive_latents,
            neutral_latents=neutral_latents,
            unconditional_latents=unconditional_latents,
        )

        # Display loss * 1000 for visibility
        pbar.set_description(f"Loss*1k: {loss.item()*1000:.4f}")
        if config.logging.use_wandb:
            wandb.log(
                {"loss": loss, "iteration": i, "lr": lr_scheduler.get_last_lr()[0]}
            )

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        del (
            positive_latents,
            neutral_latents,
            unconditional_latents,
            target_latents,
            denoised_latents,
            latents,
        )
        flush()

        if (
            i % config.save.per_steps == 0
            and i != 0
            and i != config.train.iterations - 1
        ):
            print("Saving...")
            save_path.mkdir(parents=True, exist_ok=True)
            network.save_weights(
                save_path / f"{config.save.name}_{i}steps.safetensors",
                dtype=save_weight_dtype,
                metadata=metadata,
            )

    print("Saving...")
    save_path.mkdir(parents=True, exist_ok=True)
    network.save_weights(
        save_path / f"{config.save.name}_last.safetensors",
        dtype=save_weight_dtype,
        metadata=metadata,
    )

    del (
        transformer,
        scheduler,
        loss,
        optimizer,
        network,
    )

    flush()

    print("Done.")


def predict_noise_zimage_single(
    transformer,
    timestep_normalized: torch.Tensor,
    latents: torch.FloatTensor,
    prompt_embeds: torch.FloatTensor,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.FloatTensor:
    """
    Single forward pass for Z-Image noise prediction without CFG.
    """
    # Prepare prompt embeddings list for batch
    prompt_embeds_list = [prompt_embeds.squeeze(0)] * batch_size

    # Prepare latents: add extra dimension and convert to list
    latent_input = latents.to(dtype).unsqueeze(2)
    latent_input_list = list(latent_input.unbind(dim=0))

    # Forward pass
    model_out_list = transformer(
        latent_input_list,
        timestep_normalized,
        prompt_embeds_list,
    )[0]

    # Stack and process output
    noise_pred = torch.stack([t.float() for t in model_out_list], dim=0)
    noise_pred = noise_pred.squeeze(2)
    # Z-Image requires negating the noise prediction
    noise_pred = -noise_pred

    return noise_pred.to(device, dtype=dtype)


@torch.no_grad()
def diffusion_zimage_partial(
    transformer,
    scheduler,
    latents: torch.FloatTensor,
    prompt_embeds: torch.FloatTensor,
    start_step: int,
    total_steps: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.FloatTensor:
    """
    Partial diffusion process for Z-Image (without CFG for training).
    """
    # Prepare prompt embeddings list for batch
    prompt_embeds_list = [prompt_embeds.squeeze(0)] * batch_size

    timesteps = scheduler.timesteps

    for i, t in enumerate(timesteps[start_step:total_steps]):
        # Normalized timestep for Z-Image: (1000 - t) / 1000
        timestep_normalized = (1000 - t.expand(batch_size)) / 1000

        # Prepare latents
        latent_input = latents.to(dtype).unsqueeze(2)
        latent_input_list = list(latent_input.unbind(dim=0))

        # Forward pass
        model_out_list = transformer(
            latent_input_list,
            timestep_normalized,
            prompt_embeds_list,
        )[0]

        # Process output
        noise_pred = torch.stack([out.float() for out in model_out_list], dim=0)
        noise_pred = noise_pred.squeeze(2)
        noise_pred = -noise_pred

        # Scheduler step
        latents = scheduler.step(noise_pred.to(torch.float32), t, latents, return_dict=False)[0]

    return latents


def main(args):
    config_file = args.config_file

    config = config_util.load_config_from_yaml(config_file)
    if args.name is not None:
        config.save.name = args.name
    attributes = []
    if args.attributes is not None:
        attributes = args.attributes.split(",")
        attributes = [a.strip() for a in attributes]

    if args.prompts_file is not None:
        config.prompts_file = args.prompts_file
    if args.alpha is not None:
        config.network.alpha = args.alpha
    if args.rank is not None:
        config.network.rank = args.rank
    config.save.name += f"_alpha{config.network.alpha}"
    config.save.name += f"_rank{config.network.rank}"
    config.save.name += f"_{config.network.training_method}"
    config.save.path += f"/{config.save.name}"

    prompts = prompt_util.load_prompts_from_yaml(config.prompts_file, attributes)
    print(prompts)
    device = torch.device(f"cuda:{args.device}")
    train(config, prompts, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        required=True,
        help="Config file for training.",
    )
    parser.add_argument(
        "--prompts_file",
        required=False,
        help="Prompts file for training.",
        default=None,
    )
    # config_file 'data/config.yaml'
    parser.add_argument(
        "--alpha",
        type=float,
        required=False,
        default=None,
        help="LoRA weight.",
    )
    # --alpha 1.0
    parser.add_argument(
        "--rank",
        type=int,
        required=False,
        help="Rank of LoRA.",
        default=None,
    )
    # --rank 4
    parser.add_argument(
        "--device",
        type=int,
        required=False,
        default=0,
        help="Device to train on.",
    )
    # --device 0
    parser.add_argument(
        "--name",
        type=str,
        required=False,
        default=None,
        help="Name for the output model.",
    )
    # --name 'eyesize_slider'
    parser.add_argument(
        "--attributes",
        type=str,
        required=False,
        default=None,
        help="Attributes to disentangle (comma separated string)",
    )

    args = parser.parse_args()

    main(args)
