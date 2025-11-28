# based on https://github.com/Stability-AI/ModelSpec
import datetime
from typing import List, Optional, Tuple, Union

r"""
# Metadata Example
metadata = {
    # === Must ===
    "modelspec.sai_model_spec": "1.0.0", # Required version ID for the spec
    "modelspec.architecture": "stable-diffusion-xl-v1-base", # Architecture, reference the ID of the original model of the arch to match the ID
    "modelspec.implementation": "sgm",
    "modelspec.title": "Example Model Version 1.0", # Clean, human-readable title. May use your own phrasing/language/etc
    # === Should ===
    "modelspec.author": "Example Corp", # Your name or company name
    "modelspec.description": "This is my example model to show you how to do it!", # Describe the model in your own words/language/etc. Focus on what users need to know
    "modelspec.date": "2023-07-20", # ISO-8601 compliant date of when the model was created
    # === Can ===
    "modelspec.license": "ExampleLicense-1.0", # eg CreativeML Open RAIL, etc.
    "modelspec.usage_hint": "Use keyword 'example'" # In your own language, very short hints about how the user should use the model
}
"""

BASE_METADATA = {
    # === Must ===
    "modelspec.sai_model_spec": "1.0.0",  # Required version ID for the spec
    "modelspec.architecture": None,
    "modelspec.implementation": None,
    "modelspec.title": None,
    "modelspec.resolution": None,
    # === Should ===
    "modelspec.description": None,
    "modelspec.author": None,
    "modelspec.date": None,
    # === Can ===
    "modelspec.license": None,
    "modelspec.tags": None,
    "modelspec.merged_from": None,
    "modelspec.prediction_type": None,
    "modelspec.timestep_range": None,
    "modelspec.encoder_layer": None,
    "ss_base_model_version":"sdxl_base_v1-0",
    "ss_v2":False,
}

# 別に使うやつだけ定義
MODELSPEC_TITLE = "modelspec.title"

ARCH_SD_V1 = "stable-diffusion-v1"
ARCH_SD_V2_512 = "stable-diffusion-v2-512"
ARCH_SD_V2_768_V = "stable-diffusion-v2-768-v"
ARCH_SD_XL_V1_BASE = "stable-diffusion-xl-v1-base"
ARCH_ZIMAGE = "z-image"

ADAPTER_LORA = "sliders"

IMPL_STABILITY_AI = "https://github.com/Stability-AI/generative-models"
IMPL_DIFFUSERS = "diffusers"

PRED_TYPE_EPSILON = "epsilon"
PRED_TYPE_V = "v"


def build_metadata(
    v2: bool,
    v_parameterization: bool,
    sdxl: bool,
    timestamp: float,
    title: Optional[str] = None,
    reso: Optional[Union[int, Tuple[int, int]]] = None,
    author: Optional[str] = None,
    description: Optional[str] = None,
    license: Optional[str] = None,
    tags: Optional[str] = None,
    merged_from: Optional[str] = None,
    timesteps: Optional[Tuple[int, int]] = None,
    clip_skip: Optional[int] = None,
    zimage: Optional[str] = None,
):
    metadata = {}
    metadata.update(BASE_METADATA)

    if zimage is not None:
        metadata["ss_base_model_version"] = f"zimage_{zimage}"
        del metadata["ss_v2"]
        arch = ARCH_ZIMAGE
    elif sdxl:
        metadata["ss_base_model_version"] = "sdxl_base_v1-0"
        del metadata["ss_v2"]
        arch = ARCH_SD_XL_V1_BASE
    elif v2:
        metadata["ss_v2"] = True
        metadata["ss_base_model_version"] = "sd2.1"
        if v_parameterization:
            arch = ARCH_SD_V2_768_V
        else:
            arch = ARCH_SD_V2_512
    else:
        metadata["ss_base_model_version"] = "sd1.5"
        del metadata["ss_v2"]
        arch = ARCH_SD_V1

    arch += f"/{ADAPTER_LORA}"

    metadata["modelspec.architecture"] = arch

    if sdxl:
        # Stable Diffusion ckpt, TI, SDXL LoRA
        impl = IMPL_STABILITY_AI
    else:
        # v1/v2 LoRA or Diffusers
        impl = IMPL_DIFFUSERS
    metadata["modelspec.implementation"] = impl

    if title is None:
        title = "Sliders"
        title += f"@{timestamp}"
    metadata[MODELSPEC_TITLE] = title

    if author is not None:
        metadata["modelspec.author"] = author
    else:
        del metadata["modelspec.author"]

    if description is not None:
        metadata["modelspec.description"] = description
    else:
        del metadata["modelspec.description"]

    if merged_from is not None:
        metadata["modelspec.merged_from"] = merged_from
    else:
        del metadata["modelspec.merged_from"]

    if license is not None:
        metadata["modelspec.license"] = license
    else:
        del metadata["modelspec.license"]

    if tags is not None:
        metadata["modelspec.tags"] = tags
    else:
        del metadata["modelspec.tags"]

    # remove microsecond from time
    int_ts = int(timestamp)

    # time to iso-8601 compliant date
    date = datetime.datetime.fromtimestamp(int_ts).isoformat()
    metadata["modelspec.date"] = date

    if reso is not None:
        # comma separated to tuple
        if isinstance(reso, str):
            reso = tuple(map(int, reso.split(",")))
        if len(reso) == 1:
            reso = (reso[0], reso[0])
    else:
        # resolution is defined in dataset, so use default
        if sdxl or zimage is not None:
            reso = 1024
        elif v2 and v_parameterization:
            reso = 768
        else:
            reso = 512
    if isinstance(reso, int):
        reso = (reso, reso)

    metadata["modelspec.resolution"] = f"{reso[0]}x{reso[1]}"

    if zimage is not None:
        del metadata["modelspec.prediction_type"]  # Z-Image uses flow matching, not epsilon/v prediction
    elif v_parameterization:
        metadata["modelspec.prediction_type"] = PRED_TYPE_V
    else:
        metadata["modelspec.prediction_type"] = PRED_TYPE_EPSILON

    if timesteps is not None:
        if isinstance(timesteps, str) or isinstance(timesteps, int):
            timesteps = (timesteps, timesteps)
        if len(timesteps) == 1:
            timesteps = (timesteps[0], timesteps[0])
        metadata["modelspec.timestep_range"] = f"{timesteps[0]},{timesteps[1]}"
    else:
        del metadata["modelspec.timestep_range"]

    if clip_skip is not None:
        metadata["modelspec.encoder_layer"] = f"{clip_skip}"
    else:
        del metadata["modelspec.encoder_layer"]

    # # assert all values are filled
    # assert all([v is not None for v in metadata.values()]), metadata
    if not all([v is not None for v in metadata.values()]):
        print(f"Internal error: some metadata values are None: {metadata}")

    return metadata
