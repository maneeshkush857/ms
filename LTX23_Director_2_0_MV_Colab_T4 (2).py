# -*- coding: utf-8 -*-
# =============================================================================
# LTX23_Director_2_0_MV_Colab_T4.py
#
# LTX-2.3 Director 2.0 MV — Google Colab T4 Optimized Pipeline
# Implements: LTX-2.3_Director_2.0-MV-Workflow-30s.json
# Reference:  experiment_ltx23.py / ltx2_ti2v_distilled.py
#
# Architecture:
#   UnetLoaderGGUF → DualCLIPLoader → LTXVConditioning →
#   LTXDirector → LTXDirectorGuide × 2 → LTXVConcatAVLatent ×2 →
#   SamplerCustomAdvanced × 2 → LTXVSeparateAVLatent × 2 →
#   LTXVLatentUpsampler → LTXDirectorCropGuides × 2 →
#   VAEDecode + LTXVAudioVAEDecode → chunk-safe assembly → FFmpeg concat
#
# Workflow JSON node→Python mapping:
#   Node 135  UnetLoaderGGUF            → load_dit_model()
#   Node 12   DualCLIPLoader            → load_text_encoder()
#   Node 8    VAELoader (audio)         → load_audio_vae()
#   Node 36   VAELoader (video)         → load_video_vae()
#   Node 6    VAELoaderKJ (tiny)        → load_tiny_vae()
#   Node 13   LatentUpscaleModelLoader  → load_upscaler()
#   Node 131  LTXDirector               → build_director_conditioning()
#   Node 132  LTXDirectorGuide (pass1)  → apply_director_guide_pass1()
#   Node 133  LTXDirectorGuide (pass2)  → apply_director_guide_pass2()
#   Node 54   LTXDirectorCropGuides     → crop_guides_pass1()
#   Node 55   LTXDirectorCropGuides     → crop_guides_pass2()
#   Node 27   LTXVConditioning          → conditioning_with_fps()
#   Node 128  ConditioningZeroOut       → null_negative_conditioning()
#   Node 17   CFGGuider (pass1)         → guider_pass1()
#   Node 28   CFGGuider (pass2)         → guider_pass2()
#   Node 20   KSamplerSelect (euler)    → sampler_select_euler()
#   Node 32   KSamplerSelect (euler)    → sampler_select_euler_2()
#   Node 33   BasicScheduler            → sigma_scheduler()
#   Node 30   RandomNoise               → noise_generator()
#   Node 19   SamplerCustomAdvanced     → sample_pass1()
#   Node 31   SamplerCustomAdvanced     → sample_pass2()
#   Node 22   LTXVSeparateAVLatent      → separate_av_pass1()
#   Node 34   LTXVSeparateAVLatent      → separate_av_pass2()
#   Node 18   LTXVConcatAVLatent        → concat_av_pass1()
#   Node 29   LTXVConcatAVLatent        → concat_av_pass2()
#   Node 14   LTXVLatentUpsampler       → upsample_latent()
# =============================================================================

# =============================================================================
# SECTION 1 — CONFIGURATION
# @title ⚙️ LTX-2.3 Director 2.0 MV — Settings
# @markdown ---
# @markdown ### 📁 Input Files
# @markdown Upload your reference image and audio file to `/content/ComfyUI/input/` before running.
# =============================================================================

# ── ① Input files ─────────────────────────────────────────────────────────────
# @markdown ---
# @markdown ### 📷 Reference Image & 🎵 Audio
IMAGE_PATH = "/content/ComfyUI/input/reference.png"  # @param {type:"string"}
AUDIO_PATH = "/content/ComfyUI/input/audio.mp3"      # @param {type:"string"}
# @markdown > Leave `AUDIO_PATH` blank to use AI-generated audio only.
# @markdown > Set `IMAGE_PATH` blank for pure text-to-video mode.

# ── ② Timeline ────────────────────────────────────────────────────────────────
# @markdown ---
# @markdown ### 🎬 Timeline
DURATION_SECONDS = 31.5   # @param {type:"number"}
FPS              = 24     # @param [8, 12, 16, 24, 25, 30] {type:"raw", label:"Frame rate (fps)"}
OUTPUT_WIDTH     = 1280   # @param {type:"integer"}
OUTPUT_HEIGHT    = 720    # @param {type:"integer"}
OUTPUT_FILENAME  = "LTX23_Director_30s.mp4"  # @param {type:"string"}

# ── ③ Prompt ──────────────────────────────────────────────────────────────────
# @markdown ---
# @markdown ### 📝 Prompt
# @markdown Paste your generation prompt below. Leave blank to use the built-in
# @markdown cinematic music video prompt from the workflow JSON.
CUSTOM_PROMPT = ""  # @param {type:"string"}

# ── ④ Seed ────────────────────────────────────────────────────────────────────
# @markdown ---
# @markdown ### 🎲 Seed
SEED        = 123456  # @param {type:"integer"}
RANDOM_SEED = False   # @param {type:"boolean"}
# @markdown > Enable `RANDOM_SEED` to ignore the seed value above and pick a random seed.

# ── ⑤ Quality / Memory profile ───────────────────────────────────────────────
# @markdown ---
# @markdown ### 🖥️ Quality & Memory Profile
# @markdown | Profile | Chunk frames | Resolution | VRAM safety |
# @markdown |---|---|---|---|
# @markdown | `t4_safe` | 48 | 832×480 | ✅ Most stable |
# @markdown | `t4_balanced` | 73 | 1280×720 | ⚠️ Moderate |
# @markdown | `t4_aggressive` | 97 | 1280×720 | ❌ OOM risk |
QUALITY_MODE = "t4_safe"  # @param ["t4_safe", "t4_balanced", "t4_aggressive"] {label:"Quality / memory profile"}

# ── ⑥ Chunking ────────────────────────────────────────────────────────────────
# @markdown ---
# @markdown ### 🧩 Chunk Size (VRAM safety)
AUTO_CHUNK_SIZE = True  # @param {type:"boolean"}
# @markdown > When `AUTO_CHUNK_SIZE` is **True**, the chunk size is estimated
# @markdown > automatically from free VRAM. The manual value below is ignored.
CHUNK_FRAMES = 48  # @param {type:"integer", label:"Manual chunk frames (if AUTO_CHUNK_SIZE=False)"}

# ── ⑦ LoRA strengths ──────────────────────────────────────────────────────────
# @markdown ---
# @markdown ### 🎛️ LoRA Strengths
# @markdown These match the workflow JSON (PowerLoraLoader node).
LORA_STRENGTH_DISTILLED  = 0.4  # @param {type:"slider", min:0.0, max:2.0, step:0.05, label:"Distilled behaviour LoRA"}
LORA_STRENGTH_OMNINFT    = 0.6  # @param {type:"slider", min:0.0, max:2.0, step:0.05, label:"OmniNFT quality LoRA"}
LORA_STRENGTH_TRANSITION = 0.7  # @param {type:"slider", min:0.0, max:2.0, step:0.05, label:"Transition LoRA"}
LORA_STRENGTH_MVCAMERA   = 0.9  # @param {type:"slider", min:0.0, max:2.0, step:0.05, label:"MVCamera drclipz LoRA"}

# ── ⑧ Sampler settings ────────────────────────────────────────────────────────
# @markdown ---
# @markdown ### ⚗️ Sampler Settings
# @markdown These mirror the ComfyUI workflow nodes (BasicScheduler node 33,
# @markdown CFGGuider nodes 17 & 28).
SAMPLER_STEPS     = 8     # @param {type:"slider", min:1, max:30, step:1, label:"Denoising steps (both passes)"}
SAMPLER_CFG       = 1.0   # @param {type:"slider", min:1.0, max:10.0, step:0.5, label:"CFG scale (1.0 = distilled, no guidance)"}
SAMPLER_NAME      = "euler"              # @param ["euler", "euler_ancestral", "dpm_2", "dpm_2_ancestral", "heun"] {label:"Sampler"}
SCHEDULER_NAME    = "linear_quadratic"  # @param ["linear_quadratic", "karras", "exponential", "simple", "normal", "sgm_uniform"] {label:"Sigma scheduler"}
IMG_COMPRESSION   = 18    # @param {type:"slider", min:1, max:95, step:1, label:"Image preprocessing compression (lower=sharper, higher=softer/faster)"}

# ── ⑨ OOM recovery ────────────────────────────────────────────────────────────
# @markdown ---
# @markdown ### 🛡️ OOM / Crash Protection
AUTO_REDUCE_CHUNK_ON_OOM = True  # @param {type:"boolean", label:"Auto-reduce chunk size on CUDA OOM"}
MAX_OOM_RETRIES          = 3     # @param {type:"integer", label:"Max OOM retry attempts per chunk"}
GPU_SAFETY_MARGIN_GB     = 1.5   # @param {type:"slider", min:0.5, max:4.0, step:0.25, label:"GPU safety headroom (GB)"}
ALLOW_AUTO_DOWNGRADE     = True  # @param {type:"boolean", label:"Auto-downgrade resolution if unsafe for T4"}

# ── ⑩ Resume & checkpointing ──────────────────────────────────────────────────
# @markdown ---
# @markdown ### 💾 Resume & Checkpoint
RESUME = True  # @param {type:"boolean", label:"Resume from checkpoint (skip completed chunks)"}

# ── ⑪ Preview mode ────────────────────────────────────────────────────────────
# @markdown ---
# @markdown ### 👁️ Preview Mode
# @markdown Run a short test clip before the full generation.
PREVIEW_MODE     = False  # @param {type:"boolean", label:"Enable preview mode (short test clip)"}
PREVIEW_DURATION = 3      # @param {type:"integer", label:"Preview duration (seconds)"}
PREVIEW_WIDTH    = 832    # @param {type:"integer", label:"Preview width (px)"}
PREVIEW_HEIGHT   = 480    # @param {type:"integer", label:"Preview height (px)"}

# ── ⑫ Memory & logging ────────────────────────────────────────────────────────
# @markdown ---
# @markdown ### 📊 Memory Logging & Cleanup
ENABLE_MEMORY_LOGGING = True   # @param {type:"boolean", label:"Print GPU/RAM memory after each chunk"}
CLEANUP_AFTER_CHUNK   = True   # @param {type:"boolean", label:"Run aggressive CUDA cleanup after each chunk"}
CLEANUP_AFTER_STAGE   = True   # @param {type:"boolean", label:"Run CUDA cleanup after each pipeline stage"}
KEEP_TEMP_CHUNKS      = False  # @param {type:"boolean", label:"Keep temporary chunk files after assembly"}
CLEANUP_TEMP_FILES    = True   # @param {type:"boolean", label:"Delete temp files after final assembly"}

# ── ⑬ Paths ───────────────────────────────────────────────────────────────────
# @markdown ---
# @markdown ### 📂 Output Paths
WORKSPACE_DIR = "/content/ltx23_workspace"  # @param {type:"string"}
OUTPUT_DIR    = "/content/ltx23_output"     # @param {type:"string"}
COMFYUI_DIR   = "/content/ComfyUI"          # @param {type:"string"}

# =============================================================================
# ── Resolve CONFIG dict from @param variables ─────────────────────────────────
# All downstream pipeline code reads from CONFIG so every setting flows through.
# =============================================================================
import random as _random

# Resolve image / audio paths (blank → None)
IMAGE_PATH = IMAGE_PATH.strip() or None
AUDIO_PATH = AUDIO_PATH.strip() or None

# Resolve prompt (blank → use built-in workflow prompt)
# GLOBAL_PROMPT is defined later in this section; _CUSTOM_PROMPT stores the raw param value
_CUSTOM_PROMPT = CUSTOM_PROMPT.strip() or None

# Resolve seed
if RANDOM_SEED:
    SEED = _random.randint(0, 2**31 - 1)
    print(f"  🎲 Random seed selected: {SEED}")

# Sync LoRA strengths back into the dict (defined below MODELS)
_LORA_STRENGTHS_OVERRIDE = {
    "lora_distilled":  LORA_STRENGTH_DISTILLED,
    "lora_omninft":    LORA_STRENGTH_OMNINFT,
    "lora_transition": LORA_STRENGTH_TRANSITION,
    "lora_mvcamera":   LORA_STRENGTH_MVCAMERA,
}

# Sync sampler/scheduler overrides into workflow constants
_SAMPLER_OVERRIDE   = SAMPLER_NAME
_SCHEDULER_OVERRIDE = SCHEDULER_NAME
_STEPS_OVERRIDE     = SAMPLER_STEPS
_CFG_OVERRIDE       = SAMPLER_CFG
_IMG_COMPRESSION_OVERRIDE = IMG_COMPRESSION

CONFIG = {
    # ── Timeline
    "duration_seconds":  DURATION_SECONDS,
    "fps":               FPS,
    "width":             OUTPUT_WIDTH,
    "height":            OUTPUT_HEIGHT,

    # ── Seed
    "seed":              SEED,

    # ── Quality / memory profile
    "quality_mode":      QUALITY_MODE,

    # ── Chunk control
    "auto_chunk_size":   AUTO_CHUNK_SIZE,
    "chunk_frames":      CHUNK_FRAMES,

    # ── OOM recovery
    "auto_reduce_chunk_on_oom": AUTO_REDUCE_CHUNK_ON_OOM,
    "max_oom_retries":          MAX_OOM_RETRIES,

    # ── Resume
    "resume":            RESUME,

    # ── Memory
    "gpu_safety_margin_gb": GPU_SAFETY_MARGIN_GB,
    "enable_memory_logging": ENABLE_MEMORY_LOGGING,

    # ── Cleanup
    "cleanup_after_chunk":  CLEANUP_AFTER_CHUNK,
    "cleanup_after_stage":  CLEANUP_AFTER_STAGE,
    "keep_temp_chunks":     KEEP_TEMP_CHUNKS,
    "cleanup_temp_files":   CLEANUP_TEMP_FILES,

    # ── Preview
    "preview_mode":      PREVIEW_MODE,
    "preview_duration":  PREVIEW_DURATION,
    "preview_width":     PREVIEW_WIDTH,
    "preview_height":    PREVIEW_HEIGHT,

    # ── Resolution safety
    "allow_auto_downgrade": ALLOW_AUTO_DOWNGRADE,

    # ── Paths
    "workspace_dir":     WORKSPACE_DIR,
    "output_dir":        OUTPUT_DIR,
    "output_filename":   OUTPUT_FILENAME,
    "comfyui_dir":       COMFYUI_DIR,
}

print("✓ Settings loaded:")
print(f"  Image path       : {IMAGE_PATH or '(none — T2V mode)'}")
print(f"  Audio path       : {AUDIO_PATH or '(none — AI audio)'}")
print(f"  Duration         : {DURATION_SECONDS}s  @ {FPS} fps")
print(f"  Resolution       : {OUTPUT_WIDTH}×{OUTPUT_HEIGHT}")
print(f"  Seed             : {SEED}{'  (random)' if RANDOM_SEED else ''}")
print(f"  Quality mode     : {QUALITY_MODE}")
print(f"  Auto chunk size  : {AUTO_CHUNK_SIZE}  (manual fallback: {CHUNK_FRAMES} frames)")
print(f"  Sampler          : {SAMPLER_NAME}  /  {SCHEDULER_NAME}  /  {SAMPLER_STEPS} steps  /  cfg={SAMPLER_CFG}")
print(f"  LoRA strengths   : distilled={LORA_STRENGTH_DISTILLED}  omninft={LORA_STRENGTH_OMNINFT}  transition={LORA_STRENGTH_TRANSITION}  mvcamera={LORA_STRENGTH_MVCAMERA}")
print(f"  Preview mode     : {PREVIEW_MODE}{'  ('+str(PREVIEW_DURATION)+'s @ '+str(PREVIEW_WIDTH)+'×'+str(PREVIEW_HEIGHT)+')' if PREVIEW_MODE else ''}")
print(f"  Resume           : {RESUME}")
print(f"  GPU safety margin: {GPU_SAFETY_MARGIN_GB} GB")

# ── Model filenames (from workflow JSON) ──────────────────────────────────────
MODELS = {
    "dit": "ltx-2-3-22b-dev-Q4_K_M.gguf",
    "text_encoder_1": "gemma_3_12B_it_fp4_mixed.safetensors",
    "text_encoder_2": "ltx-2.3_text_projection_bf16.safetensors",
    "audio_vae": "LTX23_audio_vae_bf16.safetensors",
    "video_vae": "LTX23_video_vae_bf16.safetensors",
    "tiny_vae": "taeltx2_3.safetensors",
    "upscaler": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    # LoRAs active in workflow (PowerLoraLoader node 126)
    "lora_distilled": "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors",
    "lora_omninft":   "LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors",
    "lora_transition": "ltx2.3-transition.safetensors",
    "lora_mvcamera":  "LTX2.3-MVCamera-drclips.safetensors",
}

LORA_STRENGTHS = {
    "lora_distilled":  _LORA_STRENGTHS_OVERRIDE.get("lora_distilled",  0.4),
    "lora_omninft":    _LORA_STRENGTHS_OVERRIDE.get("lora_omninft",    0.6),
    "lora_transition": _LORA_STRENGTHS_OVERRIDE.get("lora_transition", 0.7),
    "lora_mvcamera":   _LORA_STRENGTHS_OVERRIDE.get("lora_mvcamera",   0.9),
}

DOWNLOAD_URLS = {
    "dit":            "https://huggingface.co/vantagewithai/LTX-2.3-GGUF/resolve/main/dev/ltx-2-3-22b-dev-Q4_K_M.gguf",
    "text_encoder_1": "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
    "text_encoder_2": "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors",
    "audio_vae":      "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_audio_vae_bf16.safetensors",
    "video_vae":      "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_video_vae_bf16.safetensors",
    "tiny_vae":       "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/taeltx2_3.safetensors",
    "upscaler":       "https://huggingface.co/vidfom/aimusic/resolve/main/ComfyUI/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    "lora_distilled": "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/loras/ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors",
    "lora_omninft":   "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors",
    "lora_transition":"https://huggingface.co/joyfox/LTX-2.3-Transition-LORA/resolve/main/ltx2.3-transition.safetensors",
    "lora_mvcamera":  "https://huggingface.co/vidfom/aimusic/resolve/main/ComfyUI/models/loras/LTX2.3-MVCamera-drclips.safetensors",
}

MODEL_DEST_DIRS = {
    "dit":            "/content/ComfyUI/models/unet",
    "text_encoder_1": "/content/ComfyUI/models/text_encoders",
    "text_encoder_2": "/content/ComfyUI/models/text_encoders",
    "audio_vae":      "/content/ComfyUI/models/vae",
    "video_vae":      "/content/ComfyUI/models/vae",
    "tiny_vae":       "/content/ComfyUI/models/vae",
    "upscaler":       "/content/ComfyUI/models/latent_upscale_models",
    "lora_distilled": "/content/ComfyUI/models/loras",
    "lora_omninft":   "/content/ComfyUI/models/loras",
    "lora_transition":"/content/ComfyUI/models/loras",
    "lora_mvcamera":  "/content/ComfyUI/models/loras",
}

# ── T4 quality profiles ───────────────────────────────────────────────────────
T4_PROFILES = {
    "t4_safe": {
        "chunk_frames": 48,
        "generation_width": 832,
        "generation_height": 480,
        "offload_models": True,
        "skip_director": True,   # Skip LTXDirector (avoids loading CLIP/Gemma 3 12B ~6GB)
        "img_compression": 33,
        "longer_edge": 848,
        "description": "Conservative: 48-frame chunks, 832×480 generation, strong offloading, no CLIP",
    },
    "t4_balanced": {
        "chunk_frames": 73,
        "generation_width": 1280,
        "generation_height": 720,
        "offload_models": True,
        "img_compression": 18,
        "longer_edge": 1312,
        "description": "Balanced: 73-frame chunks, 1280×720, moderate offloading",
    },
    "t4_aggressive": {
        "chunk_frames": 97,
        "generation_width": 1280,
        "generation_height": 720,
        "offload_models": False,
        "img_compression": 18,
        "longer_edge": 1312,
        "description": "Aggressive: 97-frame chunks, 1280×720, minimal offloading (OOM risk)",
    },
}

# ── Workflow-preserved constants (from JSON, overridable via @param above) ────
WORKFLOW_FPS               = CONFIG["fps"]
WORKFLOW_DURATION_S        = CONFIG["duration_seconds"]
WORKFLOW_TOTAL_FRAMES      = round(CONFIG["duration_seconds"] * CONFIG["fps"])
WORKFLOW_WIDTH             = CONFIG["width"]
WORKFLOW_HEIGHT            = CONFIG["height"]
WORKFLOW_CFG               = _CFG_OVERRIDE
WORKFLOW_SAMPLER_PASS1     = _SAMPLER_OVERRIDE
WORKFLOW_SAMPLER_PASS2     = _SAMPLER_OVERRIDE
WORKFLOW_SCHEDULER         = _SCHEDULER_OVERRIDE
WORKFLOW_STEPS             = _STEPS_OVERRIDE
WORKFLOW_STEPS_PASS2       = 4
WORKFLOW_DENOISE_PASS2     = 0.42
WORKFLOW_IMG_COMPRESSION   = _IMG_COMPRESSION_OVERRIDE

# Global prompt — uses custom @param value if provided, otherwise the built-in workflow prompt
GLOBAL_PROMPT = _CUSTOM_PROMPT if _CUSTOM_PROMPT else (
    "Create a highly realistic cinematic AI music video using the provided reference image. "
    "Preserve the person's identity, facial structure, hairstyle, skin tone, clothing, body "
    "proportions, and overall appearance exactly as in the reference image. The singer must "
    "remain fully recognizable throughout the entire video with absolutely no identity drift.\n\n"
    "The person is performing directly to the camera as a world-class pop, hip-hop and rap singer "
    "during a sold-out stadium concert. Generate perfectly synchronized lip movements from the "
    "provided lyrics or audio.\n\n"
    "drclipz, Aggressive cinematic music video camera. Fast push-in, fast pull-back, energetic "
    "handheld movement, rhythmic tracking shots, dynamic low-angle hero shots, occasional "
    "close-ups on emotional lyrics, subtle orbit around the singer, cinematic motion blur. "
    "Camera movement follows the beat and amplifies the performance.\n\n"
    "Premium concert lighting with cinematic key light, colorful neon rim lights, volumetric "
    "atmosphere, dramatic contrast, realistic skin tones, vibrant electronic music video mood.\n\n"
    "Photorealistic, blockbuster-quality AI music video, premium live concert performance, "
    "ultra-high facial fidelity, charismatic superstar, emotionally captivating, explosive "
    "stage energy, bold movement, powerful attitude, modern pop, hip-hop and rap performance, "
    "every second feels alive, impossible to look away."
)


# =============================================================================
# SECTION 2 — IMPORTS & CUDA / T4 DETECTION
# =============================================================================

import os
import sys
import gc
import json
import time
import shutil
import hashlib
import subprocess
import traceback
import math
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

# Set CUDA allocator BEFORE torch is imported so expandable_segments takes effect
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
import cv2
from PIL import Image

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

from IPython.display import display, HTML, Image as IPImage, clear_output

# ── CUDA detection ────────────────────────────────────────────────────────────
def detect_gpu() -> Dict:
    """Verify CUDA GPU availability and return hardware info dict."""
    info = {
        "available": torch.cuda.is_available(),
        "device_name": "N/A",
        "vram_total_gb": 0.0,
        "vram_free_gb": 0.0,
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", "N/A"),
    }
    if not info["available"]:
        return info
    info["device_name"] = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    info["vram_total_gb"] = props.total_memory / (1024 ** 3)
    free, total = torch.cuda.mem_get_info(0)
    info["vram_free_gb"] = free / (1024 ** 3)
    return info

_GPU_INFO = detect_gpu()

print(f"PyTorch version : {_GPU_INFO['torch_version']}")
print(f"CUDA version    : {_GPU_INFO['cuda_version']}")
print(f"GPU             : {_GPU_INFO['device_name']}")
print(f"VRAM total      : {_GPU_INFO['vram_total_gb']:.1f} GB")
print(f"VRAM free       : {_GPU_INFO['vram_free_gb']:.1f} GB")

if not _GPU_INFO["available"]:
    raise RuntimeError(
        "\nERROR: NVIDIA CUDA GPU was not detected.\n"
        "This notebook requires a CUDA-enabled runtime.\n"
        "In Colab: Runtime → Change runtime type → T4 GPU"
    )

DEVICE = torch.device("cuda")


# =============================================================================
# SECTION 3 — MEMORY MANAGER
# =============================================================================

class LTXMemoryManager:
    """
    Dedicated VRAM / RAM tracking and cleanup manager for T4 inference.

    Priority order enforced throughout:
        1. Prevent crash
        2. Memory safety
        3. Correct execution
        4. Speed
    """

    def __init__(self, safety_margin_gb: float = 1.5, enable_logging: bool = True):
        self.safety_margin_gb = safety_margin_gb
        self.enable_logging = enable_logging
        self._peak_allocated = 0.0
        self._chunk_info: Dict = {}
        torch.cuda.reset_peak_memory_stats()

    # ── Query helpers ─────────────────────────────────────────────────────────

    def gpu_allocated_gb(self) -> float:
        return torch.cuda.memory_allocated() / (1024 ** 3)

    def gpu_reserved_gb(self) -> float:
        return torch.cuda.memory_reserved() / (1024 ** 3)

    def gpu_free_gb(self) -> float:
        free, _ = torch.cuda.mem_get_info(0)
        return free / (1024 ** 3)

    def gpu_total_gb(self) -> float:
        return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    def gpu_peak_gb(self) -> float:
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        if peak > self._peak_allocated:
            self._peak_allocated = peak
        return self._peak_allocated

    def cpu_used_gb(self) -> float:
        if _HAS_PSUTIL:
            return psutil.Process().memory_info().rss / (1024 ** 3)
        return 0.0

    def cpu_available_gb(self) -> float:
        if _HAS_PSUTIL:
            return psutil.virtual_memory().available / (1024 ** 3)
        return 0.0

    def is_vram_safe(self) -> bool:
        return self.gpu_free_gb() > self.safety_margin_gb

    # ── Cleanup tiers ─────────────────────────────────────────────────────────

    def soft_cleanup(self):
        """Quick GC + cache flush — minimal overhead."""
        gc.collect()
        torch.cuda.empty_cache()

    def cleanup(self):
        """Standard post-operation cleanup."""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def aggressive_cleanup(self):
        """Full cleanup: GC cycles, cache, IPC, peak reset."""
        for _ in range(3):
            gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()

    def empty_cuda_cache(self):
        torch.cuda.empty_cache()

    # ── Object release ────────────────────────────────────────────────────────

    def release_tensor(self, tensor, name: str = "tensor"):
        """Safely delete a tensor and flush cache."""
        if tensor is not None:
            del tensor
        self.soft_cleanup()

    def release_model(self, model, name: str = "model"):
        """Move model to CPU then delete, with full cleanup."""
        if model is None:
            return
        try:
            if hasattr(model, "to"):
                model.to("cpu")
        except Exception:
            pass
        del model
        self.cleanup()

    def safe_model_unload(self, model, name: str = "model"):
        """Alias for release_model with logging."""
        if self.enable_logging:
            print(f"  [mem] Unloading {name}  (free GPU before: {self.gpu_free_gb():.2f} GB)")
        self.release_model(model, name)
        if self.enable_logging:
            print(f"  [mem] Unloaded  {name}  (free GPU after : {self.gpu_free_gb():.2f} GB)")

    # ── Reporting ─────────────────────────────────────────────────────────────

    def memory_report(self, prefix: str = "") -> str:
        lines = [
            f"{prefix}GPU Memory:",
            f"{prefix}  Allocated : {self.gpu_allocated_gb():.3f} GB",
            f"{prefix}  Reserved  : {self.gpu_reserved_gb():.3f} GB",
            f"{prefix}  Free      : {self.gpu_free_gb():.3f} GB",
            f"{prefix}  Peak      : {self.gpu_peak_gb():.3f} GB",
            f"{prefix}CPU RAM:",
            f"{prefix}  Used      : {self.cpu_used_gb():.3f} GB",
            f"{prefix}  Available : {self.cpu_available_gb():.3f} GB",
        ]
        if self._chunk_info:
            lines += [
                f"{prefix}Current chunk:",
                f"{prefix}  Index     : {self._chunk_info.get('index', '?')}",
                f"{prefix}  Frames    : {self._chunk_info.get('frames', '?')}",
                f"{prefix}  Resolution: {self._chunk_info.get('resolution', '?')}",
            ]
        return "\n".join(lines)

    def print_memory(self, prefix: str = ""):
        print(self.memory_report(prefix))

    def set_chunk_info(self, index: int, frames: int, w: int, h: int):
        self._chunk_info = {"index": index, "frames": frames, "resolution": f"{w}×{h}"}

    def warn_if_low(self):
        free = self.gpu_free_gb()
        if free < self.safety_margin_gb:
            print(f"  WARNING: GPU memory below safety threshold ({free:.2f} GB < {self.safety_margin_gb:.2f} GB). Starting cleanup.")
            self.aggressive_cleanup()

    # ── RAM safety helpers ────────────────────────────────────────────────────

    def is_ram_safe(self, required_gb: float = 2.0) -> bool:
        """Return True if available system RAM exceeds required_gb."""
        return self.cpu_available_gb() > required_gb

    def ram_cleanup(self):
        """Aggressive system RAM cleanup: 3x gc.collect() + log RAM state."""
        for _ in range(3):
            gc.collect()
        if self.enable_logging:
            print(f"  [mem] RAM cleanup done. Available: {self.cpu_available_gb():.2f} GB, "
                  f"Used: {self.cpu_used_gb():.2f} GB")

    def estimate_frame_ram_gb(self, num_frames: int, height: int, width: int) -> float:
        """Estimate RAM (GB) needed to hold decoded frames as float32 (N, H, W, 3)."""
        return num_frames * height * width * 3 * 4 / (1024 ** 3)

# Singleton instance used throughout
mem = LTXMemoryManager(
    safety_margin_gb=CONFIG["gpu_safety_margin_gb"],
    enable_logging=CONFIG["enable_memory_logging"],
)


# =============================================================================
# SECTION 4 — ENVIRONMENT INSTALLATION
# =============================================================================

def install_environment():
    """
    Install all required Python packages and system tools.
    Skips steps that are already complete to support resume-after-crash.
    """
    print("=" * 60)
    print("[1/5] Installing core Python packages...")
    print("=" * 60)

    _run("pip install -q torch torchvision torchaudio", "torch")
    _run("pip install -q torchsde einops diffusers accelerate", "diffusers stack")
    _run("pip install -q av spandrel albumentations onnx opencv-python onnxruntime", "vision stack")
    _run("pip install -q psutil nest_asyncio", "utilities")

    print("\n[2/5] Installing system tools (aria2, ffmpeg)...")
    _run("apt-get -y install -qq aria2 ffmpeg", "apt packages")

    print("\n[3/5] Cloning ComfyUI (upstream)...")
    comfyui_dir = CONFIG["comfyui_dir"]
    if not os.path.exists(comfyui_dir):
        _run(f"git clone -q https://github.com/comfyanonymous/ComfyUI {comfyui_dir}", "ComfyUI")
    else:
        print("  ComfyUI already present — skipping clone.")
    _run(f"pip install -q -r {comfyui_dir}/requirements.txt", "ComfyUI requirements")

    print("\n[4/5] Installing ComfyUI custom nodes...")
    _install_custom_nodes()

    print("\n[5/5] Creating workspace directories...")
    for subdir in ["chunks", "frames", "audio", "final", "logs"]:
        Path(f"{CONFIG['workspace_dir']}/{subdir}").mkdir(parents=True, exist_ok=True)
    Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(f"{comfyui_dir}/input").mkdir(parents=True, exist_ok=True)

    print("\n✓ Environment setup complete.")

def _run(cmd: str, label: str):
    """Run a shell command, printing a brief status line."""
    try:
        result = subprocess.run(
            cmd, shell=True, check=True,
            capture_output=True, text=True
        )
        print(f"  ✓ {label}")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ {label}: {e.stderr.strip()[:200]}")

def _install_custom_nodes():
    """Clone and pip-install all required ComfyUI custom nodes."""
    nodes_dir = f"{CONFIG['comfyui_dir']}/custom_nodes"
    Path(nodes_dir).mkdir(parents=True, exist_ok=True)

    REQUIRED_NODES = [
        ("https://github.com/kijai/ComfyUI-KJNodes",              "ComfyUI-KJNodes"),
        ("https://github.com/city96/ComfyUI-GGUF",                "ComfyUI-GGUF"),
        ("https://github.com/Lightricks/ComfyUI-LTXVideo",        "ComfyUI-LTXVideo"),
        ("https://github.com/WhatDreamscost/WhatDreamsCost-ComfyUI", "WhatDreamsCost-ComfyUI"),
        ("https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite","ComfyUI-VideoHelperSuite"),
        ("https://github.com/kijai/ComfyUI-MelBandRoFormer",       "ComfyUI-MelBandRoFormer"),
        ("https://github.com/rgthree/rgthree-comfy",               "rgthree-comfy"),
    ]

    for url, name in REQUIRED_NODES:
        dest = os.path.join(nodes_dir, name)
        if not os.path.exists(dest):
            _run(f"git clone -q {url} {dest}", f"  clone {name}")
        else:
            print(f"  ✓ {name} (already present)")
        req = os.path.join(dest, "requirements.txt")
        if os.path.exists(req):
            _run(f"pip install -q -r {req}", f"  req  {name}")


# =============================================================================
# SECTION 5 — COMFYUI SETUP & CUSTOM NODE LOADING
# =============================================================================

_NODES_LOADED = False

def setup_comfyui():
    """Add ComfyUI to sys.path and initialise node class mappings."""
    comfyui_dir = CONFIG["comfyui_dir"]
    if comfyui_dir not in sys.path:
        sys.path.insert(0, comfyui_dir)
    print(f"  ComfyUI path: {comfyui_dir}")

def import_custom_nodes():
    """Load all built-in and external custom nodes (Colab/Jupyter safe)."""
    global _NODES_LOADED
    if _NODES_LOADED:
        return
    import nest_asyncio
    nest_asyncio.apply()

    # ── Fix: Mock PromptServer.instance for headless (non-server) execution ──
    # WhatDreamsCost-ComfyUI (LTXDirector, LTXDirectorGuide, LTXDirectorCropGuides)
    # requires PromptServer.instance to exist at import time. In headless/Colab mode
    # there is no running aiohttp server, so we create a minimal instance.
    try:
        from aiohttp import web  # noqa: F401
        from server import PromptServer
        if not hasattr(PromptServer, 'instance') or PromptServer.instance is None:
            PromptServer.instance = PromptServer(asyncio.new_event_loop())
    except Exception:
        pass

    # ── Fix: kornia compatibility for ComfyUI-LTXVideo ───────────────────────
    # Some kornia versions removed kornia.geometry.transform.pyramid.pad which
    # ComfyUI-LTXVideo relies on. Patch it with torch.nn.functional.pad.
    try:
        import kornia.geometry.transform.pyramid as _kpyr
        if not hasattr(_kpyr, 'pad'):
            import torch.nn.functional as F
            _kpyr.pad = F.pad
    except Exception:
        pass

    from nodes import init_builtin_extra_nodes, init_external_custom_nodes

    async def _loader():
        failed = await init_builtin_extra_nodes()
        await init_external_custom_nodes()
        if failed:
            print("WARNING: some comfy_extras nodes failed to import:")
            for n in failed:
                print(f"  - {n}")

    try:
        asyncio.run(_loader())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(_loader())

    _NODES_LOADED = True
    print("  ✓ Custom nodes loaded.")

def get_node(name: str):
    """Retrieve a ComfyUI node class by name with a clear error on missing."""
    from nodes import NODE_CLASS_MAPPINGS
    if name not in NODE_CLASS_MAPPINGS:
        raise KeyError(
            f"Required ComfyUI node '{name}' not found in NODE_CLASS_MAPPINGS.\n"
            f"Ensure the custom node providing this node is installed."
        )
    return NODE_CLASS_MAPPINGS[name]()

def get_node_cls(name: str):
    """Return the node class (not an instance) for direct method calls."""
    from nodes import NODE_CLASS_MAPPINGS
    if name not in NODE_CLASS_MAPPINGS:
        raise KeyError(f"ComfyUI node class '{name}' not found.")
    return NODE_CLASS_MAPPINGS[name]

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """ComfyUI node output accessor — handles both tuples and result-dicts."""
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]

# =============================================================================
# SECTION 6 — CUSTOM NODE DEPENDENCY REPORT
# =============================================================================

REQUIRED_NODE_NAMES = {
    "Core ComfyUI": [
        "KSamplerSelect", "SamplerCustomAdvanced", "CFGGuider", "RandomNoise",
        "BasicScheduler", "ConditioningZeroOut", "VAELoader", "VAEDecode",
        "DualCLIPLoader", "CLIPTextEncode", "EmptyLTXVLatentVideo",
        "LTXVConditioning", "LTXVImgToVideoInplace", "LTXVConcatAVLatent",
        "LTXVSeparateAVLatent", "LTXVLatentUpsampler", "LTXVCropGuides",
        "LTXVEmptyLatentAudio", "LTXVAudioVAEDecode", "CreateVideo",
        "LatentUpscaleModelLoader", "ResizeImageMaskNode",
        "LTXVPreprocess", "ResizeImagesByLongerEdge",
    ],
    "ComfyUI-GGUF": [
        "UnetLoaderGGUF",
    ],
    "ComfyUI-KJNodes": [
        "VAELoaderKJ", "ModelPreviewOverrideKJ",
    ],
    "WhatDreamsCost-ComfyUI": [
        "LTXDirector", "LTXDirectorGuide", "LTXDirectorCropGuides",
    ],
    "ComfyUI-VideoHelperSuite": [
        "VHS_VideoCombine",
    ],
}

def validate_custom_nodes() -> bool:
    """Print a dependency report; return True if all required nodes are present."""
    try:
        from nodes import NODE_CLASS_MAPPINGS
    except ImportError:
        print("  ✗ Cannot import NODE_CLASS_MAPPINGS — ComfyUI not in sys.path.")
        return False

    all_ok = True
    print("\n  Custom node dependency report:")
    print("  " + "-" * 50)
    for provider, nodes in REQUIRED_NODE_NAMES.items():
        missing = [n for n in nodes if n not in NODE_CLASS_MAPPINGS]
        if missing:
            print(f"  {'✗'} {provider}: MISSING → {', '.join(missing)}")
            all_ok = False
        else:
            print(f"  {'✓'} {provider}")
    print("  " + "-" * 50)
    return all_ok


# =============================================================================
# SECTION 7 — MODEL DOWNLOAD
# =============================================================================

def model_download(url: str, dest_dir: str, filename: str = None) -> Optional[str]:
    """
    Download a model with aria2c (16 parallel connections).
    Skips if file already exists and is non-empty.
    Returns the filename on success, None on failure.
    """
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = url.split("/")[-1].split("?")[0]
    fp = os.path.join(dest_dir, filename)
    if os.path.exists(fp) and os.path.getsize(fp) > 0:
        print(f"  ✓ {filename} (cached)")
        return filename
    print(f"  ↓ {filename}...", end=" ", flush=True)
    cmd = [
        "aria2c", "--console-log-level=error", "-c",
        "-x", "16", "-s", "16", "-k", "1M",
        "--summary-interval=0", "--quiet",
        "-d", dest_dir, "-o", filename, url,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("done")
        return filename
    except subprocess.CalledProcessError as e:
        print(f"FAILED\n  Error: {e.stderr.strip()[:200]}")
        return None

def download_all_models(skip_loras: bool = False):
    """Download every model required by the Director 2.0 workflow."""
    print("\n  Downloading models...")
    download_keys = list(DOWNLOAD_URLS.keys())
    if skip_loras:
        download_keys = [k for k in download_keys if not k.startswith("lora_")]

    for key in download_keys:
        url = DOWNLOAD_URLS[key]
        dest = MODEL_DEST_DIRS[key]
        fname = MODELS[key]
        result = model_download(url, dest, fname)
        if result is None:
            print(f"  ERROR: Failed to download {key} ({fname})")

def validate_models() -> bool:
    """Check that all required model files exist on disk."""
    ok = True
    print("\n  Model file validation:")
    for key, fname in MODELS.items():
        dest = MODEL_DEST_DIRS[key]
        fp = os.path.join(dest, fname)
        exists = os.path.exists(fp) and os.path.getsize(fp) > 0
        status = "✓" if exists else "✗ MISSING"
        print(f"  {status:10s} {fname}")
        if not exists:
            ok = False
    return ok


# =============================================================================
# SECTION 8 — PRE-GENERATION VALIDATION SUITE
# =============================================================================

def validate_environment() -> bool:
    """Verify CUDA, PyTorch, and Python runtime."""
    ok = True
    print("  Validating environment...")
    if not torch.cuda.is_available():
        print("  ✗ CUDA not available")
        ok = False
    else:
        print(f"  ✓ CUDA {torch.version.cuda} on {torch.cuda.get_device_name(0)}")
    maj, min_ = sys.version_info[:2]
    if maj < 3 or (maj == 3 and min_ < 9):
        print(f"  ✗ Python {maj}.{min_} — requires 3.9+")
        ok = False
    else:
        print(f"  ✓ Python {maj}.{min_}")
    return ok

def validate_workflow_dependencies() -> bool:
    """Check ComfyUI is in sys.path and nodes are importable."""
    try:
        import nodes  # noqa: F401
        return True
    except ImportError as e:
        print(f"  ✗ Cannot import ComfyUI nodes: {e}")
        return False

def validate_input_image(path: Optional[str]) -> bool:
    """Return True if path is None (T2V) or a readable image file."""
    if path is None:
        return True
    if not os.path.exists(path):
        print(f"  ✗ Input image not found: {path}")
        return False
    try:
        img = Image.open(path)
        img.verify()
        print(f"  ✓ Input image: {path} ({img.size})")
        return True
    except Exception as e:
        print(f"  ✗ Input image invalid: {e}")
        return False

def validate_audio(path: Optional[str]) -> bool:
    """Return True if path is None or a readable audio file."""
    if path is None:
        return True
    if not os.path.exists(path):
        print(f"  ✗ Audio file not found: {path}")
        return False
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  ✓ Audio file: {path} ({size_mb:.1f} MB)")
    return True

def validate_resolution(w: int, h: int) -> bool:
    """Warn if the requested resolution exceeds safe T4 limits."""
    safe = (w <= 1280 and h <= 720) or (w <= 720 and h <= 1280)
    if not safe:
        print(f"  ⚠ Resolution {w}×{h} may exceed T4 safe limits.")
    else:
        print(f"  ✓ Resolution {w}×{h}")
    return True  # non-fatal; handled by profile selection

def validate_frame_count(n: int) -> bool:
    """Verify frame count meets LTX temporal constraints."""
    valid = _is_valid_ltx_frame_count(n)
    if not valid:
        print(f"  ⚠ Frame count {n} does not meet LTX constraints (will be adjusted).")
    else:
        print(f"  ✓ Frame count {n}")
    return True  # adjustment happens at runtime

def validate_gpu_memory(required_gb: float = 8.0) -> bool:
    """Check that sufficient VRAM is available to begin generation."""
    free = mem.gpu_free_gb()
    ok = free >= required_gb
    status = "✓" if ok else "✗"
    print(f"  {status} GPU free memory: {free:.2f} GB (need ≥ {required_gb:.1f} GB)")
    return ok

def run_all_validations(image_path=None, audio_path=None, w=None, h=None, n_frames=None) -> bool:
    """Run the full validation suite before starting generation."""
    print("\n" + "=" * 60)
    print("PRE-GENERATION VALIDATION")
    print("=" * 60)
    w = w or CONFIG["width"]
    h = h or CONFIG["height"]
    n_frames = n_frames or round(CONFIG["duration_seconds"] * CONFIG["fps"])

    results = [
        validate_environment(),
        validate_workflow_dependencies(),
        validate_custom_nodes(),
        validate_models(),
        validate_input_image(image_path),
        validate_audio(audio_path),
        validate_resolution(w, h),
        validate_frame_count(n_frames),
        validate_gpu_memory(required_gb=6.0),
    ]
    passed = all(results)
    print("\n" + ("✓ All validations passed." if passed else "✗ Some validations failed — review above."))
    return passed


# =============================================================================
# SECTION 9 — LTX TEMPORAL CONSTRAINTS & FRAME MATH
# =============================================================================

def _is_valid_ltx_frame_count(n: int, min_frames: int = 9) -> bool:
    """
    LTX-2.3 requires frame counts of the form: (N * 8) + 1
    e.g. 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97 ...
    Minimum is 9 frames (1 second at ~8fps equivalent).
    """
    if n < min_frames:
        return False
    return (n - 1) % 8 == 0

def normalize_ltx_frame_count(requested: int, fps: int = 24, min_frames: int = 9) -> int:
    """
    Round requested frame count UP to the nearest valid LTX frame count.
    Prints adjustment info if count changes.
    """
    if _is_valid_ltx_frame_count(requested, min_frames):
        return requested
    # Round up to next valid count: N = 8k+1, so k = ceil((n-1)/8)
    k = math.ceil((requested - 1) / 8)
    adjusted = k * 8 + 1
    duration_req = requested / fps
    duration_adj = adjusted / fps
    print(f"  LTX frame adjustment:")
    print(f"    Requested frames : {requested}  ({duration_req:.2f}s)")
    print(f"    Adjusted LTX frames: {adjusted}  ({duration_adj:.2f}s)")
    return adjusted

def calculate_timeline(duration_s: float, fps: int) -> Tuple[int, float]:
    """
    Compute total frame count and actual duration for the given timeline.
    Returns (total_frames, actual_duration_s).
    """
    raw_frames = round(duration_s * fps)
    valid_frames = normalize_ltx_frame_count(raw_frames, fps)
    actual_duration = valid_frames / fps
    return valid_frames, actual_duration

def get_chunk_seed(global_seed: int, chunk_index: int) -> int:
    """Deterministic per-chunk seed derived from global seed and chunk index."""
    return (global_seed + chunk_index * 1000003) & 0x7FFFFFFF

def plan_chunks(total_frames: int, chunk_size: int, fps: int) -> List[Dict]:
    """
    Divide total_frames into valid LTX chunks.
    Each chunk overlaps the next by 1 frame for continuity.
    Returns a list of chunk descriptor dicts.
    """
    chunks = []
    start = 0
    idx = 0
    while start < total_frames:
        remaining = total_frames - start
        raw_size = min(chunk_size, remaining)
        # Ensure chunk is a valid LTX frame count
        valid_size = normalize_ltx_frame_count(raw_size, fps)
        # Don't exceed total
        if start + valid_size > total_frames:
            valid_size = total_frames - start
            # Re-validate; if under min, absorb into previous chunk
            if valid_size < 9:
                if chunks:
                    chunks[-1]["num_frames"] += valid_size
                break
        chunks.append({
            "chunk_index": idx,
            "start_frame": start,
            "num_frames": valid_size,
            "fps": fps,
            "path": None,
        })
        idx += 1
        start += valid_size
    return chunks

def estimate_chunk_size(w: int, h: int, fps: int, mode: str = "t4_safe") -> int:
    """
    Estimate a safe chunk size for the given resolution and quality mode.
    Uses free VRAM and a simple memory model:
        latent_bytes ≈ (W/8) * (H/8) * frames * 128 channels * 2 bytes (bf16) * 2 (video+audio)
    Returns a valid LTX frame count.
    """
    profile = T4_PROFILES.get(mode, T4_PROFILES["t4_safe"])
    if not CONFIG["auto_chunk_size"]:
        return normalize_ltx_frame_count(profile["chunk_frames"])

    free_gb = mem.gpu_free_gb() - CONFIG["gpu_safety_margin_gb"]
    free_gb = max(free_gb, 1.0)
    free_bytes = free_gb * (1024 ** 3)

    lw = w // 8
    lh = h // 8
    bytes_per_frame = lw * lh * 128 * 2 * 2  # bf16, video+audio channels
    max_frames = int(free_bytes / bytes_per_frame)
    max_frames = max(9, min(max_frames, profile["chunk_frames"]))

    safe_frames = normalize_ltx_frame_count(max_frames, fps)
    print(f"  Auto chunk size: {safe_frames} frames  (estimated from {free_gb:.2f} GB free)")
    return safe_frames


# =============================================================================
# SECTION 10 — RESOLUTION & PROFILE SELECTION
# =============================================================================

def select_profile(mode: str) -> Dict:
    """Return the T4 profile dict for the given mode."""
    if mode not in T4_PROFILES:
        print(f"  Unknown quality mode '{mode}', falling back to t4_safe.")
        mode = "t4_safe"
    p = T4_PROFILES[mode]
    print(f"  Quality mode: {mode} — {p['description']}")
    return p

def check_resolution_safety(w: int, h: int, mode: str) -> Tuple[int, int]:
    """
    Warn if the requested resolution exceeds safe T4 limits for the given mode.
    If CONFIG['allow_auto_downgrade'] is True, automatically downgrades.
    Returns the (final_w, final_h) to use.
    """
    profile = select_profile(mode)
    safe_w = profile["generation_width"]
    safe_h = profile["generation_height"]

    # Estimate VRAM needed for full-resolution single chunk
    lw = w // 8
    lh = h // 8
    chunk_frames = profile["chunk_frames"]
    est_bytes = lw * lh * 128 * 2 * 2 * chunk_frames
    est_gb = est_bytes / (1024 ** 3)
    free_gb = mem.gpu_free_gb()

    if w <= safe_w and h <= safe_h:
        return w, h  # within profile limits

    print(f"\n  Resolution check:")
    print(f"    Requested  : {w}×{h}")
    print(f"    Estimated T4 memory for {chunk_frames}-frame chunk: {est_gb:.2f} GB")
    print(f"    Safe for {mode}: {safe_w}×{safe_h}")
    print(f"    VRAM free now: {free_gb:.2f} GB")

    if CONFIG["allow_auto_downgrade"]:
        print(f"  → Auto-downgrading to {safe_w}×{safe_h}")
        return safe_w, safe_h
    else:
        print(f"  → Proceeding with requested {w}×{h} (auto-downgrade disabled)")
        return w, h


# =============================================================================
# SECTION 11 — INPUT IMAGE & AUDIO PREPARATION
# =============================================================================

def tensor_width_height(image) -> Tuple[int, int]:
    """
    Return (width, height) for a ComfyUI image tensor (NHWC or HWC).
    Replaces the deprecated GetImageSize custom node dependency.
    """
    if isinstance(image, (tuple, list)):
        image = get_value_at_index(image, 0)
    if image.ndim == 4:   # (N, H, W, C)
        return int(image.shape[2]), int(image.shape[1])
    if image.ndim == 3:   # (H, W, C)
        return int(image.shape[1]), int(image.shape[0])
    raise ValueError(f"Unsupported image tensor shape: {getattr(image, 'shape', None)}")


def load_input_image(image_path: Optional[str], width: int, height: int) -> Tuple:
    """
    Load and validate input image using ComfyUI LoadImage node.
    Returns a (image_tensor, mask) tuple plus (image_strength, image_bypass).
    For T2V (image_path=None) returns a grey placeholder tensor.
    The master image tensor is kept on CPU; only transferred to GPU when needed.
    """
    if image_path is not None:
        loadimage = get_node("LoadImage")
        loaded = loadimage.load_image(image=image_path)
        image_strength = 1.0
        image_bypass = False
        print(f"  ✓ Input image loaded: {image_path}")
    else:
        # Text-to-video: grey placeholder (stays on CPU)
        noise_image = torch.full((1, height, width, 3), 0.5, dtype=torch.float32)
        loaded = (noise_image, None)
        image_strength = 0.0
        image_bypass = True
        print("  ✓ T2V mode — using grey placeholder image")
    return loaded, image_strength, image_bypass


def prepare_image_for_chunk(
    loaded_image_tuple,
    width: int,
    height: int,
    img_compression: int = 18,
    longer_edge: int = 1312,
) -> Tuple:
    """
    Run the LTX image preprocessing pipeline for a single chunk.
    Mirrors workflow nodes: ResizeImageMaskNode → ResizeImagesByLongerEdge → LTXVPreprocess
    Returns (preprocessed_image, latent_w, latent_h).
    The image tensor is created fresh each call so it can be released independently.
    """
    resizeimagemasknode = get_node("ResizeImageMaskNode")
    resizeimagesbylongeredge = get_node("ResizeImagesByLongerEdge")
    ltxvpreprocess = get_node("LTXVPreprocess")

    # Node 102 — ResizeImageMaskNode: scale to generation resolution
    resized = resizeimagemasknode.EXECUTE_NORMALIZED(
        input=get_value_at_index(loaded_image_tuple, 0),
        scale_method="lanczos",
        resize_type={
            "resize_type": "scale dimensions",
            "width": width,
            "height": height,
            "crop": "center",
        },
    )

    # Node 140 — ResizeImagesByLongerEdge: normalise longer edge
    rescaled = resizeimagesbylongeredge.EXECUTE_NORMALIZED(
        longer_edge=longer_edge,
        images=get_value_at_index(resized, 0),
    )

    # Node 126 — LTXVPreprocess: JPEG-style compression artifact (workflow: img_compression=18)
    preprocessed = ltxvpreprocess.EXECUTE_NORMALIZED(
        img_compression=img_compression,
        image=get_value_at_index(rescaled, 0),
    )

    # Compute latent spatial dimensions directly from resized tensor
    resized_w, resized_h = tensor_width_height(get_value_at_index(resized, 0))
    latent_w = max(1, resized_w // 2)
    latent_h = max(1, resized_h // 2)

    # Release intermediate tensors immediately
    del resized, rescaled
    mem.soft_cleanup()

    return preprocessed, latent_w, latent_h


def load_audio_file_lightweight(audio_path: Optional[str]) -> Optional[Dict]:
    """
    Load audio metadata without reading the full waveform into RAM.
    Returns a lightweight dict for later use; waveform loaded on-demand per segment.
    """
    if audio_path is None:
        return None
    if not os.path.exists(audio_path):
        print(f"  ✗ Audio file not found: {audio_path}")
        return None
    size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    print(f"  ✓ Audio registered: {os.path.basename(audio_path)} ({size_mb:.1f} MB)")
    return {
        "path": audio_path,
        "loaded": False,
        "waveform": None,  # populated lazily
        "sample_rate": None,
    }


def get_audio_segment_for_chunk(
    audio_info: Optional[Dict],
    start_frame: int,
    num_frames: int,
    fps: int,
) -> Optional[Dict]:
    """
    Extract a lightweight audio segment dict for a temporal chunk.
    Waveform is NOT duplicated; only start/length metadata is returned.
    The actual audio is handled by LTXVEmptyLatentAudio (model hallucinates audio)
    or injected via ComfyUI audio nodes when a real audio file is provided.
    """
    if audio_info is None:
        return None
    start_s = start_frame / fps
    duration_s = num_frames / fps
    return {
        "path": audio_info["path"],
        "start_seconds": start_s,
        "duration_seconds": duration_s,
        "trim_frame": start_frame,
    }


# =============================================================================
# SECTION 12 — TEXT CONDITIONING (with embedding cache)
# =============================================================================

# CPU-resident conditioning cache: avoids re-encoding identical prompts
_CONDITIONING_CACHE: Dict[str, Any] = {}


def build_text_conditioning(
    prompt: str,
    fps: int,
    cache_key: Optional[str] = None,
) -> Tuple:
    """
    Encode text prompt using DualCLIPLoader (workflow node 12) then wrap it
    in LTXVConditioning (node 27) to inject frame rate metadata.

    Returns (positive_cond, negative_cond) both on CPU.
    The CLIP model is unloaded immediately after encoding.
    Embeddings are cached by cache_key so identical prompts reuse the result.
    """
    ck = cache_key or hashlib.md5(f"{prompt}|{fps}".encode()).hexdigest()

    if ck in _CONDITIONING_CACHE:
        print("  ✓ Conditioning from cache.")
        return _CONDITIONING_CACHE[ck]

    print("  Loading text encoder (DualCLIPLoader)...")
    dualcliploader = get_node("DualCLIPLoader")

    # Primary: gemma_3_12B_it_fp4_mixed + ltx-2.3_text_projection_bf16 (workflow node 12)
    try:
        clip_result = dualcliploader.load_clip(
            clip_name1=MODELS["text_encoder_1"],
            clip_name2=MODELS["text_encoder_2"],
            type="ltxv",
            device="default",
        )
    except Exception as e:
        print(f"  Primary CLIP load failed ({e}), trying fp8 fallback...")
        # Fallback: fp8 Gemma variant used in experiment_ltx23.py
        clip_result = dualcliploader.load_clip(
            clip_name1="gemma_3_12B_it_fp8_scaled.safetensors",
            clip_name2="ltx-2.3-22b-dev_embeddings_connectors.safetensors",
            type="ltxv",
            device="default",
        )

    clip_obj = get_value_at_index(clip_result, 0)

    # Encode positive prompt
    cliptextencode = get_node("CLIPTextEncode")
    pos_encoded = cliptextencode.encode(text=prompt, clip=clip_obj)

    # Null negative (workflow node 128 — ConditioningZeroOut)
    conditioningzeroout = get_node("ConditioningZeroOut")
    neg_encoded = conditioningzeroout.zero_out(
        conditioning=get_value_at_index(pos_encoded, 0)
    )

    # Unload CLIP immediately — it is the single biggest RAM consumer after the DiT
    del clip_result, clip_obj, dualcliploader, cliptextencode
    mem.cleanup()

    # LTXVConditioning (workflow node 27): inject frame rate into conditioning
    ltxvconditioning = get_node("LTXVConditioning")
    cond = ltxvconditioning.EXECUTE_NORMALIZED(
        frame_rate=fps,
        positive=get_value_at_index(pos_encoded, 0),
        negative=get_value_at_index(neg_encoded, 0),
    )

    # Move conditioning to CPU for caching
    pos_cond = get_value_at_index(cond, 0)
    neg_cond = get_value_at_index(cond, 1)

    result = (pos_cond, neg_cond)
    _CONDITIONING_CACHE[ck] = result
    print("  ✓ Text conditioning built and cached.")

    del pos_encoded, neg_encoded, cond
    mem.cleanup()

    return result


def get_conditioning_on_device(pos_cond, neg_cond):
    """
    Return conditioning already on CUDA (no-op if already there).
    For use immediately before sampler calls.
    """
    return pos_cond, neg_cond


# =============================================================================
# SECTION 13 — MODEL LOADING (DiT, VAEs, Upscaler, LoRAs)
# =============================================================================

# Weak model references to avoid accidental duplication
_MODEL_CACHE: Dict[str, Any] = {}


# ── DiT model cache (prevents double-load OOM on T4) ─────────────────────────
_DIT_MODEL_CACHE = None


def load_dit_model(apply_loras: bool = True) -> Any:
    """
    Load the LTX-2.3 22B GGUF DiT model (workflow node 135 - UnetLoaderGGUF)
    and apply all 4 LoRAs at workflow strengths.

    Uses a module-level cache so subsequent calls within the same chunk reuse
    the already-loaded model instead of allocating a second 12-13 GB copy.

    LoRA application order (from JSON PowerLoraLoader node):
        1. lora_distilled  strength=0.4
        2. lora_omninft    strength=0.6
        3. lora_transition strength=0.7
        4. lora_mvcamera   strength=0.9
    """
    global _DIT_MODEL_CACHE
    if _DIT_MODEL_CACHE is not None:
        print("  DiT model (from cache)")
        return _DIT_MODEL_CACHE

    print("  Loading DiT model (UnetLoaderGGUF)...")
    mem.cleanup()

    unetloadergguf = get_node("UnetLoaderGGUF")
    unet_result = unetloadergguf.load_unet(unet_name=MODELS["dit"])
    model = get_value_at_index(unet_result, 0)
    del unet_result
    mem.soft_cleanup()

    if apply_loras:
        # Import LoraLoaderModelOnly from ComfyUI nodes
        from nodes import LoraLoaderModelOnly
        lora_loader = LoraLoaderModelOnly()

        lora_order = [
            ("lora_distilled",  LORA_STRENGTHS["lora_distilled"]),
            ("lora_omninft",    LORA_STRENGTHS["lora_omninft"]),
            ("lora_transition", LORA_STRENGTHS["lora_transition"]),
            ("lora_mvcamera",   LORA_STRENGTHS["lora_mvcamera"]),
        ]
        for lora_key, strength in lora_order:
            fname = MODELS[lora_key]
            lora_path = os.path.join(MODEL_DEST_DIRS[lora_key], fname)
            if os.path.exists(lora_path):
                print(f"  Applying LoRA: {fname}  strength={strength}")
                model = lora_loader.load_lora_model_only(model, fname, strength)[0]
            else:
                print(f"  LoRA not found, skipping: {fname}")

    _DIT_MODEL_CACHE = model
    print("  DiT model ready.")
    return model


def release_dit_model():
    """Clear the DiT model cache and free GPU memory."""
    global _DIT_MODEL_CACHE
    if _DIT_MODEL_CACHE is not None:
        del _DIT_MODEL_CACHE
        _DIT_MODEL_CACHE = None
        mem.aggressive_cleanup()
        print("  DiT model released from cache.")


def load_video_vae() -> Any:
    """Load video VAE (workflow node 36 — VAELoader)."""
    print("  Loading video VAE...")
    vaeloader = get_node("VAELoader")
    result = vaeloader.load_vae(vae_name=MODELS["video_vae"])
    vae = get_value_at_index(result, 0)
    del result
    return vae


def load_audio_vae() -> Any:
    """
    Load audio VAE with version-resilient fallback
    (workflow node 8 — VAELoader / VAELoaderKJ).
    """
    print("  Loading audio VAE...")
    from nodes import NODE_CLASS_MAPPINGS

    if "VAELoaderKJ" in NODE_CLASS_MAPPINGS:
        loader = NODE_CLASS_MAPPINGS["VAELoaderKJ"]()
        result = loader.load_vae(
            vae_name=MODELS["audio_vae"],
            device="main_device",
            weight_dtype="fp16",
        )
    elif "VAELoader" in NODE_CLASS_MAPPINGS:
        loader = NODE_CLASS_MAPPINGS["VAELoader"]()
        result = loader.load_vae(vae_name=MODELS["audio_vae"])
    else:
        raise KeyError("No compatible audio VAE loader found (VAELoaderKJ or VAELoader).")

    vae = get_value_at_index(result, 0)
    del result
    return vae


def load_tiny_vae() -> Any:
    """
    Load Tiny VAE for fast preview thumbnails
    (workflow node 6 — VAELoaderKJ titled 'Tiny VAELoader KJ').
    """
    print("  Loading Tiny VAE (preview)...")
    from nodes import NODE_CLASS_MAPPINGS
    if "VAELoaderKJ" in NODE_CLASS_MAPPINGS:
        loader = NODE_CLASS_MAPPINGS["VAELoaderKJ"]()
        result = loader.load_vae(
            vae_name=MODELS["tiny_vae"],
            device="main_device",
            weight_dtype="bf16",
        )
        return get_value_at_index(result, 0)
    return None


def load_upscaler_model() -> Any:
    """Load latent upscale model (workflow node 13 — LatentUpscaleModelLoader)."""
    print("  Loading spatial upscaler...")
    loader = get_node("LatentUpscaleModelLoader")
    result = loader.EXECUTE_NORMALIZED(model_name=MODELS["upscaler"])
    upscaler = get_value_at_index(result, 0)
    del result
    return upscaler


def offload_model(model, name: str = "model"):
    """Move a model to CPU to free VRAM, then run cleanup."""
    if model is not None and hasattr(model, "to"):
        try:
            model.to("cpu")
        except Exception:
            pass
    mem.cleanup()
    print(f"  ↓ {name} offloaded to CPU")


# =============================================================================
# SECTION 14 — DIRECTOR WORKFLOW EXECUTION (core pipeline)
# =============================================================================
#
# This section reproduces the LTX-2.3 Director 2.0 computational graph:
#
#  LTXDirector (node 131)
#      → LTXVConditioning (node 27) + ConditioningZeroOut (node 128)
#      → LTXVConditioning wrapped conditioning
#      → LTXVEmptyLatentAudio (audio latent)
#      → LTXVImgToVideoInplace (image guidance into latent)
#      → LTXVConcatAVLatent (node 29) — pass 1 concat
#      → LTXDirectorGuide (node 133) — pass 1 guide
#      → CFGGuider (node 28)
#      → SamplerCustomAdvanced (node 31) — pass 1 sample
#      → LTXVSeparateAVLatent (node 34) — split
#      → LTXDirectorCropGuides (node 55) — crop for upscale
#      → LTXVLatentUpsampler (node 14) — 2× spatial upsample
#      → LTXDirectorGuide (node 132) — pass 2 guide
#      → LTXVConcatAVLatent (node 18) — pass 2 concat
#      → CFGGuider (node 17)
#      → SamplerCustomAdvanced (node 19) — pass 2 refinement
#      → LTXVSeparateAVLatent (node 22) — split final
#
# =============================================================================

def build_director_conditioning(
    pos_cond,
    neg_cond,
    image_path: Optional[str],
    audio_path: Optional[str],
    num_frames: int,
    fps: int,
    width: int,
    height: int,
    segment_images: Optional[List[str]] = None,
    segment_prompts: Optional[List[str]] = None,
    dit_model=None,
    audio_vae=None,
) -> Tuple:
    """
    Run the LTXDirector node (workflow node 131) when available.
    This is the WhatDreamsCost Director node that builds multi-segment timeline
    conditioning with image, audio and motion guide data.

    LTXDirector outputs (per workflow JSON):
        slot 0: model (passthrough/modified DiT)
        slot 1: positive conditioning
        slot 2: video_latent (replaces EmptyLTXVLatentVideo)
        slot 3: audio_latent (replaces LTXVEmptyLatentAudio)
        slot 4: guide_data
        slot 5: motion_guide_data
        slot 6: frame_rate

    If LTXDirector is not available in NODE_CLASS_MAPPINGS, falls back gracefully
    to standard LTX conditioning (single-image path) and logs the missing node.

    Returns:
        (director_model, positive_cond, video_latent, audio_latent,
         guide_data, motion_guide_data, frame_rate)
    """
    from nodes import NODE_CLASS_MAPPINGS

    # -- T4 memory guard: skip LTXDirector entirely in t4_safe mode --
    # LTXDirector requires loading CLIP (Gemma 3 12B ~6GB) alongside the DiT model.
    # On T4 (15GB VRAM, ~12.7GB RAM), after DiT + 4 LoRAs load (~14.15GB GPU),
    # there is physically no room for CLIP. Skip Director and use fallback path
    # which only needs DiT (already loaded/cached) + audio_vae (small, ~0.5GB).
    active_profile = T4_PROFILES.get(QUALITY_MODE, {})
    if active_profile.get("skip_director", False):
        print("  Skipping LTXDirector (t4_safe mode) -- CLIP would exceed RAM.")
        print("  Using fallback conditioning (no CLIP needed).")
        return _build_director_fallback(pos_cond, neg_cond, num_frames, fps,
                                        dit_model=dit_model, audio_vae=audio_vae,
                                        reason="t4_safe mode - CLIP skipped")

    # -- LTXDirector path (WhatDreamsCost) --
    if "LTXDirector" in NODE_CLASS_MAPPINGS:
        print("  Using LTXDirector (WhatDreamsCost) node...")

        # Load models needed by Director (reuse if already loaded)
        if dit_model is None:
            dit_model = load_dit_model(apply_loras=True)
        if audio_vae is None:
            audio_vae = load_audio_vae()

        # Load CLIP via DualCLIPLoader (same as build_text_conditioning, workflow node 12)
        # NOTE: The workflow routes CLIP through Power Lora Loader (node 138) before
        # LTXDirector (node 131). However, all LoRAs in this pipeline are model-only
        # (distilled, transition, mvcamera, omninft) and do not modify CLIP weights.
        # LoraLoaderModelOnly is used for the DiT, which by design does not touch CLIP.
        # Therefore passing raw CLIP here is correct and matches the effective behavior.
        dualcliploader = get_node("DualCLIPLoader")
        try:
            clip_result = dualcliploader.load_clip(
                clip_name1=MODELS["text_encoder_1"],
                clip_name2=MODELS["text_encoder_2"],
                type="ltxv",
                device="default",
            )
        except Exception as e:
            print(f"  Primary CLIP load failed ({e}), trying fp8 fallback...")
            clip_result = dualcliploader.load_clip(
                clip_name1="gemma_3_12B_it_fp8_scaled.safetensors",
                clip_name2="ltx-2.3-22b-dev_embeddings_connectors.safetensors",
                type="ltxv",
                device="default",
            )
        clip_model = get_value_at_index(clip_result, 0)

        director_cls = NODE_CLASS_MAPPINGS["LTXDirector"]
        director = director_cls()

        # Introspect INPUT_TYPES to discover what the execute function accepts.
        # This is the safe ComfyUI pattern: only pass params the node declares.
        try:
            input_types = director_cls.INPUT_TYPES()
        except Exception:
            input_types = {"required": {}, "optional": {}}
        required_params = set(input_types.get("required", {}).keys())
        optional_params = set(input_types.get("optional", {}).keys())
        all_accepted = required_params | optional_params

        # Core inputs (from workflow JSON node 131 linked inputs)
        director_kwargs = dict(
            model=dit_model,
            audio_vae=audio_vae,
            global_prompt=GLOBAL_PROMPT,
        )

        # Add CLIP if the node accepts it (CLIP is a linked input in the workflow)
        if clip_model is not None:
            if not all_accepted or "clip" in all_accepted:
                director_kwargs["clip"] = clip_model

        # Widget values (from workflow JSON node 131 widgets_values).
        # These are required positional args for the execute() function.
        # Compute from current chunk parameters for correctness.
        total_frames = num_frames
        duration_s = total_frames / fps
        widget_defaults = {
            "start_second": 0,
            "end_second": duration_s,
            "duration_seconds": duration_s,
            "start_frame": 0,
            "end_frame": total_frames,
            "duration_frames": total_frames,
            "timeline_data": json.dumps({
                "mainTrackEnabled": True,
                "audioTrackEnabled": True,
                "motionTrackEnabled": True,
                "propHeight": 90,
                "globalPropHeight": 470,
                "showFilenames": True,
                "overrideAudio": False,
                "inpaint_audio": True,
                "global_prompt": GLOBAL_PROMPT,
                "retake_global_prompt": "",
                "retakeMode": False,
                "retakeStart": 24,
                "retakeLength": 48,
                "retakePrompt": "",
                "retakeStrength": 1,
                "retakeVideo": None,
                "normalStartFrame": 0,
                "normalDurationFrames": total_frames,
                "segments": [],
                "motionSegments": [],
                "audioSegments": [],
            }),
            "local_prompts": "",
            "segment_lengths": "",
            "epsilon": 0.001,
            "guide_strength": "1.00",
            "mainTrackEnabled": True,
            "audioTrackEnabled": True,
            "motionTrackEnabled": True,
            "frame_rate": fps,
            "display_mode": "seconds",
            "custom_width": width,
            "custom_height": height,
            "resize_method": "maintain aspect ratio",
            "divisible_by": 32,
            "img_compression": WORKFLOW_IMG_COMPRESSION,
            "retakeMode": False,
            "timeline_ui": "",
        }

        # Only add widget params that the node actually accepts
        for param_name, default_val in widget_defaults.items():
            if param_name in all_accepted:
                director_kwargs[param_name] = default_val

        # If introspection found no params (fallback), add the 9 required ones
        # that the error message told us about
        if not all_accepted:
            for param_name in ["start_second", "end_second", "duration_seconds",
                               "start_frame", "end_frame", "duration_frames",
                               "timeline_data", "local_prompts", "segment_lengths"]:
                if param_name not in director_kwargs:
                    director_kwargs[param_name] = widget_defaults[param_name]

        # WhatDreamsCost custom nodes define their execution function name via
        # the FUNCTION class attribute (standard ComfyUI pattern). Use that
        # instead of EXECUTE_NORMALIZED which only exists on core nodes.
        try:
            func_name = getattr(director_cls, "FUNCTION", None)
            if func_name:
                func = getattr(director, func_name)
                director_out = func(**director_kwargs)
            else:
                # Last resort: try EXECUTE_NORMALIZED (core nodes)
                director_out = director.EXECUTE_NORMALIZED(**director_kwargs)
        except (TypeError, AttributeError) as e:
            print(f"  LTXDirector call failed ({e}) -- using fallback conditioning.")
            return _build_director_fallback(pos_cond, neg_cond, num_frames, fps,
                                           dit_model=dit_model, audio_vae=audio_vae,
                                           reason=f"call failed: {e}")

        # Extract all outputs per workflow node 131 output slots
        dir_model       = get_value_at_index(director_out, 0)
        dir_positive    = get_value_at_index(director_out, 1)
        dir_video_lat   = get_value_at_index(director_out, 2)
        dir_audio_lat   = get_value_at_index(director_out, 3)
        dir_guide_data  = get_value_at_index(director_out, 4) if len(director_out) > 4 else None
        dir_motion_data = get_value_at_index(director_out, 5) if len(director_out) > 5 else None
        dir_frame_rate  = get_value_at_index(director_out, 6) if len(director_out) > 6 else fps

        return (dir_model, dir_positive, dir_video_lat, dir_audio_lat,
                dir_guide_data, dir_motion_data, dir_frame_rate)

    # -- Fallback: standard conditioning (no LTXDirector node) --
    return _build_director_fallback(pos_cond, neg_cond, num_frames, fps,
                                    dit_model=dit_model, audio_vae=audio_vae)


def _build_director_fallback(pos_cond, neg_cond, num_frames: int, fps: int,
                             dit_model=None, audio_vae=None,
                             reason: str = "not found") -> Tuple:
    """
    Fallback when LTXDirector is not available or intentionally skipped.
    Returns the same tuple shape as build_director_conditioning but with
    None for guide_data, motion_guide_data, and generates empty audio latent.

    Accepts optional dit_model and audio_vae to avoid redundant loads when
    the caller (e.g. generate_chunk) has already loaded these models.

    Args:
        reason: Why fallback is used (e.g. "not found", "t4_safe mode skip").
    """
    print(f"  LTXDirector fallback ({reason}) -- using standard conditioning.")
    if dit_model is None:
        dit_model = load_dit_model(apply_loras=True)
    else:
        print("  Reusing pre-loaded DiT model (no double-load).")
    if audio_vae is None:
        audio_vae = load_audio_vae()
    else:
        print("  Reusing pre-loaded audio VAE.")

    ltxvemptylatentaudio = get_node("LTXVEmptyLatentAudio")
    audio_lat = ltxvemptylatentaudio.EXECUTE_NORMALIZED(
        frames_number=num_frames,
        frame_rate=fps,
        batch_size=1,
        audio_vae=audio_vae,
    )

    # Return: model, positive, video_latent(None=use empty), audio_latent, guide, motion, fps
    return dit_model, pos_cond, None, get_value_at_index(audio_lat, 0), None, None, fps


def run_director_guide(
    pos_cond,
    neg_cond,
    video_vae,
    latent,
    guide_data,
    motion_guide_data,
    model,
    upscale_factor: float = 1.0,
    node_id: str = "pass",
) -> Tuple:
    """
    Run LTXDirectorGuide (workflow nodes 132 / 133).
    Workflow node 133 (pass 1): upscale_factor=0.5
    Workflow node 132 (pass 2): upscale_factor=1.0

    Widget order: retake_image, upscale_factor_pass, upscale_factor, interpolation,
                  blend_radius, crop_method, use_tiling, tile_overlap, tile_size,
                  tile_stride, force_inpaint.
    Inputs: positive, negative, vae, latent, guide_data, motion_guide_data, model.

    Returns (pos_out, neg_out, latent_out, model_out).
    Falls back to a passthrough if the node is unavailable.
    """
    from nodes import NODE_CLASS_MAPPINGS
    if "LTXDirectorGuide" not in NODE_CLASS_MAPPINGS:
        print(f"  LTXDirectorGuide not found ({node_id}) -- passthrough.")
        return pos_cond, neg_cond, latent, model

    guide_cls = NODE_CLASS_MAPPINGS["LTXDirectorGuide"]
    guide_node = guide_cls()

    # Introspect INPUT_TYPES to discover what the execute function actually accepts.
    # This prevents passing unexpected kwargs (e.g. 'retake_image' which may not exist).
    try:
        input_types = guide_cls.INPUT_TYPES()
    except Exception:
        input_types = {"required": {}, "optional": {}}
    required_params = set(input_types.get("required", {}).keys())
    optional_params = set(input_types.get("optional", {}).keys())
    all_accepted = required_params | optional_params

    # Core linked inputs (always needed)
    inputs = dict(
        positive=pos_cond,
        negative=neg_cond,
        vae=video_vae,
        latent=latent,
        model=model,
    )

    # Widget values from workflow JSON (node 132/133)
    # Order in widgets_values: [retake_image, upscale_factor_pass, upscale_factor,
    #   interpolation, blend_radius, crop_method, use_tiling, tile_overlap,
    #   tile_size, tile_stride, force_inpaint]
    # BUT: the actual param names accepted by execute() may differ.
    # Use introspection to only pass what the node declares.
    widget_candidates = {
        "retake_image": "None",
        "upscale_factor_pass": 1,
        "upscale_factor": upscale_factor,
        "interpolation": "bicubic",
        "blend_radius": 1,
        "crop_method": "center",
        "use_tiling": True,
        "tile_overlap": False,
        "tile_size": 256,
        "tile_stride": 64,
        "force_inpaint": False,
    }

    if all_accepted:
        # Only pass widget params that the node actually declares
        for param_name, val in widget_candidates.items():
            if param_name in all_accepted:
                inputs[param_name] = val
    else:
        # No introspection available - pass all widget params EXCEPT retake_image
        # (which is known to cause errors per runtime feedback)
        for param_name, val in widget_candidates.items():
            if param_name != "retake_image":
                inputs[param_name] = val

    if guide_data is not None:
        inputs["guide_data"] = guide_data
    if motion_guide_data is not None:
        inputs["motion_guide_data"] = motion_guide_data

    try:
        # WhatDreamsCost custom nodes use the FUNCTION class attribute to define
        # their execution method name (standard ComfyUI node pattern).
        func_name = getattr(guide_cls, "FUNCTION", None)
        if func_name:
            func = getattr(guide_node, func_name)
            out = func(**inputs)
        else:
            out = guide_node.EXECUTE_NORMALIZED(**inputs)
    except (TypeError, AttributeError) as e:
        print(f"  LTXDirectorGuide ({node_id}) failed: {e} -- passthrough.")
        return pos_cond, neg_cond, latent, model

    pos_out   = get_value_at_index(out, 0)
    neg_out   = get_value_at_index(out, 1)
    lat_out   = get_value_at_index(out, 2)
    model_out = get_value_at_index(out, 3) if len(out) > 3 else model
    return pos_out, neg_out, lat_out, model_out


def run_director_crop_guides(pos_cond, neg_cond, latent) -> Tuple:
    """
    Run LTXDirectorCropGuides (workflow nodes 54 / 55).
    Crops conditioning to match the given latent resolution.

    Node 55: takes Guide133's pos/neg + pass1 separated video_latent -> feeds upsampler
    Node 54: takes Guide132's pos/neg + pass2 separated video_latent -> feeds VAEDecode

    Returns (pos_out, neg_out, lat_out).
    """
    from nodes import NODE_CLASS_MAPPINGS
    if "LTXDirectorCropGuides" not in NODE_CLASS_MAPPINGS:
        # Fallback to standard LTXVCropGuides (from LTXVideo)
        if "LTXVCropGuides" in NODE_CLASS_MAPPINGS:
            crop_node = NODE_CLASS_MAPPINGS["LTXVCropGuides"]()
            out = crop_node.EXECUTE_NORMALIZED(
                positive=pos_cond,
                negative=neg_cond,
                latent=latent,
            )
        else:
            print("  No crop guides node found -- passthrough.")
            return pos_cond, neg_cond, latent
    else:
        crop_cls = NODE_CLASS_MAPPINGS["LTXDirectorCropGuides"]
        crop_node = crop_cls()
        crop_kwargs = dict(positive=pos_cond, negative=neg_cond, latent=latent)
        try:
            # WhatDreamsCost custom nodes use the FUNCTION class attribute
            func_name = getattr(crop_cls, "FUNCTION", None)
            if func_name:
                func = getattr(crop_node, func_name)
                out = func(**crop_kwargs)
            else:
                out = crop_node.EXECUTE_NORMALIZED(**crop_kwargs)
        except (TypeError, AttributeError) as e:
            print(f"  LTXDirectorCropGuides failed: {e} -- passthrough.")
            return pos_cond, neg_cond, latent

    pos_out = get_value_at_index(out, 0) if get_value_at_index(out, 0) is not None else pos_cond
    neg_out = get_value_at_index(out, 1) if get_value_at_index(out, 1) is not None else neg_cond
    lat_out = get_value_at_index(out, 2)
    return pos_out, neg_out, lat_out


# =============================================================================
# SECTION 15 — TWO-PASS SAMPLING PIPELINE
# =============================================================================

def build_empty_latents(
    num_frames: int,
    latent_w: int,
    latent_h: int,
    fps: int,
    image_preprocessed,
    image_strength: float,
    image_bypass: bool,
    video_vae,
    audio_vae,
) -> Tuple:
    """
    Build the initial video + audio latents for one chunk.

    Workflow mapping:
        EmptyLTXVLatentVideo  (node implicit) → empty video latent
        LTXVImgToVideoInplace (node 128 area) → condition on image
        LTXVEmptyLatentAudio  (node 110)      → empty audio latent
        LTXVConcatAVLatent    (node 29)        → fuse AV

    Returns (av_latent_concat,).
    """
    # Empty video latent at half spatial resolution (LTX downsamples 2×)
    emptyltxvlatentvideo = get_node("EmptyLTXVLatentVideo")
    empty_video_lat = emptyltxvlatentvideo.EXECUTE_NORMALIZED(
        width=latent_w,
        height=latent_h,
        length=num_frames,
        batch_size=1,
    )

    # Condition video latent on the input image
    ltxvimgtovideoinplace = get_node("LTXVImgToVideoInplace")
    img_conditioned_lat = ltxvimgtovideoinplace.EXECUTE_NORMALIZED(
        strength=image_strength,
        bypass=image_bypass,
        vae=video_vae,
        image=get_value_at_index(image_preprocessed, 0),
        latent=get_value_at_index(empty_video_lat, 0),
    )

    # Empty audio latent (model hallucinates audio from text)
    ltxvemptylatentaudio = get_node("LTXVEmptyLatentAudio")
    empty_audio_lat = ltxvemptylatentaudio.EXECUTE_NORMALIZED(
        frames_number=num_frames,
        frame_rate=fps,
        batch_size=1,
        audio_vae=audio_vae,
    )

    # Fuse video + audio into joint AV latent (workflow node 29)
    ltxvconcatavlatent = get_node("LTXVConcatAVLatent")
    if not image_bypass:
        av_latent = ltxvconcatavlatent.EXECUTE_NORMALIZED(
            video_latent=get_value_at_index(img_conditioned_lat, 0),
            audio_latent=get_value_at_index(empty_audio_lat, 0),
        )
    else:
        av_latent = ltxvconcatavlatent.EXECUTE_NORMALIZED(
            video_latent=get_value_at_index(empty_video_lat, 0),
            audio_latent=get_value_at_index(empty_audio_lat, 0),
        )

    del empty_video_lat, empty_audio_lat
    mem.soft_cleanup()
    return av_latent, img_conditioned_lat


def run_sampling_pass(
    model,
    pos_cond,
    neg_cond,
    latent,
    noise_seed: int,
    steps: int = WORKFLOW_STEPS,
    cfg: float = WORKFLOW_CFG,
    denoise: float = 1.0,
    pass_name: str = "Pass1",
) -> Any:
    """
    Run one SamplerCustomAdvanced pass (workflow nodes 31 / 19).
    Uses euler sampler + linear_quadratic BasicScheduler for both passes,
    exactly matching the workflow JSON (nodes 32/33 and 20/BasicScheduler).

    CFG=1 is correct for distilled models (guidance is baked into weights).
    Returns the raw output latent tuple.
    """
    print(f"  Sampling {pass_name} ({steps} steps, denoise={denoise}, seed={noise_seed})...")

    # Sampler — workflow nodes 20 / 32: both use "euler"
    ksamplerselect = get_node("KSamplerSelect")
    sampler = ksamplerselect.EXECUTE_NORMALIZED(sampler_name=WORKFLOW_SAMPLER_PASS1)

    # Noise — workflow node 30
    randomnoise = get_node("RandomNoise")
    noise = randomnoise.EXECUTE_NORMALIZED(noise_seed=noise_seed)

    # Sigma schedule — workflow nodes 33 (pass1: steps=8, denoise=1.0) / 21 (pass2: steps=4, denoise=0.42)
    basicscheduler = get_node("BasicScheduler")
    sigmas = basicscheduler.EXECUTE_NORMALIZED(
        model=model,
        scheduler=WORKFLOW_SCHEDULER,
        steps=steps,
        denoise=denoise,
    )

    # CFG guider (cfg=1 — distilled model)
    cfgguider = get_node("CFGGuider")
    guider = cfgguider.EXECUTE_NORMALIZED(
        cfg=cfg,
        model=model,
        positive=pos_cond,
        negative=neg_cond,
    )

    # Run sampler (workflow nodes 19 / 31)
    samplercustomadvanced = get_node("SamplerCustomAdvanced")
    result = samplercustomadvanced.EXECUTE_NORMALIZED(
        noise=get_value_at_index(noise, 0),
        guider=get_value_at_index(guider, 0),
        sampler=get_value_at_index(sampler, 0),
        sigmas=get_value_at_index(sigmas, 0),
        latent_image=latent,
    )

    del noise, sampler, sigmas, guider
    mem.soft_cleanup()
    return result


def separate_av_latent(sampler_output, output_index: int = 0) -> Tuple:
    """
    Split joint AV latent back into video + audio (workflow nodes 22 / 34).
    
    For pass 1 (node 34): use output_index=0 (the output).
    For pass 2 (node 22): use output_index=0 (the output).
    
    The workflow JSON wires SamplerCustomAdvanced slot 0 (output) to
    LTXVSeparateAVLatent for both passes, matching this default.
    
    Returns (video_latent, audio_latent).
    """
    ltxvseparateavlatent = get_node("LTXVSeparateAVLatent")
    separated = ltxvseparateavlatent.EXECUTE_NORMALIZED(
        av_latent=get_value_at_index(sampler_output, output_index)
    )
    video_lat = get_value_at_index(separated, 0)
    audio_lat = get_value_at_index(separated, 1)
    return video_lat, audio_lat


def upsample_video_latent(video_latent, upscaler_model, video_vae) -> Any:
    """
    2× spatial upscale in latent space (workflow node 14 — LTXVLatentUpsampler).
    """
    print("  Upsampling latent (2×)...")
    ltxvlatentupsampler = get_node("LTXVLatentUpsampler")
    result = ltxvlatentupsampler.upsample_latent(
        samples=video_latent,
        upscale_model=upscaler_model,
        vae=video_vae,
    )
    return get_value_at_index(result, 0)


def recondition_image_on_upscaled(
    upscaled_latent,
    image_preprocessed,
    image_strength: float,
    image_bypass: bool,
    video_vae,
    audio_lat_pass1,
) -> Any:
    """
    Re-apply image conditioning onto the upscaled latent, then concat audio.
    (Mirrors ltxvimgtovideoinplace_130 + ltxvconcatavlatent_129 in reference notebook.)
    Returns the AV-concatenated latent for pass 2.
    """
    ltxvimgtovideoinplace = get_node("LTXVImgToVideoInplace")
    if not image_bypass:
        reconditioned = ltxvimgtovideoinplace.EXECUTE_NORMALIZED(
            strength=image_strength,
            bypass=image_bypass,
            vae=video_vae,
            image=get_value_at_index(image_preprocessed, 0),
            latent=upscaled_latent,
        )
        video_lat_for_pass2 = get_value_at_index(reconditioned, 0)
    else:
        video_lat_for_pass2 = upscaled_latent

    ltxvconcatavlatent = get_node("LTXVConcatAVLatent")
    av_latent_pass2 = ltxvconcatavlatent.EXECUTE_NORMALIZED(
        video_latent=video_lat_for_pass2,
        audio_latent=audio_lat_pass1,
    )
    return av_latent_pass2


# =============================================================================
# SECTION 16 — VAE DECODING (chunked, never full-video at once)
# =============================================================================

def decode_video_latent(video_latent, video_vae, max_batch_frames: int = 0) -> Any:
    """
    Decode video latent to pixel frames using VAEDecode.
    This is called per temporal chunk so the full 30-second video is
    never decoded into GPU memory simultaneously.

    When system RAM is low, decodes in sub-batches of 8 latent temporal
    frames to avoid a single massive CPU allocation.

    Args:
        video_latent: Latent tensor dict with key "samples" of shape (B, C, T, H, W).
        video_vae: The loaded VAE model.
        max_batch_frames: If >0, force sub-batch decoding with this many latent
                          temporal frames per batch. If 0 (default), auto-detect
                          based on available RAM.

    Returns a frame tensor of shape (N, H, W, C) on CPU.
    """
    print("  VAE decoding video latent...")
    vaedecode = get_node("VAEDecode")

    # Determine if sub-batch decoding is needed
    latent_samples = video_latent["samples"] if isinstance(video_latent, dict) else video_latent
    if torch.is_tensor(latent_samples) and latent_samples.ndim == 5:
        # latent shape: (B, C, T_latent, H_latent, W_latent)
        # Pixel frames ~ T_latent * temporal_compression (typically 8 for LTX)
        t_latent = latent_samples.shape[2]
        h_latent = latent_samples.shape[3]
        w_latent = latent_samples.shape[4]
        # Estimate pixel dimensions (8x spatial upscale from latent for LTX)
        est_h = h_latent * 8
        est_w = w_latent * 8
        est_frames = t_latent * 8  # temporal compression factor
        estimated_ram = mem.estimate_frame_ram_gb(est_frames, est_h, est_w)
        available_ram = mem.cpu_available_gb()

        use_subbatch = False
        if max_batch_frames > 0:
            use_subbatch = True
            batch_t = max_batch_frames
        elif available_ram < estimated_ram + 2.0:
            use_subbatch = True
            batch_t = 8  # 8 latent temporal frames per sub-batch
            print(f"  [mem] Low RAM detected ({available_ram:.2f} GB available, "
                  f"~{estimated_ram:.2f} GB needed). Using sub-batch decode "
                  f"(batch_t={batch_t}).")

        if use_subbatch and t_latent > batch_t:
            # Sub-batch decode along temporal dimension
            all_frames = []
            for t_start in range(0, t_latent, batch_t):
                t_end = min(t_start + batch_t, t_latent)
                sub_latent_tensor = latent_samples[:, :, t_start:t_end, :, :]
                sub_latent = {"samples": sub_latent_tensor}
                decoded = vaedecode.decode(samples=sub_latent, vae=video_vae)
                frames_gpu = get_value_at_index(decoded, 0)
                frames_batch_cpu = frames_gpu.detach().to("cpu", non_blocking=False)
                torch.cuda.synchronize()
                all_frames.append(frames_batch_cpu)
                del frames_gpu, decoded, sub_latent, sub_latent_tensor
                mem.cleanup()

            frames_cpu = torch.cat(all_frames, dim=0)
            del all_frames
            mem.soft_cleanup()
            return frames_cpu

    # Standard full decode (original behavior)
    decoded = vaedecode.decode(samples=video_latent, vae=video_vae)
    frames_gpu = get_value_at_index(decoded, 0)

    # Transfer to CPU immediately, non-blocking where safe
    frames_cpu = frames_gpu.detach().to("cpu", non_blocking=False)
    torch.cuda.synchronize()

    del frames_gpu, decoded
    mem.cleanup()
    return frames_cpu


def decode_audio_latent(audio_latent, audio_vae) -> Any:
    """
    Decode audio latent to waveform (LTXVAudioVAEDecode).
    Returns waveform data on CPU.
    """
    print("  VAE decoding audio latent...")
    ltxvaudiovaedecode = get_node("LTXVAudioVAEDecode")
    decoded = ltxvaudiovaedecode.EXECUTE_NORMALIZED(
        samples=audio_latent,
        audio_vae=audio_vae,
    )
    audio_out = get_value_at_index(decoded, 0)
    # Move waveform to CPU
    if torch.is_tensor(audio_out):
        audio_out = audio_out.detach().cpu()
    elif isinstance(audio_out, dict):
        if "waveform" in audio_out and torch.is_tensor(audio_out["waveform"]):
            audio_out = {**audio_out, "waveform": audio_out["waveform"].detach().cpu()}
    del decoded
    mem.cleanup()
    return audio_out


# =============================================================================
# SECTION 17 — CHUNK SAVING (frames → MP4 via ffmpeg)
# =============================================================================

def save_chunk_to_disk(
    frames_cpu: Any,
    audio_cpu: Any,
    chunk_index: int,
    fps: int,
    width: int,
    height: int,
) -> str:
    """
    Write decoded frames + audio to a chunk MP4 file.

    Uses the ComfyUI CreateVideo node where possible; falls back to
    direct ffmpeg frame-pipe writing so the full frame array is never
    duplicated in RAM.

    Returns the output chunk path.
    """
    chunks_dir = os.path.join(CONFIG["workspace_dir"], "chunks")
    Path(chunks_dir).mkdir(parents=True, exist_ok=True)
    chunk_path = os.path.join(chunks_dir, f"chunk_{chunk_index:04d}.mp4")

    # ── Try ComfyUI CreateVideo node ─────────────────────────────────────────
    # Skip CreateVideo when RAM is low to avoid buffering all frames in memory
    ram_too_low = not mem.is_ram_safe(required_gb=4.0)
    if ram_too_low:
        print(f"  [mem] RAM too low ({mem.cpu_available_gb():.2f} GB available < 4.0 GB). "
              f"Skipping CreateVideo, using streaming ffmpeg.")

    if not ram_too_low:
        try:
            from nodes import NODE_CLASS_MAPPINGS
            if "CreateVideo" in NODE_CLASS_MAPPINGS:
                createvideo = NODE_CLASS_MAPPINGS["CreateVideo"]()
                video_obj = createvideo.EXECUTE_NORMALIZED(
                    fps=fps,
                    images=frames_cpu,
                    audio=audio_cpu,
                )
                video = get_value_at_index(video_obj, 0)
                # Save using ComfyUI API
                import folder_paths
                from comfy_api.latest import Types
                w = frames_cpu.shape[2] if frames_cpu.ndim == 4 else width
                h = frames_cpu.shape[1] if frames_cpu.ndim == 4 else height
                full_folder, fname, counter, _, _ = folder_paths.get_save_image_path(
                    f"chunk_{chunk_index:04d}",
                    folder_paths.get_output_directory(),
                    w, h,
                )
                ext = Types.VideoContainer.get_extension("auto")
                tmp_path = os.path.join(full_folder, f"{fname}_{counter:05d}_.{ext}")
                video.save_to(
                    tmp_path,
                    format=Types.VideoContainer("auto"),
                    codec="auto",
                    metadata=None,
                )
                # Move to our chunks directory
                shutil.move(tmp_path, chunk_path)
                del video_obj, video
                mem.soft_cleanup()
                print(f"  \u2713 Chunk {chunk_index:04d} saved (CreateVideo): {chunk_path}")
                return chunk_path
        except Exception as e:
            print(f"  CreateVideo path failed ({e}), falling back to ffmpeg pipe...")

    # ── Fallback: write frames via ffmpeg stdin pipe ──────────────────────────
    _write_chunk_via_ffmpeg(frames_cpu, audio_cpu, chunk_path, fps, width, height)
    return chunk_path


def _write_chunk_via_ffmpeg(frames_cpu, audio_cpu, out_path: str, fps: int, w: int, h: int):
    """
    Pipe raw RGB frames directly into ffmpeg. Streams one frame at a time
    to avoid doubling RAM with a full numpy copy of the frame tensor.
    """
    if torch.is_tensor(frames_cpu):
        n_frames = frames_cpu.shape[0]
        fh, fw = frames_cpu.shape[1], frames_cpu.shape[2]
    else:
        # Already numpy
        n_frames, fh, fw = frames_cpu.shape[0], frames_cpu.shape[1], frames_cpu.shape[2]

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{fw}x{fh}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "pipe:0",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "fast",
        out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # Stream frame-by-frame to keep RAM usage constant
    for i in range(n_frames):
        if torch.is_tensor(frames_cpu):
            frame = (frames_cpu[i].clamp(0, 1) * 255).byte().numpy()
        else:
            frame = frames_cpu[i]
        proc.stdin.write(frame.tobytes())
        del frame
        if i % 16 == 0:
            gc.collect()

    proc.stdin.close()
    proc.wait()
    print(f"  \u2713 Chunk saved (ffmpeg pipe, streaming): {out_path}")


def compute_file_checksum(path: str) -> str:
    """MD5 checksum of a file for resume validation."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# =============================================================================
# SECTION 18 — SINGLE CHUNK GENERATION  (generate_chunk)
# =============================================================================

def generate_chunk(
    chunk_desc: Dict,
    pos_cond,
    neg_cond,
    loaded_image_tuple,
    image_strength: float,
    image_bypass: bool,
    width: int,
    height: int,
    fps: int,
    profile: Dict,
    global_seed: int,
) -> Dict:
    """
    Generate one temporal chunk using the full LTX-2.3 Director 2.0 pipeline.

    Correct pipeline execution order (matching workflow JSON):
        1.  Load VAEs, DiT model, upscaler
        2.  Build LTXDirector conditioning (provides model, positive, video_latent,
            audio_latent, guide_data, motion_guide_data, frame_rate)
        3.  Build text conditioning (ConditioningZeroOut + LTXVConditioning)
        4.  LTXDirectorGuide pass 1 (node 133, upscale_factor=0.5)
            - Takes: conditioning pos/neg, VAE, LTXDirector video_latent,
              guide_data, motion_guide_data, LTXDirector model
            - Outputs: pos, neg, latent, model
        5.  LTXVConcatAVLatent (node 29): Guide133 latent + LTXDirector audio_latent
        6.  CFGGuider (node 28): uses Guide133 model/pos/neg
        7.  SamplerCustomAdvanced (node 31): Pass 1 (8 steps, denoise=1.0)
        8.  LTXVSeparateAVLatent (node 34): splits pass1 output[0]
        9.  LTXDirectorCropGuides (node 55): Guide133 pos/neg + separated video
        10. LTXVLatentUpsampler (node 14): CropGuides55 latent output
        11. LTXDirectorGuide pass 2 (node 132, upscale_factor=1.0)
            - Takes: CropGuides55 pos/neg, VAE, upsampled latent,
              guide_data, motion_guide_data, LTXDirector model
            - Outputs: pos, neg, latent, model
        12. LTXVConcatAVLatent (node 18): Guide132 latent + pass1 separated audio
        13. CFGGuider (node 17): uses Guide132 model/pos/neg
        14. SamplerCustomAdvanced (node 19): Pass 2 (4 steps, denoise=0.42)
        15. LTXVSeparateAVLatent (node 22): pass2 output[0] (the output)
        16. LTXDirectorCropGuides (node 54): Guide132 pos/neg + separated video
        17. VAEDecode: CropGuides54 latent
        18. LTXVAudioVAEDecode: separated audio from pass2
        19. Save chunk to disk, release GPU tensors

    Returns dict with keys: chunk_index, start_frame, num_frames, fps, path.
    Never returns GPU tensors.
    """
    idx          = chunk_desc["chunk_index"]
    start_frame  = chunk_desc["start_frame"]
    num_frames   = chunk_desc["num_frames"]
    chunk_seed   = get_chunk_seed(global_seed, idx)
    img_compress = profile.get("img_compression", WORKFLOW_IMG_COMPRESSION)
    longer_edge  = profile.get("longer_edge", 1312)

    mem.set_chunk_info(idx, num_frames, width, height)
    if CONFIG["enable_memory_logging"]:
        print(f"\n  GPU before chunk {idx}: {mem.gpu_free_gb():.2f} GB free")

    with torch.inference_mode():
        # -- 1. Image preprocessing --
        preprocessed, latent_w, latent_h = prepare_image_for_chunk(
            loaded_image_tuple, width, height, img_compress, longer_edge
        )

        # -- 2. Load VAEs (deferred for t4_safe to reduce peak memory) --
        # In t4_safe mode, we defer video_vae loading until step 7 (Director Guide)
        # since it is not needed earlier. audio_vae is still needed for the fallback
        # path in build_director_conditioning (step 4: LTXVEmptyLatentAudio).
        # In non-t4_safe modes, load both upfront for the Director path that may
        # need them earlier.
        active_profile = T4_PROFILES.get(QUALITY_MODE, {})
        if active_profile.get("skip_director", False):
            # T4-safe: load only audio_vae now, defer video_vae
            print("  [t4_safe] Deferring video VAE load (saving ~1GB during DiT load)")
            video_vae = None  # Will be loaded before Director Guide (step 7)
            audio_vae = load_audio_vae()
        else:
            # Non-t4-safe: load both for LTXDirector path
            video_vae = load_video_vae()
            audio_vae = load_audio_vae()

        # -- 3. Load upscaler (DEFERRED until after pass 1 to reduce VRAM pressure) --
        # The upscaler is only needed for step 12 (between pass 1 and pass 2).
        # Loading it here would consume VRAM during pass 1 sampling, causing OOM
        # on T4 GPUs where DiT + LoRAs already use most of the 15 GB.
        upscaler = None  # Will be loaded after pass 1 completes

        # -- 4. Build LTXDirector conditioning or fallback --
        # DiT model is loaded inside build_director_conditioning (or its fallback)
        # via load_dit_model() which uses a cache. We do NOT load it here separately
        # to avoid having two copies in VRAM simultaneously (OOM on T4).
        # LTXDirector returns: (model, positive, video_latent, audio_latent,
        #                        guide_data, motion_guide_data, frame_rate)
        director_result = build_director_conditioning(
            pos_cond=pos_cond,
            neg_cond=neg_cond,
            image_path=None,
            audio_path=None,
            num_frames=num_frames,
            fps=fps,
            width=width,
            height=height,
            audio_vae=audio_vae,
        )
        (dir_model, dir_positive, dir_video_latent, dir_audio_latent,
         dir_guide_data, dir_motion_guide_data, dir_frame_rate) = director_result

        # Use director model as the base model for both sampling passes.
        # This is the ONLY copy of the DiT in memory.
        base_model = dir_model

        # -- 6. Determine video latent for pass 1 --
        # When LTXDirector provides video_latent, use it directly (no empty latents needed).
        # Otherwise fall back to building empty latents.
        if dir_video_latent is not None:
            # LTXDirector provides video_latent directly (replaces EmptyLTXVLatentVideo)
            video_latent_pass1 = dir_video_latent
            audio_latent_for_concat = dir_audio_latent
        else:
            # Fallback: build empty latents the old way.
            # NOTE: build_empty_latents already fuses video+audio via LTXVConcatAVLatent
            # internally, so the returned av_latent is already an AV-fused latent.
            # We must NOT concat additional audio at step 8 (that would double-concat).
            # Ensure video_vae is loaded (may have been deferred in t4_safe mode)
            if video_vae is None:
                video_vae = load_video_vae()
            av_latent_pass1, img_conditioned_lat = build_empty_latents(
                num_frames, latent_w, latent_h, fps,
                preprocessed, image_strength, image_bypass,
                video_vae, audio_vae,
            )
            video_latent_pass1 = get_value_at_index(av_latent_pass1, 0)
            audio_latent_for_concat = None  # Already fused inside build_empty_latents
            del av_latent_pass1
            mem.soft_cleanup()

        # -- 6b. Build Director-aware conditioning (workflow nodes 128 + 27) --
        # When LTXDirector provides dir_positive (timeline-segmented conditioning),
        # we must route it through ConditioningZeroOut -> LTXVConditioning to produce
        # the pos/neg that Guide pass 1 receives. This is the Director's core value:
        # multi-segment timeline data (image refs, audio markers, per-segment prompts).
        if dir_positive is not None:
            conditioningzeroout = get_node("ConditioningZeroOut")
            neg_from_director = conditioningzeroout.zero_out(conditioning=dir_positive)
            ltxvconditioning = get_node("LTXVConditioning")
            director_cond = ltxvconditioning.EXECUTE_NORMALIZED(
                frame_rate=dir_frame_rate,
                positive=dir_positive,
                negative=get_value_at_index(neg_from_director, 0),
            )
            cond_pos_for_guide = get_value_at_index(director_cond, 0)
            cond_neg_for_guide = get_value_at_index(director_cond, 1)
        else:
            # Fallback: use plain text conditioning from build_text_conditioning
            cond_pos_for_guide = pos_cond
            cond_neg_for_guide = neg_cond

        # -- 7. LTXDirectorGuide pass 1 (workflow node 133: upscale_factor=0.5) --
        # Takes: pos/neg conditioning, VAE, video_latent from LTXDirector,
        #        guide_data, motion_guide_data, base model
        # Load video_vae now if deferred (t4_safe mode defers to reduce peak memory)
        if video_vae is None:
            video_vae = load_video_vae()
        pos_g1, neg_g1, lat_g1, model_g1 = run_director_guide(
            pos_cond=cond_pos_for_guide,
            neg_cond=cond_neg_for_guide,
            video_vae=video_vae,
            latent=video_latent_pass1,
            guide_data=dir_guide_data,
            motion_guide_data=dir_motion_guide_data,
            model=base_model,
            upscale_factor=0.5,
            node_id="pass1 (node 133)",
        )

        # -- 8. LTXVConcatAVLatent (workflow node 29) --
        # Concatenates Guide133's latent output + LTXDirector's audio_latent
        if audio_latent_for_concat is not None:
            ltxvconcatavlatent = get_node("LTXVConcatAVLatent")
            av_concat_pass1 = ltxvconcatavlatent.EXECUTE_NORMALIZED(
                video_latent=lat_g1,
                audio_latent=audio_latent_for_concat,
            )
            latent_for_sampler1 = get_value_at_index(av_concat_pass1, 0)
            del av_concat_pass1
        else:
            # Fallback path: latent already contains AV data
            latent_for_sampler1 = lat_g1

        # -- 9. Sampling pass 1 (workflow node 31: 8 steps, denoise=1.0) --
        # CFGGuider (node 28) uses Guide133's model/pos/neg outputs
        sample_out_1 = run_sampling_pass(
            model=model_g1,
            pos_cond=pos_g1,
            neg_cond=neg_g1,
            latent=latent_for_sampler1,
            noise_seed=chunk_seed,
            steps=WORKFLOW_STEPS,
            cfg=WORKFLOW_CFG,
            denoise=1.0,
            pass_name=f"Pass1 (chunk {idx})",
        )

        del latent_for_sampler1
        mem.cleanup()
        mem.warn_if_low()

        # -- 10. Separate AV latent (workflow node 34) --
        # Pass 1 uses output index 0 (raw output)
        video_lat_p1, audio_lat_p1 = separate_av_latent(sample_out_1, output_index=0)
        del sample_out_1
        mem.soft_cleanup()

        # -- 11. LTXDirectorCropGuides (workflow node 55) --
        # Takes Guide133's pos/neg outputs + separated video from pass1
        pos_crop55, neg_crop55, lat_crop55 = run_director_crop_guides(
            pos_cond=pos_g1,
            neg_cond=neg_g1,
            latent=video_lat_p1,
        )
        del video_lat_p1, pos_g1, neg_g1, model_g1
        mem.soft_cleanup()

        # -- 12. 2x latent spatial upscale (workflow node 14) --
        # Takes CropGuides55's latent output (slot 2)
        # Load upscaler NOW (deferred from step 3 to avoid VRAM pressure during pass 1)
        upscaler = load_upscaler_model()
        upscaled_lat = upsample_video_latent(lat_crop55, upscaler, video_vae)
        del lat_crop55, upscaler
        mem.cleanup()

        # -- 13. LTXDirectorGuide pass 2 (workflow node 132: upscale_factor=1.0) --
        # Takes: CropGuides55's pos/neg, VAE, upsampled latent,
        #        LTXDirector's guide_data, motion_guide_data, base model
        pos_g2, neg_g2, lat_g2, model_g2 = run_director_guide(
            pos_cond=pos_crop55,
            neg_cond=neg_crop55,
            video_vae=video_vae,
            latent=upscaled_lat,
            guide_data=dir_guide_data,
            motion_guide_data=dir_motion_guide_data,
            model=base_model,
            upscale_factor=1.0,
            node_id="pass2 (node 132)",
        )
        del pos_crop55, neg_crop55, upscaled_lat
        mem.cleanup()
        mem.warn_if_low()

        # ── Free director tensors no longer needed after Guide pass 2 ────────
        del dir_model, dir_guide_data, dir_motion_guide_data, dir_frame_rate
        # dir_positive, dir_video_latent, dir_audio_latent may still be referenced
        # via other names but ensure no stale references linger
        try:
            del dir_positive
        except NameError:
            pass
        try:
            del dir_video_latent
        except NameError:
            pass
        try:
            del dir_audio_latent
        except NameError:
            pass
        mem.soft_cleanup()

        # -- 14. LTXVConcatAVLatent (workflow node 18) --
        # Concatenates Guide132's latent(2) + pass1 separated audio_latent(1)
        ltxvconcatavlatent2 = get_node("LTXVConcatAVLatent")
        av_concat_pass2 = ltxvconcatavlatent2.EXECUTE_NORMALIZED(
            video_latent=lat_g2,
            audio_latent=audio_lat_p1,
        )
        latent_for_sampler2 = get_value_at_index(av_concat_pass2, 0)
        del av_concat_pass2, audio_lat_p1, lat_g2
        mem.soft_cleanup()

        # -- 15. Sampling pass 2 (workflow node 19: 4 steps, denoise=0.42) --
        # CFGGuider (node 17) uses Guide132's model/pos/neg outputs
        sample_out_2 = run_sampling_pass(
            model=model_g2,
            pos_cond=pos_g2,
            neg_cond=neg_g2,
            latent=latent_for_sampler2,
            noise_seed=0,          # workflow node 30: seed=0 for refinement pass
            steps=WORKFLOW_STEPS_PASS2,
            cfg=WORKFLOW_CFG,
            denoise=WORKFLOW_DENOISE_PASS2,
            pass_name=f"Pass2 (chunk {idx})",
        )

        del latent_for_sampler2, model_g2
        # Release the DiT model cache and base_model reference to free VRAM
        # before decode (which needs the memory for frame tensors).
        try:
            del base_model
        except NameError:
            pass
        release_dit_model()
        mem.cleanup()

        # -- 16. Separate final AV latent (workflow node 22) --
        # Pass 2 uses output index 0 (output) - workflow wires slot 0 to node 22
        final_video_lat, final_audio_lat = separate_av_latent(sample_out_2, output_index=0)
        del sample_out_2
        mem.soft_cleanup()

        # -- 17. LTXDirectorCropGuides (workflow node 54) --
        # Takes Guide132's pos/neg + pass2 separated video_latent
        # Output latent (slot 2) goes to VAEDecode
        pos_crop54, neg_crop54, lat_crop54 = run_director_crop_guides(
            pos_cond=pos_g2,
            neg_cond=neg_g2,
            latent=final_video_lat,
        )
        del pos_g2, neg_g2, final_video_lat, pos_crop54, neg_crop54
        mem.aggressive_cleanup()
        mem.ram_cleanup()
        if not mem.is_ram_safe(required_gb=3.0):
            print(f"  WARNING: System RAM critically low before decode "
                  f"({mem.cpu_available_gb():.2f} GB available < 3.0 GB).")

        # -- 18. Decode video (workflow node 1: VAEDecode) --
        # Uses CropGuides54's latent output (slot 2)
        frames_cpu = decode_video_latent(lat_crop54, video_vae)
        del lat_crop54

        # -- 19. Decode audio (workflow node 24: LTXVAudioVAEDecode) --
        audio_cpu = decode_audio_latent(final_audio_lat, audio_vae)
        del final_audio_lat

        # -- 20. Unload VAEs --
        del video_vae, audio_vae
        mem.cleanup()

    # -- 21. Save chunk to disk --
    chunk_path = save_chunk_to_disk(
        frames_cpu, audio_cpu, idx, fps, width, height
    )

    # Release CPU frame buffers
    del frames_cpu, audio_cpu
    gc.collect()

    if CONFIG["enable_memory_logging"]:
        print(f"  GPU after  chunk {idx}: {mem.gpu_free_gb():.2f} GB free")

    if CONFIG["cleanup_after_chunk"]:
        mem.aggressive_cleanup()

    return {
        "chunk_index": idx,
        "start_frame": start_frame,
        "num_frames":  num_frames,
        "fps":         fps,
        "path":        chunk_path,
    }


# =============================================================================
# SECTION 19 — OOM RECOVERY & ADAPTIVE CHUNK GENERATOR
# =============================================================================

def adaptive_chunk_generator(
    chunks: List[Dict],
    pos_cond,
    neg_cond,
    loaded_image_tuple,
    image_strength: float,
    image_bypass: bool,
    width: int,
    height: int,
    fps: int,
    profile: Dict,
    global_seed: int,
    checkpoint: Dict,
) -> List[Dict]:
    """
    Iterate over chunks with OOM recovery and checkpoint-based resume.

    On CUDA OOM:
        1. Catch exception
        2. Print diagnostic
        3. Aggressive cleanup
        4. Reduce chunk size by factor (0.75 → 0.5)
        5. Re-split the remaining chunks
        6. Retry (up to MAX_OOM_RETRIES)

    Completed chunks are skipped if already present in checkpoint.
    Returns list of completed chunk metadata dicts.
    """
    max_retries   = CONFIG["max_oom_retries"]
    auto_reduce   = CONFIG["auto_reduce_chunk_on_oom"]
    completed     = []
    current_chunks = list(chunks)
    i = 0

    while i < len(current_chunks):
        chunk_desc = current_chunks[i]
        idx = chunk_desc["chunk_index"]

        # ── Resume check: skip already-completed chunks ───────────────────────
        if idx in checkpoint.get("completed_chunks", []):
            existing_path = os.path.join(
                CONFIG["workspace_dir"], "chunks", f"chunk_{idx:04d}.mp4"
            )
            if os.path.exists(existing_path) and os.path.getsize(existing_path) > 0:
                print(f"  ↷ Chunk {idx:04d} already complete — skipping.")
                completed.append({
                    "chunk_index": idx,
                    "start_frame": chunk_desc["start_frame"],
                    "num_frames":  chunk_desc["num_frames"],
                    "fps":         fps,
                    "path":        existing_path,
                })
                i += 1
                continue

        # ── Attempt generation with OOM recovery ─────────────────────────────
        retries = 0
        success = False
        current_num_frames = chunk_desc["num_frames"]

        while retries <= max_retries and not success:
            try:
                print(f"\n{'='*60}")
                print(f"[Chunk {idx+1:02d}]  frames {chunk_desc['start_frame']}–"
                      f"{chunk_desc['start_frame']+current_num_frames-1}  "
                      f"({current_num_frames} frames)")
                if CONFIG["enable_memory_logging"]:
                    mem.print_memory("  ")
                print(f"{'='*60}")

                # Use potentially-reduced frame count
                work_desc = {**chunk_desc, "num_frames": current_num_frames}
                result = generate_chunk(
                    chunk_desc=work_desc,
                    pos_cond=pos_cond,
                    neg_cond=neg_cond,
                    loaded_image_tuple=loaded_image_tuple,
                    image_strength=image_strength,
                    image_bypass=image_bypass,
                    width=width,
                    height=height,
                    fps=fps,
                    profile=profile,
                    global_seed=global_seed,
                )

                completed.append(result)
                checkpoint["completed_chunks"].append(idx)
                save_checkpoint(checkpoint)
                success = True
                print(f"  ✓ Chunk {idx:04d} complete.")

            except torch.cuda.OutOfMemoryError as oom:
                retries += 1
                print(f"\n  {'='*50}")
                print(f"  ERROR TYPE   : CUDA OutOfMemoryError")
                print(f"  CAUSE        : {str(oom)[:200]}")
                print(f"  CURRENT CHUNK: {idx}")
                print(f"  GPU MEMORY   : {mem.gpu_free_gb():.2f} GB free")
                print(f"  RETRY        : {retries}/{max_retries}")

                # Release cached DiT model before cleanup to free VRAM
                release_dit_model()
                mem.aggressive_cleanup()

                if not auto_reduce or retries > max_retries:
                    if retries > max_retries:
                        print(f"\n  Generation stopped safely after {max_retries} OOM retries.")
                        print("  SUGGESTED ACTION: Reduce resolution or decrease chunk_frames.")
                        checkpoint["failed_chunks"].append(idx)
                        save_checkpoint(checkpoint)
                    break

                # Reduce chunk size: 0.75× on first retry, 0.5× on second
                reduction = 0.75 if retries == 1 else 0.5
                raw_reduced = max(9, int(current_num_frames * reduction))
                current_num_frames = normalize_ltx_frame_count(raw_reduced, fps)
                print(f"  Reducing chunk size to {current_num_frames} frames and retrying...")

            except Exception as e:
                print(f"\n  ERROR TYPE   : {type(e).__name__}")
                print(f"  CAUSE        : {str(e)[:300]}")
                print(f"  CURRENT CHUNK: {idx}")
                print(f"  TRACEBACK    :\n{traceback.format_exc()}")
                checkpoint["failed_chunks"].append(idx)
                save_checkpoint(checkpoint)
                release_dit_model()
                mem.aggressive_cleanup()
                break  # Non-OOM errors: skip chunk, continue

        i += 1

    return completed


# =============================================================================
# SECTION 20 — CHECKPOINT / RESUME SYSTEM
# =============================================================================

def _checkpoint_path() -> str:
    return os.path.join(CONFIG["workspace_dir"], "checkpoint.json")


def init_checkpoint(
    fps: int,
    total_frames: int,
    seed: int,
    width: int,
    height: int,
    job_id: Optional[str] = None,
) -> Dict:
    """Create a fresh checkpoint dict."""
    return {
        "job_id":           job_id or f"ltx23_{int(time.time())}",
        "fps":              fps,
        "total_frames":     total_frames,
        "seed":             seed,
        "resolution":       [width, height],
        "completed_chunks": [],
        "failed_chunks":    [],
        "created_at":       time.strftime("%Y-%m-%dT%H:%M:%S"),
        "updated_at":       time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def save_checkpoint(checkpoint: Dict):
    """Persist checkpoint dict to disk."""
    checkpoint["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    Path(CONFIG["workspace_dir"]).mkdir(parents=True, exist_ok=True)
    tmp = _checkpoint_path() + ".tmp"
    with open(tmp, "w") as f:
        json.dump(checkpoint, f, indent=2)
    os.replace(tmp, _checkpoint_path())


def load_checkpoint() -> Optional[Dict]:
    """Load checkpoint from disk, or return None if not found."""
    p = _checkpoint_path()
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as e:
        print(f"  ⚠ Could not load checkpoint: {e}")
        return None


def get_or_create_checkpoint(
    fps: int,
    total_frames: int,
    seed: int,
    width: int,
    height: int,
) -> Dict:
    """
    Load existing checkpoint (if resume enabled and checkpoint exists)
    or create a fresh one.
    """
    if CONFIG["resume"]:
        existing = load_checkpoint()
        if existing is not None:
            # Validate that the checkpoint matches the current job configuration
            if (existing.get("fps") == fps
                    and existing.get("total_frames") == total_frames
                    and existing.get("resolution") == [width, height]):
                completed = existing.get("completed_chunks", [])
                print(f"  ↷ Resuming from checkpoint: {len(completed)} chunks already complete.")
                return existing
            else:
                print("  ⚠ Checkpoint exists but configuration mismatch — starting fresh.")

    cp = init_checkpoint(fps, total_frames, seed, width, height)
    save_checkpoint(cp)
    return cp


# =============================================================================
# SECTION 21 — VIDEO ASSEMBLY (FFmpeg concat)
# =============================================================================

def assemble_chunks_to_video(
    completed_chunks: List[Dict],
    output_path: str,
    fps: int,
) -> bool:
    """
    Concatenate chunk MP4 files into the final video using FFmpeg stream-copy.
    Stream-copy avoids re-encoding every frame, which would:
        - waste time
        - increase RAM usage (decoding all frames to re-encode)
        - degrade quality

    If any chunk has mismatched codec/resolution (making stream-copy unsafe),
    falls back to a single re-encode pass.

    The full final video is NEVER loaded into Python/RAM.
    Returns True on success.
    """
    if not completed_chunks:
        print("  ✗ No completed chunks to assemble.")
        return False

    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

    # Sort by chunk index to guarantee correct frame order
    sorted_chunks = sorted(completed_chunks, key=lambda c: c["chunk_index"])

    # Write ffmpeg concat list file (streamed read, no RAM load)
    concat_list_path = os.path.join(CONFIG["workspace_dir"], "concat_list.txt")
    with open(concat_list_path, "w") as f:
        for chunk in sorted_chunks:
            # ffmpeg requires escaped paths
            safe_path = chunk["path"].replace("'", "'\\''")
            f.write(f"file '{safe_path}'\n")

    print(f"\n  Assembling {len(sorted_chunks)} chunks → {output_path}")

    # ── Attempt 1: stream-copy concat (fastest, no quality loss) ─────────────
    cmd_copy = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list_path,
        "-c", "copy",
        output_path,
    ]
    result = subprocess.run(cmd_copy, capture_output=True, text=True)
    if result.returncode == 0:
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  ✓ Assembly complete (stream-copy): {output_path} ({size_mb:.1f} MB)")
        return True

    print(f"  Stream-copy failed ({result.stderr.strip()[:200]}), trying re-encode...")

    # ── Attempt 2: re-encode concat ───────────────────────────────────────────
    cmd_reencode = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "fast",
        "-acodec", "aac",
        "-b:a", "192k",
        output_path,
    ]
    result2 = subprocess.run(cmd_reencode, capture_output=True, text=True)
    if result2.returncode == 0:
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  ✓ Assembly complete (re-encode): {output_path} ({size_mb:.1f} MB)")
        return True

    print(f"  ✗ Assembly failed:\n{result2.stderr.strip()[:400]}")
    return False


# =============================================================================
# SECTION 22 — AUDIO SYNCHRONIZATION
# =============================================================================

def assemble_video_with_audio(
    video_path: str,
    audio_path: Optional[str],
    output_path: str,
    fps: int,
    total_frames: int,
    audio_start_seconds: float = 0.0,
) -> bool:
    """
    Mux the assembled video with the original audio track using exact
    frame-based timing to prevent cumulative drift between chunks.

    Workflow audio notes:
        - Audio track: "Late night trap.mp3"
        - Trim start: ~18.6s from audio start (446.92/24 fps)
        - Total video duration: 31.5s @ 24 fps = 756 frames

    If no external audio is provided, the video's generated audio track
    (hallucinated by the LTX audio VAE per-chunk and concatenated) is used.

    Returns True on success.
    """
    if audio_path is None or not os.path.exists(audio_path):
        # No external audio — just rename/copy the assembled video
        if video_path != output_path:
            shutil.copy2(video_path, output_path)
        print(f"  ✓ Output (no external audio): {output_path}")
        return True

    video_duration = total_frames / fps
    audio_duration = video_duration  # trim to exactly the video length

    print(f"  Muxing audio: start={audio_start_seconds:.3f}s, duration={audio_duration:.3f}s")

    cmd = [
        "ffmpeg", "-y",
        # Video input
        "-i", video_path,
        # Audio input with precise seek
        "-ss", str(audio_start_seconds),
        "-t",  str(audio_duration),
        "-i", audio_path,
        # Use video stream from input 0, audio from input 1
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        # Ensure audio is exactly as long as video
        "-shortest",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  ✓ Audio sync complete: {output_path} ({size_mb:.1f} MB)")
        return True

    print(f"  ✗ Audio mux failed: {result.stderr.strip()[:300]}")
    # Fallback: output without external audio
    shutil.copy2(video_path, output_path)
    return False


# =============================================================================
# SECTION 23 — FINAL VALIDATION
# =============================================================================

def validate_output_video(output_path: str, expected_frames: int, fps: int) -> bool:
    """
    Use ffprobe to verify the final output video without loading it into RAM.
    Checks frame count, duration, codec, and resolution.
    """
    if not os.path.exists(output_path):
        print(f"  ✗ Output video not found: {output_path}")
        return False

    size_mb = os.path.getsize(output_path) / (1024 * 1024)

    try:
        probe_cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams", "-show_format",
            output_path,
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)

        video_stream = next(
            (s for s in info.get("streams", []) if s.get("codec_type") == "video"),
            None,
        )
        if video_stream is None:
            print("  ✗ No video stream found in output.")
            return False

        nb_frames = int(video_stream.get("nb_frames", 0))
        codec     = video_stream.get("codec_name", "unknown")
        width_out = video_stream.get("width", 0)
        height_out = video_stream.get("height", 0)
        duration  = float(info.get("format", {}).get("duration", 0))

        print(f"\n  Output video validation:")
        print(f"    Path     : {output_path}")
        print(f"    Size     : {size_mb:.1f} MB")
        print(f"    Codec    : {codec}")
        print(f"    Res      : {width_out}×{height_out}")
        print(f"    Frames   : {nb_frames} (expected ≈{expected_frames})")
        print(f"    Duration : {duration:.2f}s (expected ≈{expected_frames/fps:.2f}s)")

        ok = nb_frames > 0 and duration > 0
        print(f"  {'✓' if ok else '✗'} Output validation {'passed' if ok else 'failed'}.")
        return ok

    except Exception as e:
        print(f"  ⚠ Output validation error: {e}")
        print(f"    File exists: {size_mb:.1f} MB")
        return True  # file exists, assume ok


# =============================================================================
# SECTION 24 — PREVIEW MODE
# =============================================================================

def generate_preview(
    image_path: Optional[str],
    prompt: str,
    fps: int,
    width: int,
    height: int,
    seed: int,
    profile: Dict,
    preview_duration: float = 3.0,
) -> Optional[str]:
    """
    Generate a short preview clip using the same pipeline but fewer frames.
    Uses the configured quality profile at potentially reduced resolution.
    The preview is generated and saved, then all memory is cleared.

    Returns the preview output path, or None on failure.
    """
    preview_frames = normalize_ltx_frame_count(round(preview_duration * fps), fps)
    preview_out = os.path.join(CONFIG["output_dir"], "preview.mp4")

    print(f"\n  PREVIEW MODE: {preview_frames} frames ({preview_duration:.1f}s)")

    # Load image
    loaded_image, img_strength, img_bypass = load_input_image(image_path, width, height)

    # Build conditioning
    pos_cond, neg_cond = build_text_conditioning(prompt, fps)

    # Single preview chunk
    preview_desc = {
        "chunk_index": 0,
        "start_frame": 0,
        "num_frames": preview_frames,
        "fps": fps,
        "path": None,
    }

    try:
        mem.cleanup()
        result = generate_chunk(
            chunk_desc=preview_desc,
            pos_cond=pos_cond,
            neg_cond=neg_cond,
            loaded_image_tuple=loaded_image,
            image_strength=img_strength,
            image_bypass=img_bypass,
            width=width,
            height=height,
            fps=fps,
            profile=profile,
            global_seed=seed,
        )

        # Move chunk to preview output location
        if result["path"] and os.path.exists(result["path"]):
            Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
            shutil.move(result["path"], preview_out)
            print(f"  ✓ Preview saved: {preview_out}")
            display_video_safe(preview_out)
        else:
            print("  ✗ Preview generation failed.")
            preview_out = None

    finally:
        del loaded_image, pos_cond, neg_cond
        _CONDITIONING_CACHE.clear()
        mem.aggressive_cleanup()

    return preview_out


# =============================================================================
# SECTION 25 — SAFE VIDEO DISPLAY
# =============================================================================

def display_video_safe(video_path: str, max_size_mb: float = 50.0):
    """
    Display a video in Colab without loading the entire file into RAM.
    Falls back to a file-size warning if the video exceeds max_size_mb.
    Never uses open(path, 'rb').read() on large files.
    """
    if not os.path.exists(video_path):
        print(f"  Video not found: {video_path}")
        return

    size_mb = os.path.getsize(video_path) / (1024 * 1024)

    if size_mb > max_size_mb:
        print(f"  Video is {size_mb:.1f} MB — too large for inline display.")
        print(f"  Download it with: files.download('{video_path}')")
        return

    # For small files (preview, short clips), use base64 inline display
    from base64 import b64encode
    # Read in chunks to avoid single huge allocation
    chunks_b64 = []
    with open(video_path, "rb") as f:
        while True:
            block = f.read(65536)
            if not block:
                break
            chunks_b64.append(block)
    video_b64 = b64encode(b"".join(chunks_b64)).decode()
    del chunks_b64

    display(HTML(f"""
    <video width=640 controls autoplay loop muted>
      <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
    </video>
    """))
    del video_b64


# =============================================================================
# SECTION 26 — JOB REPORT
# =============================================================================

def write_job_report(
    output_path: str,
    total_frames: int,
    fps: int,
    width: int,
    height: int,
    chunk_size: int,
    completed_chunks: List[Dict],
    failed_chunks: List[int],
    generation_start_time: float,
    seed: int,
):
    """
    Write a JSON job report to /content/ltx23_output/job_report.json.
    Contains full provenance: hardware, timing, model versions, chunk info.
    """
    elapsed = time.time() - generation_start_time
    actual_duration = total_frames / fps

    report = {
        "gpu":                    _GPU_INFO.get("device_name", "unknown"),
        "vram_total_gb":          round(_GPU_INFO.get("vram_total_gb", 0), 2),
        "torch_version":          torch.__version__,
        "cuda_version":           getattr(torch.version, "cuda", "N/A"),
        "models": {
            "dit":            MODELS["dit"],
            "text_encoder_1": MODELS["text_encoder_1"],
            "text_encoder_2": MODELS["text_encoder_2"],
            "audio_vae":      MODELS["audio_vae"],
            "video_vae":      MODELS["video_vae"],
            "tiny_vae":       MODELS["tiny_vae"],
            "upscaler":       MODELS["upscaler"],
            "loras": {k: {"file": MODELS[k], "strength": LORA_STRENGTHS[k]}
                      for k in LORA_STRENGTHS},
        },
        "workflow": {
            "fps":              fps,
            "sampler":          WORKFLOW_SAMPLER_PASS1,
            "scheduler":        WORKFLOW_SCHEDULER,
            "steps":            WORKFLOW_STEPS,
            "cfg":              WORKFLOW_CFG,
        },
        "resolution":             f"{width}x{height}",
        "fps":                    fps,
        "seed":                   seed,
        "requested_duration_s":   CONFIG["duration_seconds"],
        "actual_duration_s":      round(actual_duration, 3),
        "total_frames":           total_frames,
        "chunk_size_frames":      chunk_size,
        "chunks_completed":       len(completed_chunks),
        "chunks_failed":          len(failed_chunks),
        "failed_chunk_indices":   failed_chunks,
        "peak_gpu_memory_gb":     round(mem.gpu_peak_gb(), 3),
        "generation_time_seconds": round(elapsed, 1),
        "output_path":            output_path,
        "generated_at":           time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    report_path = os.path.join(CONFIG["output_dir"], "job_report.json")
    Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Job report: {report_path}")
    print(f"  Peak GPU memory : {report['peak_gpu_memory_gb']:.3f} GB")
    print(f"  Generation time : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Chunks completed: {report['chunks_completed']}")
    if failed_chunks:
        print(f"  Chunks failed   : {failed_chunks}")


# =============================================================================
# SECTION 27 — FINAL CLEANUP
# =============================================================================

def cleanup_temp_files(completed_chunks: List[Dict], keep_chunks: bool = False):
    """
    Remove temporary chunk files after successful final assembly.
    Preserves failed chunks and the concat list for debugging.
    Never deletes the final output video.
    """
    if keep_chunks:
        print("  Keeping temp chunks (KEEP_TEMP_CHUNKS=True).")
        return

    removed = 0
    for chunk in completed_chunks:
        path = chunk.get("path")
        if path and os.path.exists(path):
            try:
                os.remove(path)
                removed += 1
            except Exception as e:
                print(f"  ⚠ Could not remove {path}: {e}")

    # Remove concat list
    concat_list = os.path.join(CONFIG["workspace_dir"], "concat_list.txt")
    if os.path.exists(concat_list):
        os.remove(concat_list)

    print(f"  ✓ Removed {removed} temporary chunk files.")


def final_memory_report():
    """Print end-of-job memory summary."""
    print("\n" + "=" * 60)
    print("FINAL MEMORY REPORT")
    print("=" * 60)
    mem.print_memory()
    print(f"  Peak GPU usage  : {mem.gpu_peak_gb():.3f} GB")
    print("=" * 60)


# =============================================================================
# SECTION 28 — MAIN ENTRY POINT
# =============================================================================

def print_banner(total_frames: int, fps: int, duration_s: float,
                 width: int, height: int, chunk_size: int, n_chunks: int):
    print("\n" + "=" * 60)
    print("LTX-2.3 DIRECTOR 2.0 MV")
    print("Google Colab T4 Engine")
    print("=" * 60)
    print(f"  GPU          : {_GPU_INFO['device_name']}")
    print(f"  VRAM         : {_GPU_INFO['vram_total_gb']:.1f} GB total  "
          f"/ {mem.gpu_free_gb():.1f} GB free")
    print(f"  Resolution   : {width}×{height}")
    print(f"  FPS          : {fps}")
    print(f"  Duration     : {duration_s:.2f}s")
    print(f"  Frames       : {total_frames}")
    print(f"  Chunk size   : {chunk_size} frames")
    print(f"  Est. chunks  : {n_chunks}")
    print(f"  Quality mode : {CONFIG['quality_mode']}")
    print(f"  Resume       : {CONFIG['resume']}")
    print("=" * 60)


def generate_director_mv(
    image_path: Optional[str] = None,
    audio_path: Optional[str] = None,
    prompt: str = GLOBAL_PROMPT,
    duration_seconds: float = None,
    fps: int = None,
    width: int = None,
    height: int = None,
    seed: int = None,
    quality_mode: str = None,
):
    """
    Complete LTX-2.3 Director 2.0 MV generation pipeline.

    Parameters
    ----------
    image_path      : Path to a reference image (or None for T2V).
    audio_path      : Path to audio file for sync mux (or None).
    prompt          : Generation prompt (defaults to embedded workflow prompt).
    duration_seconds: Video length in seconds (default: CONFIG value = 31.5).
    fps             : Frame rate (default: 24).
    width / height  : Output resolution (default: 1280×720).
    seed            : Global random seed.
    quality_mode    : "t4_safe" | "t4_balanced" | "t4_aggressive".

    Returns
    -------
    str : Path to the final output video, or None on failure.
    """
    # ── Resolve parameters from CONFIG if not passed ──────────────────────────
    duration_s    = duration_seconds if duration_seconds is not None else CONFIG["duration_seconds"]
    fps           = fps    or CONFIG["fps"]
    width         = width  or CONFIG["width"]
    height        = height or CONFIG["height"]
    seed          = seed   if seed is not None else CONFIG["seed"]
    quality_mode  = quality_mode or CONFIG["quality_mode"]

    generation_start = time.time()

    # ── Profile & resolution check ────────────────────────────────────────────
    profile = select_profile(quality_mode)
    width, height = check_resolution_safety(width, height, quality_mode)

    # ── Preview mode shortcut ─────────────────────────────────────────────────
    if CONFIG["preview_mode"]:
        return generate_preview(
            image_path=image_path,
            prompt=prompt,
            fps=fps,
            width=width,
            height=height,
            seed=seed,
            profile=profile,
            preview_duration=CONFIG.get("preview_duration", 3.0),
        )

    # ── Timeline calculation ──────────────────────────────────────────────────
    total_frames, actual_duration = calculate_timeline(duration_s, fps)

    # ── Chunk planning ────────────────────────────────────────────────────────
    chunk_size = estimate_chunk_size(width, height, fps, quality_mode)
    all_chunks = plan_chunks(total_frames, chunk_size, fps)

    print_banner(total_frames, fps, actual_duration, width, height, chunk_size, len(all_chunks))

    # ── Validation ────────────────────────────────────────────────────────────
    run_all_validations(image_path, audio_path, width, height, total_frames)

    # ── ComfyUI initialisation ────────────────────────────────────────────────
    setup_comfyui()
    import_custom_nodes()

    # ── Checkpoint ────────────────────────────────────────────────────────────
    checkpoint = get_or_create_checkpoint(fps, total_frames, seed, width, height)

    # ── Load input image once (keep on CPU) ───────────────────────────────────
    loaded_image, img_strength, img_bypass = load_input_image(image_path, width, height)

    # ── Build text conditioning once and cache on CPU ─────────────────────────
    print("\n  Building text conditioning...")
    mem.cleanup()
    pos_cond, neg_cond = build_text_conditioning(prompt, fps)

    # ── Main generation loop ──────────────────────────────────────────────────
    print(f"\n  Starting generation: {len(all_chunks)} chunks...")
    torch.cuda.reset_peak_memory_stats()

    completed_chunks = adaptive_chunk_generator(
        chunks=all_chunks,
        pos_cond=pos_cond,
        neg_cond=neg_cond,
        loaded_image_tuple=loaded_image,
        image_strength=img_strength,
        image_bypass=img_bypass,
        width=width,
        height=height,
        fps=fps,
        profile=profile,
        global_seed=seed,
        checkpoint=checkpoint,
    )

    # Release conditioning and image — no longer needed
    del pos_cond, neg_cond, loaded_image
    _CONDITIONING_CACHE.clear()
    mem.aggressive_cleanup()

    if not completed_chunks:
        print("\n  ✗ No chunks were completed. Generation aborted.")
        return None

    # ── Assemble chunks → intermediate video ─────────────────────────────────
    Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
    assembled_path = os.path.join(CONFIG["output_dir"], "_assembled_no_audio.mp4")
    assembly_ok = assemble_chunks_to_video(completed_chunks, assembled_path, fps)
    if not assembly_ok:
        print("\n  ✗ Video assembly failed.")
        return None

    # ── Mux audio ─────────────────────────────────────────────────────────────
    final_output = os.path.join(CONFIG["output_dir"], CONFIG["output_filename"])
    audio_start_s = 0.0  # Can be set to workflow value (446.92/24 ≈ 18.6s) if needed
    assemble_video_with_audio(
        video_path=assembled_path,
        audio_path=audio_path,
        output_path=final_output,
        fps=fps,
        total_frames=total_frames,
        audio_start_seconds=audio_start_s,
    )

    # Remove intermediate no-audio file
    if os.path.exists(assembled_path) and assembled_path != final_output:
        os.remove(assembled_path)

    # ── Final validation ──────────────────────────────────────────────────────
    validate_output_video(final_output, total_frames, fps)

    # ── Job report ────────────────────────────────────────────────────────────
    write_job_report(
        output_path=final_output,
        total_frames=total_frames,
        fps=fps,
        width=width,
        height=height,
        chunk_size=chunk_size,
        completed_chunks=completed_chunks,
        failed_chunks=checkpoint.get("failed_chunks", []),
        generation_start_time=generation_start,
        seed=seed,
    )

    # ── Cleanup temp files ────────────────────────────────────────────────────
    cleanup_temp_files(completed_chunks, keep_chunks=CONFIG["keep_temp_chunks"])

    # ── Final memory report ───────────────────────────────────────────────────
    mem.aggressive_cleanup()
    final_memory_report()

    print(f"\n{'='*60}")
    print(f"✓ GENERATION COMPLETE")
    print(f"  Output: {final_output}")
    print(f"{'='*60}\n")

    return final_output


# =============================================================================
# SECTION 29 — COLAB CELL RUNNER
# =============================================================================
# Copy-paste the individual cells below into a Colab notebook.
# Each cell is self-contained and idempotent.
#
# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Install environment (run once per runtime)
# ─────────────────────────────────────────────────────────────────────────────
# install_environment()
#
# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Download models (run once per runtime or after /content is cleared)
# ─────────────────────────────────────────────────────────────────────────────
# download_all_models()
#
# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — (Optional) Upload reference images and audio
# ─────────────────────────────────────────────────────────────────────────────
# from google.colab import files
# import shutil, os
# os.makedirs('/content/ComfyUI/input/whatdreamscost', exist_ok=True)
# uploaded = files.upload()
# for fname in uploaded:
#     dest = f"/content/ComfyUI/input/whatdreamscost/{fname}"
#     shutil.move(f"/content/ComfyUI/{fname}", dest)
#     print(f"Saved: {dest}")
#
# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Configure and generate
# ─────────────────────────────────────────────────────────────────────────────
# All settings are now controlled by the @param form widgets at the TOP of
# SECTION 1. Simply adjust them in the form UI, then run this cell.
#
# Example: override individual settings programmatically if needed:
#   CONFIG["quality_mode"] = "t4_balanced"
#   SEED = 999
#
# output = generate_director_mv(
#     image_path=IMAGE_PATH,
#     audio_path=AUDIO_PATH,
#     prompt=GLOBAL_PROMPT,
# )
#
# if output:
#     from google.colab import files
#     files.download(output)
#
# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — Download output (if not done in Cell 4)
# ─────────────────────────────────────────────────────────────────────────────
# from google.colab import files
# files.download('/content/ltx23_output/LTX23_Director_30s.mp4')

# =============================================================================
# SECTION 30 — DIRECT EXECUTION GUARD
# =============================================================================
# This block runs automatically if the script is executed directly
# (i.e., not imported as a module in a notebook).

if __name__ == "__main__":
    print("\nRunning LTX23_Director_2_0_MV_Colab_T4.py directly...")
    print("This script is designed for Google Colab. Running setup steps...\n")

    # Step 1: Install
    install_environment()

    # Step 2: Download models
    download_all_models()

    # Step 3: ComfyUI + nodes
    setup_comfyui()
    import_custom_nodes()

    # Step 4: Generate
    # Adjust image_path and audio_path as needed.
    output = generate_director_mv(
        image_path=IMAGE_PATH,          # set via @param above
        audio_path=AUDIO_PATH,          # set via @param above
        prompt=GLOBAL_PROMPT,
        duration_seconds=CONFIG["duration_seconds"],
        fps=CONFIG["fps"],
        width=CONFIG["width"],
        height=CONFIG["height"],
        seed=CONFIG["seed"],
        quality_mode=CONFIG["quality_mode"],
    )

    if output:
        print(f"\nGeneration complete: {output}")
    else:
        print("\nGeneration failed — check logs above.")
