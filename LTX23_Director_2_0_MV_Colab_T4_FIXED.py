# -*- coding: utf-8 -*-
# =============================================================================
# LTX23_Director_2_0_MV_Colab_T4_FIXED.py  — IMPROVED v2.0
#
# LTX-2.3 Director 2.0 MV — Google Colab T4 Optimized Pipeline
# Implements: LTX-2.3_Director_2.0-MV-Workflow-30s.json
#
# IMPROVEMENTS IN THIS VERSION:
#   [MEM-1]  Enhanced LTXMemoryManager — VRAM bar, tensor registry, per-stage hooks
#   [MEM-2]  ModelCache + LazyLoRARegistry — keep DiT/VAEs alive between chunks
#   [MEM-3]  print_vram_usage(), auto_adjust_settings(), validate_lora_exists()
#   [MEM-4]  Quality-preset ManualSigmas schedules + extended T4 profiles
#   [MEM-5]  aggressive_cleanup() called after EVERY pipeline stage in generate_chunk
#   [MEM-6]  Per-chunk VRAM bar, staged model unloading, OOM pre-check guard
#   [MEM-7]  cleanup_old_cache(), calculate_shot_metrics(), apply_face_restoration()
#   [MEM-8]  VRAM-aware sub-batch VAE decode (8-frame windows, non-blocking GPU sync)
#
# Architecture (unchanged):
#   UnetLoaderGGUF → DualCLIPLoader → LTXVConditioning →
#   LTXDirector → LTXDirectorGuide×2 → LTXVConcatAVLatent×2 →
#   SamplerCustomAdvanced×2 → LTXVSeparateAVLatent×2 →
#   LTXVLatentUpsampler → LTXDirectorCropGuides×2 →
#   VAEDecode + LTXVAudioVAEDecode → chunk-safe assembly → FFmpeg concat
# =============================================================================

# =============================================================================
# SECTION 1 — CONFIGURATION
# @title ⚙️ LTX-2.3 Director 2.0 MV — Settings
# =============================================================================

IMAGE_PATH = "/content/ComfyUI/input/reference.png"  # @param {type:"string"}
AUDIO_PATH = "/content/ComfyUI/input/audio.mp3"      # @param {type:"string"}

DURATION_SECONDS = 31.5   # @param {type:"number"}
FPS              = 24     # @param [8, 12, 16, 24, 25, 30] {type:"raw"}
OUTPUT_WIDTH     = 1280   # @param {type:"integer"}
OUTPUT_HEIGHT    = 720    # @param {type:"integer"}
OUTPUT_FILENAME  = "LTX23_Director_30s.mp4"  # @param {type:"string"}

CUSTOM_PROMPT = ""  # @param {type:"string"}

SEED        = 123456  # @param {type:"integer"}
RANDOM_SEED = False   # @param {type:"boolean"}

# Quality profile: t4_safe | t4_balanced | t4_aggressive | t4_ultra_safe
QUALITY_MODE = "t4_safe"  # @param ["t4_ultra_safe","t4_safe","t4_balanced","t4_aggressive"]

AUTO_CHUNK_SIZE = True   # @param {type:"boolean"}
CHUNK_FRAMES    = 48     # @param {type:"integer"}

LORA_STRENGTH_DISTILLED  = 0.4  # @param {type:"slider",min:0.0,max:2.0,step:0.05}
LORA_STRENGTH_OMNINFT    = 0.6  # @param {type:"slider",min:0.0,max:2.0,step:0.05}
LORA_STRENGTH_TRANSITION = 0.7  # @param {type:"slider",min:0.0,max:2.0,step:0.05}
LORA_STRENGTH_MVCAMERA   = 0.9  # @param {type:"slider",min:0.0,max:2.0,step:0.05}

ENABLE_LORA_DISTILLED  = True   # @param {type:"boolean"}
ENABLE_LORA_OMNINFT    = True   # @param {type:"boolean"}
ENABLE_LORA_TRANSITION = False  # @param {type:"boolean"}
ENABLE_LORA_MVCAMERA   = False  # @param {type:"boolean"}


SAMPLER_STEPS  = 8     # @param {type:"slider",min:1,max:30,step:1}
SAMPLER_CFG    = 1.0   # @param {type:"slider",min:1.0,max:10.0,step:0.5}
SAMPLER_NAME   = "euler"             # @param ["euler","euler_ancestral","dpm_2","heun"]
SCHEDULER_NAME = "linear_quadratic" # @param ["linear_quadratic","karras","exponential","simple"]
IMG_COMPRESSION = 18  # @param {type:"slider",min:1,max:95,step:1}

# [MEM-4] Quality-preset sigma schedules (borrowed from ltx2_ti2v_distilled.py)
# These replace BasicScheduler for more granular VRAM control per pass.
SIGMA_PRESET_MODE = "scheduler"  # @param ["scheduler","manual_preview","manual_balanced","manual_maximum"]

AUTO_REDUCE_CHUNK_ON_OOM = True  # @param {type:"boolean"}
MAX_OOM_RETRIES          = 3     # @param {type:"integer"}
GPU_SAFETY_MARGIN_GB     = 1.5   # @param {type:"slider",min:0.5,max:4.0,step:0.25}
ALLOW_AUTO_DOWNGRADE     = True  # @param {type:"boolean"}

RESUME = True  # @param {type:"boolean"}

PREVIEW_MODE     = False  # @param {type:"boolean"}
PREVIEW_DURATION = 3      # @param {type:"integer"}
PREVIEW_WIDTH    = 832    # @param {type:"integer"}
PREVIEW_HEIGHT   = 480    # @param {type:"integer"}

ENABLE_MEMORY_LOGGING = True   # @param {type:"boolean"}
CLEANUP_AFTER_CHUNK   = True   # @param {type:"boolean"}
CLEANUP_AFTER_STAGE   = True   # @param {type:"boolean"}
KEEP_TEMP_CHUNKS      = False  # @param {type:"boolean"}
CLEANUP_TEMP_FILES    = True   # @param {type:"boolean"}

# [MEM-2] Model cache: keep DiT + VAEs alive between chunks (saves ~6s per chunk reload)
USE_MODEL_CACHE = True   # @param {type:"boolean"}
# [MEM-7] Auto-delete workspace cache files older than N days
CACHE_MAX_AGE_DAYS = 7   # @param {type:"integer"}
# [MEM-7] Face restoration post-processing pass
FACE_RESTORATION = False  # @param {type:"boolean"}

WORKSPACE_DIR = "/content/ltx23_workspace"  # @param {type:"string"}
OUTPUT_DIR    = "/content/ltx23_output"     # @param {type:"string"}
COMFYUI_DIR   = "/content/ComfyUI"          # @param {type:"string"}


# ── Resolve CONFIG ─────────────────────────────────────────────────────────────
import random as _random

IMAGE_PATH = IMAGE_PATH.strip() or None
AUDIO_PATH = AUDIO_PATH.strip() or None
_CUSTOM_PROMPT = CUSTOM_PROMPT.strip() or None

if RANDOM_SEED:
    SEED = _random.randint(0, 2**31 - 1)
    print(f"  🎲 Random seed: {SEED}")

_LORA_STRENGTHS_OVERRIDE = {
    "lora_distilled":  LORA_STRENGTH_DISTILLED,
    "lora_omninft":    LORA_STRENGTH_OMNINFT,
    "lora_transition": LORA_STRENGTH_TRANSITION,
    "lora_mvcamera":   LORA_STRENGTH_MVCAMERA,
}
_LORA_ENABLED_OVERRIDE = {
    "lora_distilled":  ENABLE_LORA_DISTILLED,
    "lora_omninft":    ENABLE_LORA_OMNINFT,
    "lora_transition": ENABLE_LORA_TRANSITION,
    "lora_mvcamera":   ENABLE_LORA_MVCAMERA,
}
_SAMPLER_OVERRIDE   = SAMPLER_NAME
_SCHEDULER_OVERRIDE = SCHEDULER_NAME
_STEPS_OVERRIDE     = SAMPLER_STEPS
_CFG_OVERRIDE       = SAMPLER_CFG
_IMG_COMPRESSION_OVERRIDE = IMG_COMPRESSION

CONFIG = {
    "duration_seconds":  DURATION_SECONDS,
    "fps":               FPS,
    "width":             OUTPUT_WIDTH,
    "height":            OUTPUT_HEIGHT,
    "seed":              SEED,
    "quality_mode":      QUALITY_MODE,
    "auto_chunk_size":   AUTO_CHUNK_SIZE,
    "chunk_frames":      CHUNK_FRAMES,
    "auto_reduce_chunk_on_oom": AUTO_REDUCE_CHUNK_ON_OOM,
    "max_oom_retries":          MAX_OOM_RETRIES,
    "resume":            RESUME,
    "gpu_safety_margin_gb":    GPU_SAFETY_MARGIN_GB,
    "enable_memory_logging":   ENABLE_MEMORY_LOGGING,
    "cleanup_after_chunk":     CLEANUP_AFTER_CHUNK,
    "cleanup_after_stage":     CLEANUP_AFTER_STAGE,
    "keep_temp_chunks":        KEEP_TEMP_CHUNKS,
    "cleanup_temp_files":      CLEANUP_TEMP_FILES,
    "use_model_cache":         USE_MODEL_CACHE,
    "cache_max_age_days":      CACHE_MAX_AGE_DAYS,
    "face_restoration":        FACE_RESTORATION,
    "preview_mode":      PREVIEW_MODE,
    "preview_duration":  PREVIEW_DURATION,
    "preview_width":     PREVIEW_WIDTH,
    "preview_height":    PREVIEW_HEIGHT,
    "allow_auto_downgrade": ALLOW_AUTO_DOWNGRADE,
    "workspace_dir":     WORKSPACE_DIR,
    "output_dir":        OUTPUT_DIR,
    "output_filename":   OUTPUT_FILENAME,
    "comfyui_dir":       COMFYUI_DIR,
    "sigma_preset_mode": SIGMA_PRESET_MODE,
}


# ── Model filenames ───────────────────────────────────────────────────────────
MODELS = {
    "dit":            "ltx-2-3-22b-dev-Q4_K_M.gguf",
    "text_encoder_1": "gemma_3_12B_it_fp4_mixed.safetensors",
    "text_encoder_2": "ltx-2.3_text_projection_bf16.safetensors",
    "audio_vae":      "LTX23_audio_vae_bf16.safetensors",
    "video_vae":      "LTX23_video_vae_bf16.safetensors",
    "tiny_vae":       "taeltx2_3.safetensors",
    "upscaler":       "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    "lora_distilled": "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors",
    "lora_omninft":   "LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors",
    "lora_transition":"ltx2.3-transition.safetensors",
    "lora_mvcamera":  "LTX2.3-MVCamera-drclips.safetensors",
}

LORA_STRENGTHS = {
    "lora_distilled":  _LORA_STRENGTHS_OVERRIDE.get("lora_distilled",  0.4),
    "lora_omninft":    _LORA_STRENGTHS_OVERRIDE.get("lora_omninft",    0.6),
    "lora_transition": _LORA_STRENGTHS_OVERRIDE.get("lora_transition", 0.7),
    "lora_mvcamera":   _LORA_STRENGTHS_OVERRIDE.get("lora_mvcamera",   0.9),
}

LORA_ENABLED = {
    "lora_distilled":  _LORA_ENABLED_OVERRIDE.get("lora_distilled",  True),
    "lora_omninft":    _LORA_ENABLED_OVERRIDE.get("lora_omninft",    True),
    "lora_transition": _LORA_ENABLED_OVERRIDE.get("lora_transition", False),
    "lora_mvcamera":   _LORA_ENABLED_OVERRIDE.get("lora_mvcamera",   False),
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


# ── [MEM-4] Extended T4 profiles + ultra-safe tier ───────────────────────────
T4_PROFILES = {
    # [NEW] Ultra-safe: 32-frame chunks, 704×400, no Director, no LoRAs
    "t4_ultra_safe": {
        "chunk_frames":      33,      # 33 = 8*4+1, smallest safe multi-second chunk
        "generation_width":  704,
        "generation_height": 400,
        "offload_models":    True,
        "skip_director":     True,
        "img_compression":   50,
        "longer_edge":       720,
        "description":       "Ultra-safe: 33-frame chunks 704×400, LoRAs off, no Director",
        "disable_all_loras": True,    # force-disable every LoRA
    },
    "t4_safe": {
        "chunk_frames":      49,      # 49 = 8*6+1
        "generation_width":  832,
        "generation_height": 480,
        "offload_models":    True,
        "skip_director":     True,
        "img_compression":   33,
        "longer_edge":       848,
        "description":       "Conservative: 49-frame chunks, 832×480, no CLIP",
        "disable_all_loras": False,
    },
    "t4_balanced": {
        "chunk_frames":      73,      # 73 = 8*9+1
        "generation_width":  1280,
        "generation_height": 720,
        "offload_models":    True,
        "skip_director":     False,
        "img_compression":   18,
        "longer_edge":       1312,
        "description":       "Balanced: 73-frame chunks, 1280×720, moderate offloading",
        "disable_all_loras": False,
    },
    "t4_aggressive": {
        "chunk_frames":      97,      # 97 = 8*12+1
        "generation_width":  1280,
        "generation_height": 720,
        "offload_models":    False,
        "skip_director":     False,
        "img_compression":   18,
        "longer_edge":       1312,
        "description":       "Aggressive: 97-frame chunks, 1280×720 (OOM risk)",
        "disable_all_loras": False,
    },
}

# ── [MEM-4] Manual sigma presets (ManualSigmas node, from ltx2_ti2v_distilled) ─
SIGMA_PRESETS = {
    "manual_preview": {
        "pass1": "1.0, 0.95, 0.80, 0.50, 0.20, 0.0",
        "pass2": "0.90, 0.60, 0.20, 0.0",
        "description": "6/4 steps — fastest, lower detail",
    },
    "manual_balanced": {
        "pass1": "1.0, 0.99, 0.98, 0.95, 0.90, 0.85, 0.70, 0.50, 0.25, 0.0",
        "pass2": "0.95, 0.85, 0.60, 0.30, 0.0",
        "description": "10/5 steps — balanced quality/speed",
    },
    "manual_maximum": {
        "pass1": "1.0, 0.995, 0.99, 0.98, 0.97, 0.95, 0.90, 0.85, 0.75, 0.60, 0.40, 0.20, 0.05, 0.0",
        "pass2": "0.98, 0.95, 0.88, 0.75, 0.55, 0.35, 0.15, 0.0",
        "description": "14/8 steps — maximum quality",
    },
}

# ── Workflow constants ─────────────────────────────────────────────────────────
WORKFLOW_FPS             = CONFIG["fps"]
WORKFLOW_CFG             = _CFG_OVERRIDE
WORKFLOW_SAMPLER_PASS1   = _SAMPLER_OVERRIDE
WORKFLOW_SCHEDULER       = _SCHEDULER_OVERRIDE
WORKFLOW_STEPS           = _STEPS_OVERRIDE
WORKFLOW_STEPS_PASS2     = 4
WORKFLOW_DENOISE_PASS2   = 0.42
WORKFLOW_IMG_COMPRESSION = _IMG_COMPRESSION_OVERRIDE

GLOBAL_PROMPT = _CUSTOM_PROMPT if _CUSTOM_PROMPT else (
    "Create a highly realistic cinematic AI music video using the provided reference image. "
    "Preserve the person's identity, facial structure, hairstyle, skin tone, clothing, body "
    "proportions, and overall appearance exactly as in the reference image. The singer must "
    "remain fully recognizable throughout the entire video with absolutely no identity drift.\n\n"
    "The person is performing directly to the camera as a world-class pop, hip-hop and rap singer "
    "during a sold-out stadium concert. Generate perfectly synchronized lip movements.\n\n"
    "drclipz, Aggressive cinematic music video camera. Fast push-in, fast pull-back, energetic "
    "handheld movement, rhythmic tracking shots, dynamic low-angle hero shots.\n\n"
    "Premium concert lighting with cinematic key light, colorful neon rim lights, volumetric "
    "atmosphere, dramatic contrast, realistic skin tones, vibrant electronic music video mood.\n\n"
    "Photorealistic, blockbuster-quality AI music video, ultra-high facial fidelity."
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
import threading
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

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

# Force OS to reclaim freed C-heap memory (prevents silent Colab RAM kills)
try:
    import ctypes
    _LIBC = ctypes.CDLL("libc.so.6")
except Exception:
    _LIBC = None

def _malloc_trim():
    if _LIBC is not None:
        try:
            _LIBC.malloc_trim(0)
        except Exception:
            pass

def detect_gpu() -> Dict:
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
    free, _ = torch.cuda.mem_get_info(0)
    info["vram_free_gb"] = free / (1024 ** 3)
    return info

_GPU_INFO = detect_gpu()
print(f"PyTorch : {_GPU_INFO['torch_version']}  |  CUDA : {_GPU_INFO['cuda_version']}")
print(f"GPU     : {_GPU_INFO['device_name']}")
print(f"VRAM    : {_GPU_INFO['vram_total_gb']:.1f} GB total  /  {_GPU_INFO['vram_free_gb']:.1f} GB free")

if not _GPU_INFO["available"]:
    raise RuntimeError(
        "\nERROR: No CUDA GPU detected.\n"
        "Runtime → Change runtime type → T4 GPU"
    )

DEVICE = torch.device("cuda")


# =============================================================================
# SECTION 3 — ENHANCED MEMORY MANAGER  [MEM-1]
# =============================================================================

class LTXMemoryManager:
    """
    [MEM-1] Enhanced VRAM/RAM manager for T4 inference.

    New in v2.0:
      - print_vram_bar()  : visual █░ VRAM bar (from ltx2_ti2v_distilled)
      - _tensor_registry  : weak-ref tracking of live GPU tensors
      - stage_cleanup()   : lightweight per-stage hook (replaces ad-hoc soft_cleanup calls)
      - pre_op_guard()    : VRAM check before each expensive op; triggers aggressive cleanup
      - ram_status()      : one-line CPU RAM status string
    """

    def __init__(self, safety_margin_gb: float = 1.5, enable_logging: bool = True):
        self.safety_margin_gb = safety_margin_gb
        self.enable_logging   = enable_logging
        self._peak_allocated  = 0.0
        self._chunk_info: Dict = {}
        self._stage_timers: Dict[str, float] = {}
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

    def is_ram_safe(self, required_gb: float = 2.0) -> bool:
        return self.cpu_available_gb() > required_gb

    # ── [MEM-1] Visual VRAM bar (from ltx2_ti2v_distilled) ───────────────────
    def print_vram_bar(self, prefix: str = "   "):
        """Print a visual ░█ VRAM usage bar."""
        used  = self.gpu_allocated_gb()
        total = self.gpu_total_gb()
        free  = self.gpu_free_gb()
        pct   = used / total if total > 0 else 0
        filled = int(24 * pct)
        bar   = "█" * filled + "░" * (24 - filled)
        safety_marker = "⚠" if free < self.safety_margin_gb else "✓"
        print(f"{prefix}💾 VRAM [{bar}] {used:.1f}/{total:.1f} GB "
              f"({pct*100:.1f}%)  free={free:.2f} GB {safety_marker}")

    def ram_status(self) -> str:
        avail = self.cpu_available_gb()
        used  = self.cpu_used_gb()
        return f"RAM  used={used:.2f} GB  avail={avail:.2f} GB"

    # ── Cleanup tiers ─────────────────────────────────────────────────────────
    def soft_cleanup(self):
        gc.collect()
        torch.cuda.empty_cache()

    def cleanup(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def aggressive_cleanup(self):
        """Full sweep: 3× GC, cache, IPC, peak reset, OS malloc_trim."""
        for _ in range(3):
            gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        _malloc_trim()

    # ── [MEM-5] Per-stage cleanup hook ───────────────────────────────────────
    def stage_cleanup(self, stage_name: str = ""):
        """
        [MEM-5] Lightweight per-stage hook called after every pipeline stage.
        Runs soft_cleanup always; escalates to aggressive if VRAM is low.
        """
        self.soft_cleanup()
        if not self.is_vram_safe():
            if self.enable_logging:
                print(f"  [mem] VRAM low after '{stage_name}' — running aggressive cleanup")
            self.aggressive_cleanup()
        elif self.enable_logging and stage_name:
            pass  # quiet unless low

    def ram_cleanup(self):
        """System RAM sweep: 3× GC + malloc_trim."""
        for _ in range(3):
            gc.collect()
        _malloc_trim()
        if self.enable_logging:
            print(f"  [mem] RAM cleanup → {self.ram_status()}")

    # ── [MEM-6] Pre-operation VRAM guard ─────────────────────────────────────
    def pre_op_guard(self, op_name: str, required_gb: float = 0.5) -> bool:
        """
        [MEM-6] Check VRAM before a costly operation. Triggers aggressive
        cleanup if below threshold. Returns True if safe to proceed.
        """
        free = self.gpu_free_gb()
        if free < required_gb + self.safety_margin_gb:
            print(f"  [guard] Low VRAM ({free:.2f} GB) before '{op_name}' "
                  f"(need ≥{required_gb + self.safety_margin_gb:.2f} GB) — cleaning up")
            self.aggressive_cleanup()
            free = self.gpu_free_gb()
        ok = free >= required_gb
        if not ok:
            print(f"  [guard] ⚠ Still only {free:.2f} GB free before '{op_name}' — proceeding anyway")
        return ok

    # ── Object release ────────────────────────────────────────────────────────
    def release_tensor(self, tensor, name: str = "tensor"):
        if tensor is not None:
            del tensor
        self.soft_cleanup()

    def release_model(self, model, name: str = "model"):
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
        if self.enable_logging:
            print(f"  [mem] Unloading {name}  (free GPU before: {self.gpu_free_gb():.2f} GB)")
        self.release_model(model, name)
        if self.enable_logging:
            print(f"  [mem] Unloaded  {name}  (free GPU after : {self.gpu_free_gb():.2f} GB)")

    # ── Reporting ─────────────────────────────────────────────────────────────
    def memory_report(self, prefix: str = "") -> str:
        lines = [
            f"{prefix}GPU  alloc={self.gpu_allocated_gb():.2f} GB  "
            f"reserv={self.gpu_reserved_gb():.2f} GB  "
            f"free={self.gpu_free_gb():.2f} GB  "
            f"peak={self.gpu_peak_gb():.2f} GB",
            f"{prefix}{self.ram_status()}",
        ]
        if self._chunk_info:
            lines.append(f"{prefix}Chunk {self._chunk_info.get('index','?')}  "
                         f"{self._chunk_info.get('frames','?')} frames  "
                         f"{self._chunk_info.get('resolution','?')}")
        return "\n".join(lines)

    def print_memory(self, prefix: str = ""):
        print(self.memory_report(prefix))

    def set_chunk_info(self, index: int, frames: int, w: int, h: int):
        self._chunk_info = {"index": index, "frames": frames, "resolution": f"{w}×{h}"}

    def warn_if_low(self):
        free = self.gpu_free_gb()
        if free < self.safety_margin_gb:
            print(f"  ⚠ VRAM below safety threshold ({free:.2f} GB < {self.safety_margin_gb:.2f} GB)")
            self.aggressive_cleanup()

    def estimate_frame_ram_gb(self, num_frames: int, height: int, width: int) -> float:
        return num_frames * height * width * 3 * 4 / (1024 ** 3)

# Singleton instance
mem = LTXMemoryManager(
    safety_margin_gb=CONFIG["gpu_safety_margin_gb"],
    enable_logging=CONFIG["enable_memory_logging"],
)


# =============================================================================
# SECTION 3b — MODEL CACHE & LAZY LORA REGISTRY  [MEM-2]
# =============================================================================

class _ModelCache:
    """
    [MEM-2] Keep DiT + VAEs resident between chunks to avoid ~6s reload overhead.
    Thread-safe via a simple Lock. Evict with evict_all() before OOM recovery.
    Borrowed and extended from ltx2_ti2v_distilled.py ModelCache.
    """
    def __init__(self):
        self._unet      = None
        self._video_vae = None
        self._audio_vae = None
        self._upscaler  = None
        self._lock      = threading.Lock()

    def get_unet(self, loader_fn, force_reload: bool = False):
        with self._lock:
            if self._unet is None or force_reload:
                print("   📦 [ModelCache] Loading UNet (DiT)...")
                self._unet = loader_fn()
            return self._unet

    def get_video_vae(self, loader_fn, force_reload: bool = False):
        with self._lock:
            if self._video_vae is None or force_reload:
                self._video_vae = loader_fn()
            return self._video_vae

    def get_audio_vae(self, loader_fn, force_reload: bool = False):
        with self._lock:
            if self._audio_vae is None or force_reload:
                self._audio_vae = loader_fn()
            return self._audio_vae

    def get_upscaler(self, loader_fn, force_reload: bool = False):
        with self._lock:
            if self._upscaler is None or force_reload:
                self._upscaler = loader_fn()
            return self._upscaler

    def evict_unet(self):
        with self._lock:
            if self._unet is not None:
                del self._unet
                self._unet = None
        mem.cleanup()
        print("   🗑️ [ModelCache] UNet evicted.")

    def evict_all(self):
        with self._lock:
            self._unet = self._video_vae = self._audio_vae = self._upscaler = None
        mem.aggressive_cleanup()
        print("   🗑️ [ModelCache] All models evicted.")

    def is_unet_loaded(self) -> bool:
        return self._unet is not None


MODEL_CACHE = _ModelCache() if CONFIG["use_model_cache"] else None


class _LazyLoRARegistry:
    """
    [MEM-2] Load LoRA weights on first request; cache for subsequent chunks.
    From ltx2_ti2v_distilled.py LazyLoRARegistry.
    """
    def __init__(self):
        self._loaded: Dict[str, Any] = {}

    def get(self, lora_name: str, loader_fn) -> Optional[Any]:
        if lora_name not in self._loaded:
            result = loader_fn(lora_name)
            if result is not None:
                self._loaded[lora_name] = result
                print(f"   ✓ [LazyLoRA] Loaded: {lora_name}")
        return self._loaded.get(lora_name)

    def clear(self):
        self._loaded.clear()

LORA_REGISTRY = _LazyLoRARegistry()


# =============================================================================
# SECTION 3c — UTILITY HELPERS  [MEM-3] [MEM-7]
# =============================================================================

# ── [MEM-3] VRAM helper (short alias used throughout pipeline) ────────────────
def print_vram_usage(prefix: str = "   "):
    """[MEM-3] One-liner VRAM bar — alias to mem.print_vram_bar()."""
    mem.print_vram_bar(prefix)


# ── [MEM-3] Auto-adjust quality from available VRAM ─────────────────────────
def auto_adjust_settings() -> Dict[str, Any]:
    """
    [MEM-3] Inspect available VRAM and return recommended CONFIG overrides.
    Matches the same function in ltx2_ti2v_distilled.py.
    """
    vram = mem.gpu_total_gb()
    if vram == 0:
        print("⚠️  No CUDA GPU detected.")
        return {}
    if vram < 12:
        print(f"⚠️  Low VRAM ({vram:.1f} GB) → switching to t4_ultra_safe, disabling LoRAs.")
        return {"quality_mode": "t4_ultra_safe", "use_model_cache": False}
    elif vram < 15:
        print(f"ℹ️  T4 VRAM ({vram:.1f} GB) → t4_safe profile.")
        return {"quality_mode": "t4_safe"}
    elif vram < 24:
        print(f"ℹ️  Moderate VRAM ({vram:.1f} GB) → t4_balanced profile.")
        return {"quality_mode": "t4_balanced"}
    else:
        print(f"✅  Ample VRAM ({vram:.1f} GB) → t4_aggressive available.")
        return {"quality_mode": "t4_aggressive"}


# ── [MEM-3] Validated LoRA path resolution ───────────────────────────────────
def validate_lora_exists(lora_name: str, lora_type: str = "lora") -> Optional[str]:
    """
    [MEM-3] Fully validated LoRA path resolution — searches ComfyUI folder_paths,
    then falls back to absolute path. From ltx2_ti2v_distilled.py.
    Returns full path or None.
    """
    if not lora_name:
        return None
    try:
        import folder_paths
        for base in folder_paths.get_folder_paths("loras"):
            direct = os.path.join(base, lora_name)
            if os.path.exists(direct):
                return direct
            for root, _, files in os.walk(base):
                if lora_name in files:
                    return os.path.join(root, lora_name)
    except Exception as e:
        print(f"⚠️  folder_paths error: {e}")
    fallback = os.path.join("/content/ComfyUI/models/loras", lora_name)
    if os.path.exists(fallback):
        return fallback
    print(f"⚠️  {lora_type.upper()} LoRA not found: {lora_name}")
    return None


# ── [MEM-7] Stale cache cleanup ──────────────────────────────────────────────
def cleanup_old_cache(cache_dir: str, max_age_days: int = 7) -> None:
    """[MEM-7] Remove files in cache_dir older than max_age_days."""
    if not os.path.isdir(cache_dir):
        return
    cutoff = time.time() - max_age_days * 86400
    removed = 0
    for fname in os.listdir(cache_dir):
        fp = os.path.join(cache_dir, fname)
        if os.path.isfile(fp) and os.path.getmtime(fp) < cutoff:
            try:
                os.remove(fp)
                removed += 1
            except Exception:
                pass
    if removed:
        print(f"🗑️  Removed {removed} stale cache file(s) from {cache_dir}")


# ── [MEM-7] Shot quality metrics ─────────────────────────────────────────────
def calculate_shot_metrics(video_path: str) -> Tuple[float, dict]:
    """
    [MEM-7] Compute sharpness, brightness and motion smoothness for a clip.
    From ltx2_ti2v_distilled.py — used to score chunk quality.
    """
    try:
        cap   = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step  = max(1, total // 10)
        sharp_v, bright_v, motion_v = [], [], []
        prev_gray = None
        for fi in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharp_v.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            bright_v.append(float(cv2.mean(gray)[0]))
            if prev_gray is not None:
                motion_v.append(float(np.mean(cv2.absdiff(prev_gray, gray))))
            prev_gray = gray
        cap.release()
        metrics = {
            "sharpness":  float(np.mean(sharp_v))  if sharp_v  else 0.0,
            "brightness": float(np.mean(bright_v)) if bright_v else 0.0,
            "motion_std": float(np.std(motion_v))  if motion_v else 0.0,
        }
        overall = (min(1.0, metrics["sharpness"] / 1000)
                   + min(1.0, metrics["brightness"] / 200)
                   + max(0.0, 1.0 - metrics["motion_std"] / 50)) / 3.0
        return overall, metrics
    except Exception as e:
        print(f"   ⚠️ Metrics error: {e}")
        return 0.0, {}


# ── [MEM-7] Face restoration ─────────────────────────────────────────────────
def apply_face_restoration(video_path: str) -> str:
    """
    [MEM-7] Per-frame face crop-and-sharpen using YOLO v8 face detection.
    Skipped if ultralytics is missing. From ltx2_ti2v_distilled.py.
    """
    if not CONFIG.get("face_restoration", False):
        return video_path
    print("   ✨ Running Face Restoration Pass...")
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n-face.pt")
    except ImportError:
        print("   ⚠️ Ultralytics not found — skipping face restoration.")
        return video_path
    except Exception:
        print("   ⚠️ Face model unavailable — skipping.")
        return video_path

    cap = cv2.VideoCapture(video_path)
    fps_v = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = video_path.replace(".mp4", "_facefix.mp4")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_v, (w, h))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue
                frame[y1:y2, x1:x2] = cv2.detailEnhance(face, sigma_s=10, sigma_r=0.15)
        out.write(frame)
    cap.release()
    out.release()
    if os.path.exists(out_path):
        os.replace(out_path, video_path)
    return video_path


# =============================================================================
# SECTION 4 — ENVIRONMENT INSTALLATION
# =============================================================================

def _run(cmd: str, label: str):
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"  ✓ {label}")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ {label}: {e.stderr.strip()[:200]}")

def _install_custom_nodes():
    nodes_dir = f"{CONFIG['comfyui_dir']}/custom_nodes"
    Path(nodes_dir).mkdir(parents=True, exist_ok=True)
    REQUIRED_NODES = [
        ("https://github.com/kijai/ComfyUI-KJNodes",               "ComfyUI-KJNodes"),
        ("https://github.com/city96/ComfyUI-GGUF",                 "ComfyUI-GGUF"),
        ("https://github.com/Lightricks/ComfyUI-LTXVideo",         "ComfyUI-LTXVideo"),
        ("https://github.com/WhatDreamscost/WhatDreamsCost-ComfyUI","WhatDreamsCost-ComfyUI"),
        ("https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite", "ComfyUI-VideoHelperSuite"),
        ("https://github.com/kijai/ComfyUI-MelBandRoFormer",        "ComfyUI-MelBandRoFormer"),
        ("https://github.com/rgthree/rgthree-comfy",                "rgthree-comfy"),
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

def install_environment():
    print("=" * 60)
    print("[1/5] Installing Python packages...")
    _run("pip install -q torch torchvision torchaudio", "torch")
    _run("pip install -q torchsde einops diffusers accelerate", "diffusers stack")
    _run("pip install -q av spandrel albumentations onnx opencv-python onnxruntime", "vision stack")
    _run("pip install -q psutil nest_asyncio", "utilities")
    print("\n[2/5] Installing system tools...")
    _run("apt-get -y install -qq aria2 ffmpeg", "aria2 + ffmpeg")
    print("\n[3/5] Cloning ComfyUI...")
    cdir = CONFIG["comfyui_dir"]
    if not os.path.exists(cdir):
        _run(f"git clone -q https://github.com/comfyanonymous/ComfyUI {cdir}", "ComfyUI")
    else:
        print("  ComfyUI already present.")
    _run(f"pip install -q -r {cdir}/requirements.txt", "ComfyUI requirements")
    print("\n[4/5] Installing custom nodes...")
    _install_custom_nodes()
    print("\n[5/5] Creating workspace directories...")
    for sub in ["chunks", "frames", "audio", "final", "logs", "cache"]:
        Path(f"{CONFIG['workspace_dir']}/{sub}").mkdir(parents=True, exist_ok=True)
    Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(f"{cdir}/input").mkdir(parents=True, exist_ok=True)
    print("\n✓ Environment setup complete.")


# =============================================================================
# SECTION 5 — COMFYUI SETUP & CUSTOM NODE LOADING
# =============================================================================

_NODES_LOADED = False

def setup_comfyui():
    cdir = CONFIG["comfyui_dir"]
    if cdir not in sys.path:
        sys.path.insert(0, cdir)
    print(f"  ComfyUI path: {cdir}")

def import_custom_nodes():
    global _NODES_LOADED
    if _NODES_LOADED:
        return
    import nest_asyncio
    nest_asyncio.apply()
    try:
        from aiohttp import web  # noqa
        from server import PromptServer
        if not hasattr(PromptServer, "instance") or PromptServer.instance is None:
            PromptServer.instance = PromptServer(asyncio.new_event_loop())
    except Exception:
        pass
    try:
        import kornia.geometry.transform.pyramid as _kpyr
        if not hasattr(_kpyr, "pad"):
            import torch.nn.functional as F
            _kpyr.pad = F.pad
    except Exception:
        pass
    from nodes import init_builtin_extra_nodes, init_external_custom_nodes
    async def _loader():
        failed = await init_builtin_extra_nodes()
        await init_external_custom_nodes()
        if failed:
            print("WARNING: some comfy_extras nodes failed:")
            for n in failed:
                print(f"  - {n}")
    try:
        asyncio.run(_loader())
    except RuntimeError:
        asyncio.get_event_loop().run_until_complete(_loader())
    _NODES_LOADED = True
    print("  ✓ Custom nodes loaded.")

def get_node(name: str):
    from nodes import NODE_CLASS_MAPPINGS
    if name not in NODE_CLASS_MAPPINGS:
        raise KeyError(f"ComfyUI node '{name}' not found in NODE_CLASS_MAPPINGS.")
    return NODE_CLASS_MAPPINGS[name]()

def get_node_cls(name: str):
    from nodes import NODE_CLASS_MAPPINGS
    if name not in NODE_CLASS_MAPPINGS:
        raise KeyError(f"ComfyUI node class '{name}' not found.")
    return NODE_CLASS_MAPPINGS[name]

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]

# =============================================================================
# SECTION 6 — CUSTOM NODE DEPENDENCY REPORT
# =============================================================================

REQUIRED_NODE_NAMES = {
    "Core ComfyUI": [
        "KSamplerSelect","SamplerCustomAdvanced","CFGGuider","RandomNoise",
        "BasicScheduler","ConditioningZeroOut","VAELoader","VAEDecode",
        "DualCLIPLoader","CLIPTextEncode","EmptyLTXVLatentVideo",
        "LTXVConditioning","LTXVImgToVideoInplace","LTXVConcatAVLatent",
        "LTXVSeparateAVLatent","LTXVLatentUpsampler","LTXVCropGuides",
        "LTXVEmptyLatentAudio","LTXVAudioVAEDecode","CreateVideo",
        "LatentUpscaleModelLoader","ResizeImageMaskNode",
        "LTXVPreprocess","ResizeImagesByLongerEdge",
    ],
    "ComfyUI-GGUF":         ["UnetLoaderGGUF"],
    "ComfyUI-KJNodes":      ["VAELoaderKJ","ModelPreviewOverrideKJ"],
    "WhatDreamsCost-ComfyUI":["LTXDirector","LTXDirectorGuide","LTXDirectorCropGuides"],
    "ComfyUI-VideoHelperSuite":["VHS_VideoCombine"],
}

def validate_custom_nodes() -> bool:
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
            print(f"  ✗ {provider}: MISSING → {', '.join(missing)}")
            all_ok = False
        else:
            print(f"  ✓ {provider}")
    print("  " + "-" * 50)
    return all_ok


# =============================================================================
# SECTION 7 — MODEL DOWNLOAD
# =============================================================================

def model_download(url: str, dest_dir: str, filename: str = None) -> Optional[str]:
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = url.split("/")[-1].split("?")[0]
    fp = os.path.join(dest_dir, filename)
    if os.path.exists(fp) and os.path.getsize(fp) > 0:
        print(f"  ✓ {filename} (cached)")
        return filename
    print(f"  ↓ {filename}...", end=" ", flush=True)
    cmd = ["aria2c","--console-log-level=error","-c","-x","16","-s","16","-k","1M",
           "--summary-interval=0","--quiet","-d",dest_dir,"-o",filename,url]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("done")
        return filename
    except subprocess.CalledProcessError as e:
        print(f"FAILED\n  Error: {e.stderr.strip()[:200]}")
        return None

def download_all_models(skip_loras: bool = False):
    print("\n  Downloading models...")
    for key, url in DOWNLOAD_URLS.items():
        if skip_loras and key.startswith("lora_"):
            continue
        model_download(url, MODEL_DEST_DIRS[key], MODELS[key])

def validate_models() -> bool:
    ok = True
    print("\n  Model file validation:")
    for key, fname in MODELS.items():
        fp = os.path.join(MODEL_DEST_DIRS[key], fname)
        exists = os.path.exists(fp) and os.path.getsize(fp) > 0
        print(f"  {'✓' if exists else '✗ MISSING':10s} {fname}")
        if not exists:
            ok = False
    return ok

# =============================================================================
# SECTION 8 — PRE-GENERATION VALIDATION SUITE
# =============================================================================

def validate_environment() -> bool:
    ok = True
    if not torch.cuda.is_available():
        print("  ✗ CUDA not available"); ok = False
    else:
        print(f"  ✓ CUDA {torch.version.cuda} on {torch.cuda.get_device_name(0)}")
    maj, min_ = sys.version_info[:2]
    if maj < 3 or (maj == 3 and min_ < 9):
        print(f"  ✗ Python {maj}.{min_} — need 3.9+"); ok = False
    else:
        print(f"  ✓ Python {maj}.{min_}")
    return ok

def validate_workflow_dependencies() -> bool:
    try:
        import nodes  # noqa
        return True
    except ImportError as e:
        print(f"  ✗ Cannot import ComfyUI nodes: {e}")
        return False

def validate_input_image(path: Optional[str]) -> bool:
    if path is None:
        return True
    if not os.path.exists(path):
        print(f"  ✗ Input image not found: {path}"); return False
    try:
        img = Image.open(path); img.verify()
        print(f"  ✓ Input image: {path}"); return True
    except Exception as e:
        print(f"  ✗ Input image invalid: {e}"); return False

def validate_audio(path: Optional[str]) -> bool:
    if path is None:
        return True
    if not os.path.exists(path):
        print(f"  ✗ Audio not found: {path}"); return False
    size_mb = os.path.getsize(path) / (1024*1024)
    print(f"  ✓ Audio: {path} ({size_mb:.1f} MB)")
    return True

def validate_resolution(w: int, h: int) -> bool:
    safe = (w <= 1280 and h <= 720) or (w <= 720 and h <= 1280)
    print(f"  {'✓' if safe else '⚠'} Resolution {w}×{h}")
    return True

def validate_frame_count(n: int) -> bool:
    valid = _is_valid_ltx_frame_count(n)
    print(f"  {'✓' if valid else '⚠ (will adjust)'} Frame count {n}")
    return True

def validate_gpu_memory(required_gb: float = 8.0) -> bool:
    free = mem.gpu_free_gb()
    ok = free >= required_gb
    print(f"  {'✓' if ok else '✗'} GPU free: {free:.2f} GB (need ≥ {required_gb:.1f} GB)")
    return ok

def run_all_validations(image_path=None, audio_path=None, w=None, h=None, n_frames=None) -> bool:
    print("\n" + "=" * 60)
    print("PRE-GENERATION VALIDATION")
    print("=" * 60)
    w = w or CONFIG["width"]; h = h or CONFIG["height"]
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
    print("\n" + ("✓ All validations passed." if passed else "✗ Some validations failed."))
    return passed


# =============================================================================
# SECTION 9 — LTX TEMPORAL CONSTRAINTS & FRAME MATH
# =============================================================================

def _is_valid_ltx_frame_count(n: int, min_frames: int = 9) -> bool:
    if n < min_frames:
        return False
    return (n - 1) % 8 == 0

def normalize_ltx_frame_count(requested: int, fps: int = 24, min_frames: int = 9) -> int:
    if _is_valid_ltx_frame_count(requested, min_frames):
        return requested
    k = math.ceil((requested - 1) / 8)
    adjusted = k * 8 + 1
    print(f"  LTX frame adjustment: {requested} → {adjusted} ({adjusted/fps:.2f}s)")
    return adjusted

def calculate_timeline(duration_s: float, fps: int) -> Tuple[int, float]:
    raw_frames   = round(duration_s * fps)
    valid_frames = normalize_ltx_frame_count(raw_frames, fps)
    return valid_frames, valid_frames / fps

def get_chunk_seed(global_seed: int, chunk_index: int) -> int:
    return (global_seed + chunk_index * 1000003) & 0x7FFFFFFF

def plan_chunks(total_frames: int, chunk_size: int, fps: int) -> List[Dict]:
    chunks = []; start = 0; idx = 0
    while start < total_frames:
        remaining = total_frames - start
        raw_size  = min(chunk_size, remaining)
        valid_size = normalize_ltx_frame_count(raw_size, fps)
        if start + valid_size > total_frames:
            valid_size = total_frames - start
            if valid_size < 9:
                if chunks:
                    chunks[-1]["num_frames"] += valid_size
                break
        chunks.append({"chunk_index": idx,"start_frame": start,
                        "num_frames": valid_size,"fps": fps,"path": None})
        idx += 1; start += valid_size
    return chunks

def estimate_chunk_size(w: int, h: int, fps: int, mode: str = "t4_safe") -> int:
    profile = T4_PROFILES.get(mode, T4_PROFILES["t4_safe"])
    if not CONFIG["auto_chunk_size"]:
        return normalize_ltx_frame_count(profile["chunk_frames"])
    free_gb    = max(1.0, mem.gpu_free_gb() - CONFIG["gpu_safety_margin_gb"])
    free_bytes = free_gb * (1024 ** 3)
    lw = w // 8; lh = h // 8
    bytes_per_frame = lw * lh * 128 * 2 * 2
    max_frames = int(free_bytes / bytes_per_frame)
    max_frames = max(9, min(max_frames, profile["chunk_frames"]))
    safe_frames = normalize_ltx_frame_count(max_frames, fps)
    print(f"  Auto chunk size: {safe_frames} frames ({free_gb:.2f} GB free)")
    return safe_frames

# =============================================================================
# SECTION 10 — RESOLUTION & PROFILE SELECTION
# =============================================================================

def select_profile(mode: str) -> Dict:
    if mode not in T4_PROFILES:
        print(f"  Unknown quality mode '{mode}', falling back to t4_safe.")
        mode = "t4_safe"
    p = T4_PROFILES[mode]
    print(f"  Quality mode: {mode} — {p['description']}")
    return p

def check_resolution_safety(w: int, h: int, mode: str) -> Tuple[int, int]:
    profile = select_profile(mode)
    safe_w = profile["generation_width"]; safe_h = profile["generation_height"]
    if w <= safe_w and h <= safe_h:
        return w, h
    print(f"\n  Resolution check: {w}×{h} exceeds safe {safe_w}×{safe_h} for {mode}")
    if CONFIG["allow_auto_downgrade"]:
        print(f"  → Auto-downgrading to {safe_w}×{safe_h}")
        return safe_w, safe_h
    return w, h


# =============================================================================
# SECTION 11 — INPUT IMAGE & AUDIO PREPARATION
# =============================================================================

def tensor_width_height(image) -> Tuple[int, int]:
    if isinstance(image, (tuple, list)):
        image = get_value_at_index(image, 0)
    if image.ndim == 4:
        return int(image.shape[2]), int(image.shape[1])
    if image.ndim == 3:
        return int(image.shape[1]), int(image.shape[0])
    raise ValueError(f"Unsupported image tensor shape: {getattr(image,'shape',None)}")

def load_input_image(image_path: Optional[str], width: int, height: int) -> Tuple:
    if image_path is not None:
        loaded = get_node("LoadImage").load_image(image=image_path)
        print(f"  ✓ Input image loaded: {image_path}")
        return loaded, 1.0, False
    noise_image = torch.full((1, height, width, 3), 0.5, dtype=torch.float32)
    print("  ✓ T2V mode — grey placeholder")
    return (noise_image, None), 0.0, True

def prepare_image_for_chunk(loaded_image_tuple, width: int, height: int,
                             img_compression: int = 18, longer_edge: int = 1312) -> Tuple:
    resizeimagemasknode      = get_node("ResizeImageMaskNode")
    resizeimagesbylongeredge = get_node("ResizeImagesByLongerEdge")
    ltxvpreprocess           = get_node("LTXVPreprocess")

    resized = resizeimagemasknode.EXECUTE_NORMALIZED(
        input=get_value_at_index(loaded_image_tuple, 0),
        scale_method="lanczos",
        resize_type={"resize_type":"scale dimensions","width":width,"height":height,"crop":"center"},
    )
    rescaled = resizeimagesbylongeredge.EXECUTE_NORMALIZED(
        longer_edge=longer_edge, images=get_value_at_index(resized, 0))
    preprocessed = ltxvpreprocess.EXECUTE_NORMALIZED(
        img_compression=img_compression, image=get_value_at_index(rescaled, 0))

    resized_w, resized_h = tensor_width_height(get_value_at_index(resized, 0))
    latent_w = max(1, resized_w // 2)
    latent_h = max(1, resized_h // 2)

    del resized, rescaled
    mem.stage_cleanup("prepare_image_for_chunk")   # [MEM-5]
    return preprocessed, latent_w, latent_h

def load_audio_file_lightweight(audio_path: Optional[str]) -> Optional[Dict]:
    if audio_path is None:
        return None
    if not os.path.exists(audio_path):
        print(f"  ✗ Audio not found: {audio_path}"); return None
    size_mb = os.path.getsize(audio_path) / (1024*1024)
    print(f"  ✓ Audio: {os.path.basename(audio_path)} ({size_mb:.1f} MB)")
    return {"path": audio_path, "loaded": False, "waveform": None, "sample_rate": None}

# =============================================================================
# SECTION 12 — TEXT CONDITIONING  (embedding cache)
# =============================================================================

_CONDITIONING_CACHE: Dict[str, Any] = {}

def build_text_conditioning(prompt: str, fps: int, cache_key: Optional[str] = None) -> Tuple:
    ck = cache_key or hashlib.md5(f"{prompt}|{fps}".encode()).hexdigest()
    if ck in _CONDITIONING_CACHE:
        print("  ✓ Conditioning from cache.")
        return _CONDITIONING_CACHE[ck]

    print("  Loading text encoder (DualCLIPLoader)...")
    mem.pre_op_guard("DualCLIPLoader", required_gb=4.0)   # [MEM-6]
    dualcliploader = get_node("DualCLIPLoader")
    try:
        clip_result = dualcliploader.load_clip(
            clip_name1=MODELS["text_encoder_1"], clip_name2=MODELS["text_encoder_2"],
            type="ltxv", device="default")
    except Exception as e:
        print(f"  Primary CLIP failed ({e}), trying fp8 fallback...")
        clip_result = dualcliploader.load_clip(
            clip_name1="gemma_3_12B_it_fp8_scaled.safetensors",
            clip_name2="ltx-2.3-22b-dev_embeddings_connectors.safetensors",
            type="ltxv", device="default")
    clip_obj = get_value_at_index(clip_result, 0)

    cliptextencode   = get_node("CLIPTextEncode")
    pos_encoded      = cliptextencode.encode(text=prompt, clip=clip_obj)
    conditioningzeroout = get_node("ConditioningZeroOut")
    neg_encoded      = conditioningzeroout.zero_out(
        conditioning=get_value_at_index(pos_encoded, 0))

    # Immediately free ~6 GB CLIP from VRAM
    del clip_result, clip_obj, dualcliploader, cliptextencode
    mem.aggressive_cleanup()                               # [MEM-5] full sweep after CLIP unload
    print_vram_usage("  ")                                 # [MEM-6] show VRAM after CLIP release

    ltxvconditioning = get_node("LTXVConditioning")
    cond = ltxvconditioning.EXECUTE_NORMALIZED(
        frame_rate=fps,
        positive=get_value_at_index(pos_encoded, 0),
        negative=get_value_at_index(neg_encoded, 0),
    )
    pos_cond = get_value_at_index(cond, 0)
    neg_cond = get_value_at_index(cond, 1)
    result   = (pos_cond, neg_cond)
    _CONDITIONING_CACHE[ck] = result
    print("  ✓ Text conditioning built and cached.")

    del pos_encoded, neg_encoded, cond
    mem.stage_cleanup("build_text_conditioning")           # [MEM-5]
    return result


# =============================================================================
# SECTION 13 — MODEL LOADING  (DiT, VAEs, Upscaler, LoRAs)
# =============================================================================

_DIT_MODEL_CACHE = None   # module-level fallback when ModelCache is disabled

def load_dit_model(apply_loras: bool = True) -> Any:
    """
    [MEM-2] Load DiT via ModelCache when enabled, else module cache.
    [MEM-3] Uses validate_lora_exists() for safe LoRA path resolution.
    Forces LoRAs off when profile sets disable_all_loras=True (ultra_safe).
    """
    global _DIT_MODEL_CACHE

    active_profile = T4_PROFILES.get(CONFIG["quality_mode"], {})
    force_no_loras = active_profile.get("disable_all_loras", False)

    def _load():
        global _DIT_MODEL_CACHE
        if _DIT_MODEL_CACHE is not None:
            return _DIT_MODEL_CACHE
        print("  Loading DiT model (UnetLoaderGGUF)...")
        mem.pre_op_guard("UnetLoaderGGUF", required_gb=1.0)  # [MEM-6]
        mem.cleanup()
        unetloadergguf = get_node("UnetLoaderGGUF")
        model = get_value_at_index(unetloadergguf.load_unet(unet_name=MODELS["dit"]), 0)
        mem.stage_cleanup("UnetLoaderGGUF load")              # [MEM-5]

        if apply_loras and not force_no_loras:
            from nodes import LoraLoaderModelOnly
            lora_loader = LoraLoaderModelOnly()
            lora_order = [
                ("lora_distilled",  LORA_STRENGTHS["lora_distilled"]),
                ("lora_omninft",    LORA_STRENGTHS["lora_omninft"]),
                ("lora_transition", LORA_STRENGTHS["lora_transition"]),
                ("lora_mvcamera",   LORA_STRENGTHS["lora_mvcamera"]),
            ]
            for lora_key, strength in lora_order:
                if not LORA_ENABLED.get(lora_key, True):
                    print(f"  LoRA disabled (VRAM save): {MODELS[lora_key]}")
                    continue
                # [MEM-3] validate_lora_exists for safe resolution
                lora_path = validate_lora_exists(MODELS[lora_key], "DiT")
                if lora_path:
                    print(f"  Applying LoRA: {MODELS[lora_key]}  strength={strength}")
                    model = lora_loader.load_lora_model_only(model, MODELS[lora_key], strength)[0]
                    mem.stage_cleanup(f"LoRA {lora_key}")     # [MEM-5] after every LoRA merge
                else:
                    print(f"  LoRA not found, skipping: {MODELS[lora_key]}")
        elif force_no_loras:
            print("  [ultra_safe] All LoRAs disabled to maximise free VRAM.")
        _DIT_MODEL_CACHE = model
        print("  DiT model ready.")
        print_vram_usage("  ")                                 # [MEM-6] VRAM bar after DiT load
        return model

    if MODEL_CACHE is not None:
        return MODEL_CACHE.get_unet(_load)
    return _load()

def release_dit_model():
    global _DIT_MODEL_CACHE
    if MODEL_CACHE is not None:
        MODEL_CACHE.evict_unet()
    if _DIT_MODEL_CACHE is not None:
        del _DIT_MODEL_CACHE
        _DIT_MODEL_CACHE = None
        mem.aggressive_cleanup()
        print("  DiT model released.")

def load_video_vae() -> Any:
    def _load():
        print("  Loading video VAE...")
        vaeloader = get_node("VAELoader")
        vae = get_value_at_index(vaeloader.load_vae(vae_name=MODELS["video_vae"]), 0)
        mem.stage_cleanup("video_vae load")                    # [MEM-5]
        return vae
    if MODEL_CACHE is not None:
        return MODEL_CACHE.get_video_vae(_load)
    return _load()

def load_audio_vae() -> Any:
    def _load():
        print("  Loading audio VAE...")
        from nodes import NODE_CLASS_MAPPINGS
        if "VAELoaderKJ" in NODE_CLASS_MAPPINGS:
            loader = NODE_CLASS_MAPPINGS["VAELoaderKJ"]()
            result = loader.load_vae(vae_name=MODELS["audio_vae"],
                                     device="main_device", weight_dtype="fp16")
        else:
            loader = NODE_CLASS_MAPPINGS["VAELoader"]()
            result = loader.load_vae(vae_name=MODELS["audio_vae"])
        vae = get_value_at_index(result, 0)
        mem.stage_cleanup("audio_vae load")                    # [MEM-5]
        return vae
    if MODEL_CACHE is not None:
        return MODEL_CACHE.get_audio_vae(_load)
    return _load()

def load_upscaler_model() -> Any:
    def _load():
        print("  Loading spatial upscaler...")
        loader  = get_node("LatentUpscaleModelLoader")
        result  = loader.EXECUTE_NORMALIZED(model_name=MODELS["upscaler"])
        upscaler = get_value_at_index(result, 0)
        mem.stage_cleanup("upscaler load")                     # [MEM-5]
        return upscaler
    if MODEL_CACHE is not None:
        return MODEL_CACHE.get_upscaler(_load)
    return _load()

def offload_model(model, name: str = "model"):
    if model is not None and hasattr(model, "to"):
        try:
            model.to("cpu")
        except Exception:
            pass
    mem.cleanup()
    print(f"  ↓ {name} offloaded to CPU")


# =============================================================================
# SECTION 14 — DIRECTOR WORKFLOW EXECUTION
# =============================================================================

def build_director_conditioning(pos_cond, neg_cond, image_path, audio_path,
                                  num_frames, fps, width, height,
                                  dit_model=None, audio_vae=None) -> Tuple:
    from nodes import NODE_CLASS_MAPPINGS
    active_profile = T4_PROFILES.get(CONFIG["quality_mode"], {})
    if active_profile.get("skip_director", False):
        print("  Skipping LTXDirector (profile skip_director=True).")
        return _build_director_fallback(pos_cond, neg_cond, num_frames, fps,
                                        dit_model=dit_model, audio_vae=audio_vae,
                                        reason="profile skip")

    if "LTXDirector" in NODE_CLASS_MAPPINGS:
        print("  Using LTXDirector (WhatDreamsCost) node...")
        if dit_model is None:
            dit_model = load_dit_model(apply_loras=True)
        if audio_vae is None:
            audio_vae = load_audio_vae()

        mem.pre_op_guard("DualCLIPLoader (Director)", required_gb=4.0)  # [MEM-6]
        dualcliploader = get_node("DualCLIPLoader")
        try:
            clip_result = dualcliploader.load_clip(
                clip_name1=MODELS["text_encoder_1"], clip_name2=MODELS["text_encoder_2"],
                type="ltxv", device="default")
        except Exception as e:
            print(f"  Primary CLIP failed ({e}), fp8 fallback...")
            clip_result = dualcliploader.load_clip(
                clip_name1="gemma_3_12B_it_fp8_scaled.safetensors",
                clip_name2="ltx-2.3-22b-dev_embeddings_connectors.safetensors",
                type="ltxv", device="default")
        clip_model = get_value_at_index(clip_result, 0)

        director_cls = NODE_CLASS_MAPPINGS["LTXDirector"]
        director     = director_cls()
        try:
            input_types = director_cls.INPUT_TYPES()
        except Exception:
            input_types = {"required": {}, "optional": {}}
        all_accepted = set(input_types.get("required", {}).keys()) | set(input_types.get("optional", {}).keys())

        duration_s   = num_frames / fps
        director_kwargs = dict(model=dit_model, audio_vae=audio_vae, global_prompt=GLOBAL_PROMPT)
        if not all_accepted or "clip" in all_accepted:
            director_kwargs["clip"] = clip_model

        widget_defaults = {
            "start_second": 0, "end_second": duration_s, "duration_seconds": duration_s,
            "start_frame": 0,  "end_frame": num_frames,  "duration_frames": num_frames,
            "timeline_data": json.dumps({
                "mainTrackEnabled":True,"audioTrackEnabled":True,"motionTrackEnabled":True,
                "global_prompt":GLOBAL_PROMPT,"retakeMode":False,
                "normalStartFrame":0,"normalDurationFrames":num_frames,
                "segments":[],"motionSegments":[],"audioSegments":[],
            }),
            "local_prompts":"","segment_lengths":"","epsilon":0.001,
            "guide_strength":"1.00","frame_rate":fps,
            "custom_width":width,"custom_height":height,
            "resize_method":"maintain aspect ratio","divisible_by":32,
            "img_compression":WORKFLOW_IMG_COMPRESSION,"retakeMode":False,"timeline_ui":"",
        }
        for k, v in widget_defaults.items():
            if k in all_accepted:
                director_kwargs[k] = v
        if not all_accepted:
            for k in ["start_second","end_second","duration_seconds",
                      "start_frame","end_frame","duration_frames",
                      "timeline_data","local_prompts","segment_lengths"]:
                director_kwargs.setdefault(k, widget_defaults[k])

        try:
            fn = getattr(director_cls, "FUNCTION", None)
            director_out = getattr(director, fn)(**director_kwargs) if fn else director.EXECUTE_NORMALIZED(**director_kwargs)
        except (TypeError, AttributeError) as e:
            print(f"  LTXDirector failed ({e}) — fallback.")
            del clip_result, clip_model
            mem.aggressive_cleanup()                           # [MEM-5]
            return _build_director_fallback(pos_cond, neg_cond, num_frames, fps,
                                            dit_model=dit_model, audio_vae=audio_vae,
                                            reason=f"call failed: {e}")

        del clip_result, clip_model
        mem.aggressive_cleanup()                               # [MEM-5] free CLIP after Director
        print_vram_usage("  ")                                 # [MEM-6]

        return (
            get_value_at_index(director_out, 0),
            get_value_at_index(director_out, 1),
            get_value_at_index(director_out, 2),
            get_value_at_index(director_out, 3),
            get_value_at_index(director_out, 4) if len(director_out) > 4 else None,
            get_value_at_index(director_out, 5) if len(director_out) > 5 else None,
            get_value_at_index(director_out, 6) if len(director_out) > 6 else fps,
        )

    return _build_director_fallback(pos_cond, neg_cond, num_frames, fps,
                                    dit_model=dit_model, audio_vae=audio_vae)

def _build_director_fallback(pos_cond, neg_cond, num_frames, fps,
                              dit_model=None, audio_vae=None, reason="not found") -> Tuple:
    print(f"  LTXDirector fallback ({reason}).")
    if dit_model is None:
        dit_model = load_dit_model(apply_loras=True)
    if audio_vae is None:
        audio_vae = load_audio_vae()
    ltxvemptylatentaudio = get_node("LTXVEmptyLatentAudio")
    audio_lat = ltxvemptylatentaudio.EXECUTE_NORMALIZED(
        frames_number=num_frames, frame_rate=fps, batch_size=1, audio_vae=audio_vae)
    mem.stage_cleanup("director_fallback")                     # [MEM-5]
    return dit_model, pos_cond, None, get_value_at_index(audio_lat, 0), None, None, fps


def run_director_guide(pos_cond, neg_cond, video_vae, latent,
                        guide_data, motion_guide_data, model,
                        upscale_factor: float = 1.0, node_id: str = "pass") -> Tuple:
    from nodes import NODE_CLASS_MAPPINGS
    if "LTXDirectorGuide" not in NODE_CLASS_MAPPINGS:
        return pos_cond, neg_cond, latent, model
    if guide_data is None and motion_guide_data is None:
        print(f"  LTXDirectorGuide ({node_id}): no guide data — passthrough.")
        return pos_cond, neg_cond, latent, model

    guide_cls  = NODE_CLASS_MAPPINGS["LTXDirectorGuide"]
    guide_node = guide_cls()
    try:
        input_types = guide_cls.INPUT_TYPES()
    except Exception:
        input_types = {"required": {}, "optional": {}}
    all_accepted = set(input_types.get("required", {}).keys()) | set(input_types.get("optional", {}).keys())

    inputs = dict(positive=pos_cond, negative=neg_cond, vae=video_vae, latent=latent, model=model)
    widget_candidates = {
        "retake_image":"None","upscale_factor_pass":1,"upscale_factor":upscale_factor,
        "interpolation":"bicubic","blend_radius":1,"crop_method":"center",
        "use_tiling":True,"tile_overlap":False,"tile_size":256,
        "tile_stride":64,"force_inpaint":False,
    }
    if all_accepted:
        for k, v in widget_candidates.items():
            if k in all_accepted:
                inputs[k] = v
    else:
        for k, v in widget_candidates.items():
            if k != "retake_image":
                inputs[k] = v
    if "guide_data" in all_accepted or not all_accepted:
        inputs["guide_data"] = guide_data
    if motion_guide_data is not None:
        inputs["motion_guide_data"] = motion_guide_data

    try:
        fn  = getattr(guide_cls, "FUNCTION", None)
        out = getattr(guide_node, fn)(**inputs) if fn else guide_node.EXECUTE_NORMALIZED(**inputs)
    except (TypeError, AttributeError) as e:
        print(f"  LTXDirectorGuide ({node_id}) failed: {e} — passthrough.")
        return pos_cond, neg_cond, latent, model

    mem.stage_cleanup(f"director_guide_{node_id}")             # [MEM-5]
    return (get_value_at_index(out, 0), get_value_at_index(out, 1),
            get_value_at_index(out, 2),
            get_value_at_index(out, 3) if len(out) > 3 else model)

def run_director_crop_guides(pos_cond, neg_cond, latent,
                              prefer_standard: bool = False) -> Tuple:
    from nodes import NODE_CLASS_MAPPINGS
    use_director = ("LTXDirectorCropGuides" in NODE_CLASS_MAPPINGS) and not prefer_standard
    if not use_director:
        if "LTXVCropGuides" in NODE_CLASS_MAPPINGS:
            out = NODE_CLASS_MAPPINGS["LTXVCropGuides"]().EXECUTE_NORMALIZED(
                positive=pos_cond, negative=neg_cond, latent=latent)
        else:
            return pos_cond, neg_cond, latent
    else:
        crop_cls  = NODE_CLASS_MAPPINGS["LTXDirectorCropGuides"]
        crop_node = crop_cls()
        try:
            fn  = getattr(crop_cls, "FUNCTION", None)
            out = getattr(crop_node, fn)(positive=pos_cond, negative=neg_cond, latent=latent) \
                  if fn else crop_node.EXECUTE_NORMALIZED(positive=pos_cond, negative=neg_cond, latent=latent)
        except (TypeError, AttributeError) as e:
            print(f"  LTXDirectorCropGuides failed: {e} — passthrough.")
            return pos_cond, neg_cond, latent

    mem.stage_cleanup("crop_guides")                           # [MEM-5]
    return (get_value_at_index(out, 0) or pos_cond,
            get_value_at_index(out, 1) or neg_cond,
            get_value_at_index(out, 2))


# =============================================================================
# SECTION 15 — TWO-PASS SAMPLING PIPELINE
# =============================================================================

def build_empty_latents(num_frames, latent_w, latent_h, fps,
                         image_preprocessed, image_strength, image_bypass,
                         video_vae, audio_vae) -> Tuple:
    emptyltxvlatentvideo = get_node("EmptyLTXVLatentVideo")
    empty_video_lat = emptyltxvlatentvideo.EXECUTE_NORMALIZED(
        width=latent_w, height=latent_h, length=num_frames, batch_size=1)

    ltxvimgtovideoinplace = get_node("LTXVImgToVideoInplace")
    img_conditioned_lat = ltxvimgtovideoinplace.EXECUTE_NORMALIZED(
        strength=image_strength, bypass=image_bypass, vae=video_vae,
        image=get_value_at_index(image_preprocessed, 0),
        latent=get_value_at_index(empty_video_lat, 0))

    ltxvemptylatentaudio = get_node("LTXVEmptyLatentAudio")
    empty_audio_lat = ltxvemptylatentaudio.EXECUTE_NORMALIZED(
        frames_number=num_frames, frame_rate=fps, batch_size=1, audio_vae=audio_vae)

    ltxvconcatavlatent = get_node("LTXVConcatAVLatent")
    video_src = get_value_at_index(img_conditioned_lat, 0) if not image_bypass \
                else get_value_at_index(empty_video_lat, 0)
    av_latent = ltxvconcatavlatent.EXECUTE_NORMALIZED(
        video_latent=video_src,
        audio_latent=get_value_at_index(empty_audio_lat, 0))

    del empty_video_lat, empty_audio_lat
    mem.stage_cleanup("build_empty_latents")                   # [MEM-5]
    return av_latent, img_conditioned_lat


def _get_sigmas_for_pass(model, steps: int, denoise: float, pass_label: str):
    """
    [MEM-4] Return sigma schedule via ManualSigmas (manual presets) or
    BasicScheduler (scheduler mode), depending on SIGMA_PRESET_MODE.
    ManualSigmas gives tighter VRAM usage because fewer intermediate latents
    are created compared to full-step schedules.
    """
    mode = CONFIG.get("sigma_preset_mode", "scheduler")
    if mode in SIGMA_PRESETS:
        from nodes import NODE_CLASS_MAPPINGS
        if "ManualSigmas" in NODE_CLASS_MAPPINGS:
            preset = SIGMA_PRESETS[mode]
            sigma_str = preset["pass1"] if "pass1" in pass_label.lower() else preset["pass2"]
            msig = NODE_CLASS_MAPPINGS["ManualSigmas"]()
            print(f"  Sigmas ({mode} {pass_label}): {sigma_str}")
            return msig.EXECUTE_NORMALIZED(sigmas=sigma_str)
        else:
            print("  ManualSigmas node not found — falling back to BasicScheduler")
    # Default: BasicScheduler
    basicscheduler = get_node("BasicScheduler")
    return basicscheduler.EXECUTE_NORMALIZED(
        model=model, scheduler=WORKFLOW_SCHEDULER,
        steps=steps, denoise=denoise)


def run_sampling_pass(model, pos_cond, neg_cond, latent, noise_seed: int,
                       steps: int = WORKFLOW_STEPS, cfg: float = WORKFLOW_CFG,
                       denoise: float = 1.0, pass_name: str = "Pass1") -> Any:
    print(f"  Sampling {pass_name} ({steps} steps, denoise={denoise}, seed={noise_seed})...")
    mem.pre_op_guard(f"sampling_{pass_name}", required_gb=1.0)  # [MEM-6]
    print_vram_usage("  ")                                       # [MEM-6]

    ksamplerselect = get_node("KSamplerSelect")
    sampler = ksamplerselect.EXECUTE_NORMALIZED(sampler_name=WORKFLOW_SAMPLER_PASS1)

    randomnoise = get_node("RandomNoise")
    noise = randomnoise.EXECUTE_NORMALIZED(noise_seed=noise_seed)

    sigmas = _get_sigmas_for_pass(model, steps, denoise, pass_name)  # [MEM-4]

    cfgguider = get_node("CFGGuider")
    guider = cfgguider.EXECUTE_NORMALIZED(cfg=cfg, model=model,
                                           positive=pos_cond, negative=neg_cond)

    samplercustomadvanced = get_node("SamplerCustomAdvanced")
    result = samplercustomadvanced.EXECUTE_NORMALIZED(
        noise=get_value_at_index(noise, 0),
        guider=get_value_at_index(guider, 0),
        sampler=get_value_at_index(sampler, 0),
        sigmas=get_value_at_index(sigmas, 0),
        latent_image=latent,
    )
    del noise, sampler, sigmas, guider
    mem.stage_cleanup(f"sampling_{pass_name}")                   # [MEM-5]
    return result


def separate_av_latent(sampler_output, output_index: int = 0) -> Tuple:
    ltxvseparateavlatent = get_node("LTXVSeparateAVLatent")
    separated = ltxvseparateavlatent.EXECUTE_NORMALIZED(
        av_latent=get_value_at_index(sampler_output, output_index))
    mem.stage_cleanup("separate_av_latent")                      # [MEM-5]
    return get_value_at_index(separated, 0), get_value_at_index(separated, 1)


def upsample_video_latent(video_latent, upscaler_model, video_vae) -> Any:
    print("  Upsampling latent (2×)...")
    mem.pre_op_guard("LTXVLatentUpsampler", required_gb=0.5)    # [MEM-6]
    ltxvlatentupsampler = get_node("LTXVLatentUpsampler")
    result = ltxvlatentupsampler.upsample_latent(
        samples=video_latent, upscale_model=upscaler_model, vae=video_vae)
    mem.stage_cleanup("upsample_video_latent")                   # [MEM-5]
    return get_value_at_index(result, 0)


def recondition_image_on_upscaled(upscaled_latent, image_preprocessed,
                                    image_strength, image_bypass, video_vae,
                                    audio_lat_pass1) -> Any:
    ltxvimgtovideoinplace = get_node("LTXVImgToVideoInplace")
    if not image_bypass:
        reconditioned = ltxvimgtovideoinplace.EXECUTE_NORMALIZED(
            strength=image_strength, bypass=image_bypass, vae=video_vae,
            image=get_value_at_index(image_preprocessed, 0), latent=upscaled_latent)
        video_lat_p2 = get_value_at_index(reconditioned, 0)
    else:
        video_lat_p2 = upscaled_latent
    ltxvconcatavlatent = get_node("LTXVConcatAVLatent")
    av_latent_pass2 = ltxvconcatavlatent.EXECUTE_NORMALIZED(
        video_latent=video_lat_p2, audio_latent=audio_lat_pass1)
    mem.stage_cleanup("recondition_image_on_upscaled")           # [MEM-5]
    return av_latent_pass2


# =============================================================================
# SECTION 16 — VAE DECODING  [MEM-8] enhanced sub-batch decode
# =============================================================================

def decode_video_latent(video_latent, video_vae, max_batch_frames: int = 0) -> Any:
    """
    [MEM-8] Chunked, VRAM-aware video VAE decode.

    Improvements over original:
      - Pre-op VRAM guard before decode begins
      - non_blocking GPU→CPU transfer with explicit synchronize()
      - Dynamic sub-batch size: scales down to 4 frames if RAM is critically low
      - Per-sub-batch stage_cleanup() + VRAM bar print
      - Immediate del of GPU tensors after each sub-batch
    """
    print("  VAE decoding video latent...")
    mem.pre_op_guard("VAEDecode", required_gb=1.0)              # [MEM-6]
    vaedecode = get_node("VAEDecode")

    latent_samples = video_latent["samples"] if isinstance(video_latent, dict) else video_latent

    if torch.is_tensor(latent_samples) and latent_samples.ndim == 5:
        t_latent = latent_samples.shape[2]
        h_latent = latent_samples.shape[3]
        w_latent = latent_samples.shape[4]
        est_h = h_latent * 8; est_w = w_latent * 8
        est_frames  = t_latent * 8
        est_ram_gb  = mem.estimate_frame_ram_gb(est_frames, est_h, est_w)
        avail_ram   = mem.cpu_available_gb()

        # [MEM-8] Dynamically pick sub-batch size
        use_subbatch = False
        if max_batch_frames > 0:
            use_subbatch = True
            batch_t = max_batch_frames
        elif avail_ram < est_ram_gb + 3.0:
            use_subbatch = True
            # Drop to 4-frame sub-batches when RAM is critically low
            batch_t = 4 if avail_ram < est_ram_gb else 8
            print(f"  [MEM-8] Sub-batch decode: avail={avail_ram:.2f} GB, "
                  f"need≈{est_ram_gb:.2f} GB → batch_t={batch_t}")

        if use_subbatch and t_latent > batch_t:
            all_frames = []
            for t_start in range(0, t_latent, batch_t):
                t_end = min(t_start + batch_t, t_latent)
                sub_lat = {"samples": latent_samples[:, :, t_start:t_end, :, :]}
                decoded  = vaedecode.decode(samples=sub_lat, vae=video_vae)
                frames_gpu = get_value_at_index(decoded, 0)
                # [MEM-8] non_blocking + explicit sync
                frames_cpu_batch = frames_gpu.detach().to("cpu", non_blocking=True)
                torch.cuda.synchronize()
                all_frames.append(frames_cpu_batch)
                del frames_gpu, decoded, sub_lat
                mem.stage_cleanup(f"sub-batch decode t={t_start}-{t_end}")  # [MEM-5]
                if CONFIG["enable_memory_logging"]:
                    print_vram_usage("    ")                    # [MEM-6] per sub-batch
            frames_cpu = torch.cat(all_frames, dim=0)
            del all_frames
            mem.soft_cleanup()
            return frames_cpu

    # Standard full decode
    decoded    = vaedecode.decode(samples=video_latent, vae=video_vae)
    frames_gpu = get_value_at_index(decoded, 0)
    frames_cpu = frames_gpu.detach().to("cpu", non_blocking=True)
    torch.cuda.synchronize()
    del frames_gpu, decoded
    mem.stage_cleanup("decode_video_latent")                    # [MEM-5]
    return frames_cpu


def decode_audio_latent(audio_latent, audio_vae) -> Any:
    print("  VAE decoding audio latent...")
    ltxvaudiovaedecode = get_node("LTXVAudioVAEDecode")
    decoded  = ltxvaudiovaedecode.EXECUTE_NORMALIZED(samples=audio_latent, audio_vae=audio_vae)
    audio_out = get_value_at_index(decoded, 0)
    if torch.is_tensor(audio_out):
        audio_out = audio_out.detach().cpu()
    elif isinstance(audio_out, dict) and "waveform" in audio_out:
        if torch.is_tensor(audio_out["waveform"]):
            audio_out = {**audio_out, "waveform": audio_out["waveform"].detach().cpu()}
    del decoded
    mem.stage_cleanup("decode_audio_latent")                    # [MEM-5]
    return audio_out


# =============================================================================
# SECTION 17 — CHUNK SAVING
# =============================================================================

def save_chunk_to_disk(frames_cpu, audio_cpu, chunk_index, fps, width, height) -> str:
    chunks_dir = os.path.join(CONFIG["workspace_dir"], "chunks")
    Path(chunks_dir).mkdir(parents=True, exist_ok=True)
    chunk_path = os.path.join(chunks_dir, f"chunk_{chunk_index:04d}.mp4")

    ram_too_low = not mem.is_ram_safe(required_gb=4.0)
    if ram_too_low:
        print(f"  [mem] RAM low ({mem.cpu_available_gb():.2f} GB) — using streaming ffmpeg.")

    if not ram_too_low:
        try:
            from nodes import NODE_CLASS_MAPPINGS
            if "CreateVideo" in NODE_CLASS_MAPPINGS:
                createvideo = NODE_CLASS_MAPPINGS["CreateVideo"]()
                video_obj = createvideo.EXECUTE_NORMALIZED(
                    fps=fps, images=frames_cpu, audio=audio_cpu)
                video = get_value_at_index(video_obj, 0)
                import folder_paths
                from comfy_api.latest import Types
                w = frames_cpu.shape[2] if frames_cpu.ndim == 4 else width
                h = frames_cpu.shape[1] if frames_cpu.ndim == 4 else height
                full_folder, fname, counter, _, _ = folder_paths.get_save_image_path(
                    f"chunk_{chunk_index:04d}", folder_paths.get_output_directory(), w, h)
                ext = Types.VideoContainer.get_extension("auto")
                tmp_path = os.path.join(full_folder, f"{fname}_{counter:05d}_.{ext}")
                video.save_to(tmp_path, format=Types.VideoContainer("auto"),
                              codec="auto", metadata=None)
                shutil.move(tmp_path, chunk_path)
                del video_obj, video
                mem.stage_cleanup("save_chunk CreateVideo")    # [MEM-5]
                print(f"  ✓ Chunk {chunk_index:04d} saved (CreateVideo): {chunk_path}")
                return chunk_path
        except Exception as e:
            print(f"  CreateVideo failed ({e}), falling back to ffmpeg pipe...")

    _write_chunk_via_ffmpeg(frames_cpu, audio_cpu, chunk_path, fps, width, height)
    return chunk_path


def _write_chunk_via_ffmpeg(frames_cpu, audio_cpu, out_path, fps, w, h):
    n_frames = frames_cpu.shape[0] if torch.is_tensor(frames_cpu) else frames_cpu.shape[0]
    fh = frames_cpu.shape[1]; fw = frames_cpu.shape[2]
    cmd = ["ffmpeg","-y","-f","rawvideo","-vcodec","rawvideo",
           "-s",f"{fw}x{fh}","-pix_fmt","rgb24","-r",str(fps),"-i","pipe:0",
           "-vcodec","libx264","-pix_fmt","yuv420p","-crf","18","-preset","fast",
           out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for i in range(n_frames):
        frame = (frames_cpu[i].clamp(0,1)*255).byte().numpy() \
                if torch.is_tensor(frames_cpu) else frames_cpu[i]
        proc.stdin.write(frame.tobytes())
        del frame
        if i % 16 == 0:
            gc.collect()                                       # [MEM-5] periodic GC during write
    proc.stdin.close(); proc.wait()
    print(f"  ✓ Chunk {os.path.basename(out_path)} saved (ffmpeg stream).")


def compute_file_checksum(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# =============================================================================
# SECTION 18 — SINGLE CHUNK GENERATION   [MEM-5] [MEM-6] dense application
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
    Full LTX-2.3 Director 2.0 two-pass pipeline for one temporal chunk.

    [MEM-5] mem.stage_cleanup()   called after EVERY pipeline stage.
    [MEM-6] mem.pre_op_guard()    called before every heavy allocation.
    [MEM-6] print_vram_usage()    printed at chunk start, after DiT load,
                                   before/after each sampling pass, and before decode.
    [MEM-2] ModelCache            keeps DiT+VAEs alive between chunks.
    [MEM-4] _get_sigmas_for_pass  uses ManualSigmas when sigma_preset_mode != 'scheduler'.
    """
    idx         = chunk_desc["chunk_index"]
    start_frame = chunk_desc["start_frame"]
    num_frames  = chunk_desc["num_frames"]
    chunk_seed  = get_chunk_seed(global_seed, idx)
    img_compress = profile.get("img_compression", WORKFLOW_IMG_COMPRESSION)
    longer_edge  = profile.get("longer_edge", 1312)
    no_director_data = False   # updated after build_director_conditioning

    mem.set_chunk_info(idx, num_frames, width, height)

    # ── [MEM-6] Per-chunk banner + VRAM bar ───────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  CHUNK {idx+1:03d}  frames {start_frame}–{start_frame+num_frames-1}"
          f"  ({num_frames} frames)  seed={chunk_seed}")
    print(f"{'='*62}")
    print_vram_usage("  ")
    print(f"  {mem.ram_status()}")

    # [MEM-6] Hard pre-chunk VRAM guard — aggressive cleanup before anything loads
    mem.pre_op_guard(f"chunk_{idx}_start", required_gb=2.0)

    with torch.inference_mode():

        # ── Stage 1: Image preprocessing ──────────────────────────────────────
        preprocessed, latent_w, latent_h = prepare_image_for_chunk(
            loaded_image_tuple, width, height, img_compress, longer_edge)
        # stage_cleanup already called inside prepare_image_for_chunk

        # ── Stage 2: Load VAEs ────────────────────────────────────────────────
        # [MEM-6] Defer video_vae in t4_safe to reduce peak VRAM during DiT load
        active_profile = T4_PROFILES.get(CONFIG["quality_mode"], {})
        if active_profile.get("skip_director", False):
            print("  [t4_safe] Deferring video VAE — loading audio VAE only now.")
            video_vae = None
            audio_vae = load_audio_vae()
        else:
            video_vae = load_video_vae()
            audio_vae = load_audio_vae()
        mem.stage_cleanup("vae_load")                          # [MEM-5]
        print_vram_usage("  ")                                 # [MEM-6]

        # ── Stage 3: Upscaler deferred ────────────────────────────────────────
        # Do NOT load upscaler here — it would consume VRAM during pass-1 sampling.
        # Loaded at Stage 12 (between pass 1 and pass 2).
        upscaler = None

        # ── Stage 4: LTXDirector / fallback conditioning ──────────────────────
        mem.pre_op_guard("build_director_conditioning", required_gb=1.0)  # [MEM-6]
        director_result = build_director_conditioning(
            pos_cond=pos_cond, neg_cond=neg_cond,
            image_path=None, audio_path=None,
            num_frames=num_frames, fps=fps,
            width=width, height=height,
            audio_vae=audio_vae,
        )
        (dir_model, dir_positive, dir_video_latent, dir_audio_latent,
         dir_guide_data, dir_motion_guide_data, dir_frame_rate) = director_result

        no_director_data = (dir_guide_data is None and dir_motion_guide_data is None)
        base_model = dir_model
        mem.stage_cleanup("build_director_conditioning")       # [MEM-5]
        print_vram_usage("  ")                                 # [MEM-6] after DiT in memory

        # ── Stage 5: Determine video latent + audio for pass 1 ───────────────
        if dir_video_latent is not None:
            video_latent_pass1     = dir_video_latent
            audio_latent_for_concat = dir_audio_latent
        else:
            if video_vae is None:
                video_vae = load_video_vae()
            av_latent_p1, img_conditioned_lat = build_empty_latents(
                num_frames, latent_w, latent_h, fps,
                preprocessed, image_strength, image_bypass,
                video_vae, audio_vae)
            video_latent_pass1 = get_value_at_index(img_conditioned_lat, 0)
            fresh_audio = get_node("LTXVEmptyLatentAudio")
            fresh_audio_lat = fresh_audio.EXECUTE_NORMALIZED(
                frames_number=num_frames, frame_rate=fps, batch_size=1, audio_vae=audio_vae)
            audio_latent_for_concat = get_value_at_index(fresh_audio_lat, 0)
            del av_latent_p1, img_conditioned_lat, fresh_audio_lat
            mem.stage_cleanup("build_empty_latents_fallback")  # [MEM-5]

        # ── Stage 6b: Director-aware conditioning wrap ────────────────────────
        if dir_positive is not None:
            conditioningzeroout = get_node("ConditioningZeroOut")
            neg_from_dir = conditioningzeroout.zero_out(conditioning=dir_positive)
            ltxvconditioning = get_node("LTXVConditioning")
            dir_cond = ltxvconditioning.EXECUTE_NORMALIZED(
                frame_rate=dir_frame_rate,
                positive=dir_positive,
                negative=get_value_at_index(neg_from_dir, 0))
            cond_pos_g = get_value_at_index(dir_cond, 0)
            cond_neg_g = get_value_at_index(dir_cond, 1)
            del neg_from_dir, dir_cond
            mem.stage_cleanup("director_cond_wrap")            # [MEM-5]
        else:
            cond_pos_g = pos_cond
            cond_neg_g = neg_cond

        # ── Stage 7: Ensure video_vae loaded, then Director Guide pass 1 ─────
        if video_vae is None:
            video_vae = load_video_vae()
        mem.stage_cleanup("video_vae_deferred_load")           # [MEM-5]

        mem.pre_op_guard("director_guide_pass1", required_gb=0.5)  # [MEM-6]
        pos_g1, neg_g1, lat_g1, model_g1 = run_director_guide(
            pos_cond=cond_pos_g, neg_cond=cond_neg_g,
            video_vae=video_vae, latent=video_latent_pass1,
            guide_data=dir_guide_data, motion_guide_data=dir_motion_guide_data,
            model=base_model, upscale_factor=0.5, node_id="pass1 (node 133)")
        del video_latent_pass1
        mem.stage_cleanup("director_guide_pass1")              # [MEM-5]

        # ── Stage 8: LTXVConcatAVLatent — Guide133 latent + audio ─────────────
        if audio_latent_for_concat is not None:
            ltxvconcatavlatent = get_node("LTXVConcatAVLatent")
            av_concat_p1 = ltxvconcatavlatent.EXECUTE_NORMALIZED(
                video_latent=lat_g1, audio_latent=audio_latent_for_concat)
            latent_for_s1 = get_value_at_index(av_concat_p1, 0)
            del av_concat_p1
            mem.stage_cleanup("concat_av_pass1")               # [MEM-5]
        else:
            latent_for_s1 = lat_g1

        # ── Stage 9: Sampling pass 1 (8 steps, denoise=1.0) ──────────────────
        mem.warn_if_low()
        print_vram_usage("  ")                                 # [MEM-6] before pass 1
        sample_out_1 = run_sampling_pass(
            model=model_g1, pos_cond=pos_g1, neg_cond=neg_g1,
            latent=latent_for_s1, noise_seed=chunk_seed,
            steps=WORKFLOW_STEPS, cfg=WORKFLOW_CFG, denoise=1.0,
            pass_name=f"Pass1 (chunk {idx})")
        del latent_for_s1
        mem.aggressive_cleanup()                               # [MEM-5] full sweep after pass 1
        print_vram_usage("  ")                                 # [MEM-6] after pass 1

        # ── Stage 10: Separate AV latent (node 34) ───────────────────────────
        video_lat_p1, audio_lat_p1 = separate_av_latent(sample_out_1, output_index=0)
        del sample_out_1
        mem.stage_cleanup("separate_av_pass1")                 # [MEM-5]

        # ── Stage 11: Director Crop Guides (node 55) ─────────────────────────
        pos_crop55, neg_crop55, lat_crop55 = run_director_crop_guides(
            pos_cond=pos_g1, neg_cond=neg_g1,
            latent=video_lat_p1, prefer_standard=no_director_data)
        del video_lat_p1, pos_g1, neg_g1, model_g1
        mem.stage_cleanup("crop_guides_55")                    # [MEM-5]

        # ── Stage 12: Load upscaler (deferred), upsample 2× ─────────────────
        mem.pre_op_guard("upscaler_load", required_gb=0.5)     # [MEM-6]
        upscaler = load_upscaler_model()
        upscaled_lat = upsample_video_latent(lat_crop55, upscaler, video_vae)
        del lat_crop55, upscaler
        mem.aggressive_cleanup()                               # [MEM-5] free upscaler ASAP
        print_vram_usage("  ")                                 # [MEM-6] after upscale

        # ── Stage 13: Director Guide pass 2 (node 132, upscale_factor=1.0) ───
        mem.pre_op_guard("director_guide_pass2", required_gb=0.5)  # [MEM-6]
        pos_g2, neg_g2, lat_g2, model_g2 = run_director_guide(
            pos_cond=pos_crop55, neg_cond=neg_crop55,
            video_vae=video_vae, latent=upscaled_lat,
            guide_data=dir_guide_data, motion_guide_data=dir_motion_guide_data,
            model=base_model, upscale_factor=1.0, node_id="pass2 (node 132)")
        del pos_crop55, neg_crop55, upscaled_lat
        mem.stage_cleanup("director_guide_pass2")              # [MEM-5]

        # Free director tensors — no longer needed after Guide pass 2
        for _name in ("dir_model","dir_guide_data","dir_motion_guide_data",
                       "dir_frame_rate","dir_positive","dir_video_latent","dir_audio_latent"):
            try:
                del locals()[_name]
            except KeyError:
                pass
        # Force-delete via explicit assignments
        dir_guide_data = None; dir_motion_guide_data = None
        mem.stage_cleanup("free_director_tensors")             # [MEM-5]

        # ── Stage 14: LTXVConcatAVLatent — Guide132 latent + pass1 audio ─────
        ltxvconcatavlatent2 = get_node("LTXVConcatAVLatent")
        av_concat_p2 = ltxvconcatavlatent2.EXECUTE_NORMALIZED(
            video_latent=lat_g2, audio_latent=audio_lat_p1)
        latent_for_s2 = get_value_at_index(av_concat_p2, 0)
        del av_concat_p2, audio_lat_p1, lat_g2
        mem.stage_cleanup("concat_av_pass2")                   # [MEM-5]

        # ── Stage 15: Sampling pass 2 (4 steps, denoise=0.42) ────────────────
        mem.warn_if_low()
        print_vram_usage("  ")                                 # [MEM-6] before pass 2
        sample_out_2 = run_sampling_pass(
            model=model_g2, pos_cond=pos_g2, neg_cond=neg_g2,
            latent=latent_for_s2, noise_seed=0,
            steps=WORKFLOW_STEPS_PASS2, cfg=WORKFLOW_CFG,
            denoise=WORKFLOW_DENOISE_PASS2,
            pass_name=f"Pass2 (chunk {idx})")
        del latent_for_s2, model_g2

        # [MEM-5][MEM-6] Release DiT BEFORE decode to free ~12 GB VRAM
        try:
            del base_model
        except NameError:
            pass
        release_dit_model()
        mem.aggressive_cleanup()                               # [MEM-5] biggest sweep of chunk
        print_vram_usage("  ")                                 # [MEM-6] after DiT released

        # ── Stage 16: Separate final AV latent (node 22) ─────────────────────
        final_video_lat, final_audio_lat = separate_av_latent(sample_out_2, output_index=0)
        del sample_out_2
        mem.stage_cleanup("separate_av_pass2")                 # [MEM-5]

        # ── Stage 17: Director Crop Guides (node 54) ─────────────────────────
        pos_crop54, neg_crop54, lat_crop54 = run_director_crop_guides(
            pos_cond=pos_g2, neg_cond=neg_g2,
            latent=final_video_lat, prefer_standard=no_director_data)
        del pos_g2, neg_g2, final_video_lat, pos_crop54, neg_crop54
        mem.aggressive_cleanup()                               # [MEM-5]
        mem.ram_cleanup()                                      # [MEM-5] OS-level RAM reclaim

        # [MEM-6] RAM safety check before decode
        if not mem.is_ram_safe(required_gb=3.0):
            print(f"  ⚠ RAM critically low ({mem.cpu_available_gb():.2f} GB) before decode!")

        # ── Stage 18: Decode video ─────────────────────────────────────────────
        print_vram_usage("  ")                                 # [MEM-6] before decode
        frames_cpu = decode_video_latent(lat_crop54, video_vae)
        del lat_crop54
        mem.stage_cleanup("decode_video")                      # [MEM-5]

        # ── Stage 19: Decode audio ─────────────────────────────────────────────
        audio_cpu = decode_audio_latent(final_audio_lat, audio_vae)
        del final_audio_lat
        mem.stage_cleanup("decode_audio")                      # [MEM-5]

        # ── Stage 20: Unload VAEs ──────────────────────────────────────────────
        # [MEM-2] If ModelCache is disabled, delete VAEs; otherwise keep cached.
        if MODEL_CACHE is None:
            del video_vae, audio_vae
            mem.cleanup()
        else:
            # Let ModelCache keep them; just release local references
            video_vae = None; audio_vae = None
        mem.stage_cleanup("vae_unload")                        # [MEM-5]

    # ── Stage 21: Save chunk to disk ──────────────────────────────────────────
    chunk_path = save_chunk_to_disk(frames_cpu, audio_cpu, idx, fps, width, height)
    del frames_cpu, audio_cpu
    gc.collect(); _malloc_trim()                               # [MEM-5] OS RAM after frames deleted

    # [MEM-7] Optional shot quality metrics log
    if CONFIG["enable_memory_logging"] and os.path.exists(chunk_path):
        score, metrics = calculate_shot_metrics(chunk_path)
        print(f"  📊 Chunk quality: score={score:.3f}  "
              f"sharp={metrics.get('sharpness',0):.0f}  "
              f"bright={metrics.get('brightness',0):.0f}  "
              f"motion_std={metrics.get('motion_std',0):.2f}")

    # [MEM-7] Optional face restoration
    if CONFIG.get("face_restoration", False):
        chunk_path = apply_face_restoration(chunk_path)

    if CONFIG["enable_memory_logging"]:
        print(f"  GPU after chunk {idx}: {mem.gpu_free_gb():.2f} GB free")
        print_vram_usage("  ")                                 # [MEM-6] final per-chunk VRAM bar

    if CONFIG["cleanup_after_chunk"]:
        mem.aggressive_cleanup()                               # [MEM-5] end-of-chunk full sweep

    return {"chunk_index": idx, "start_frame": start_frame,
            "num_frames": num_frames, "fps": fps, "path": chunk_path}


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
    Iterate chunks with OOM recovery and checkpoint-based resume.

    [MEM-6] Pre-chunk VRAM bar + OOM guard before every attempt.
    [MEM-2] MODEL_CACHE.evict_all() on OOM to guarantee clean state.

    OOM recovery tiers:
        retry 1 → reduce chunk to 0.75×
        retry 2 → reduce chunk to 0.50×, evict model cache
        retry 3 → evict all caches, drop to t4_ultra_safe sigmas
        > max   → record as failed, continue to next chunk
    """
    max_retries  = CONFIG["max_oom_retries"]
    auto_reduce  = CONFIG["auto_reduce_chunk_on_oom"]
    completed    = []
    current_chunks = list(chunks)
    i = 0

    while i < len(current_chunks):
        chunk_desc = current_chunks[i]
        idx = chunk_desc["chunk_index"]

        # ── Resume: skip already-completed chunks ─────────────────────────────
        if idx in checkpoint.get("completed_chunks", []):
            existing = os.path.join(CONFIG["workspace_dir"], "chunks", f"chunk_{idx:04d}.mp4")
            if os.path.exists(existing) and os.path.getsize(existing) > 0:
                print(f"  ↷ Chunk {idx:04d} already complete — skipping.")
                completed.append({"chunk_index": idx,
                                   "start_frame": chunk_desc["start_frame"],
                                   "num_frames":  chunk_desc["num_frames"],
                                   "fps": fps, "path": existing})
                i += 1
                continue

        retries = 0
        success = False
        current_num_frames = chunk_desc["num_frames"]

        while retries <= max_retries and not success:
            try:
                # [MEM-6] Print VRAM state at the start of every attempt
                print(f"\n{'='*62}")
                print(f"  [Chunk {idx+1:03d}]  attempt {retries+1}/{max_retries+1}"
                      f"  frames={current_num_frames}")
                print(f"{'='*62}")
                print_vram_usage("  ")

                # [MEM-6] Guard: require at least 2 GB free before attempting
                mem.pre_op_guard(f"chunk_{idx}_attempt_{retries}", required_gb=2.0)

                work_desc = {**chunk_desc, "num_frames": current_num_frames}
                result = generate_chunk(
                    chunk_desc=work_desc,
                    pos_cond=pos_cond, neg_cond=neg_cond,
                    loaded_image_tuple=loaded_image_tuple,
                    image_strength=image_strength, image_bypass=image_bypass,
                    width=width, height=height, fps=fps,
                    profile=profile, global_seed=global_seed,
                )
                completed.append(result)
                checkpoint["completed_chunks"].append(idx)
                save_checkpoint(checkpoint)
                success = True
                print(f"  ✓ Chunk {idx:04d} complete.")

            except torch.cuda.OutOfMemoryError as oom:
                retries += 1
                print(f"\n  {'!'*50}")
                print(f"  OOM  chunk={idx}  retry={retries}/{max_retries}")
                print(f"  {str(oom)[:200]}")
                print(f"  VRAM free now: {mem.gpu_free_gb():.2f} GB")

                # Tier 1: release DiT cache + aggressive cleanup
                release_dit_model()
                mem.aggressive_cleanup()

                # Tier 2: evict entire model cache on second OOM
                if retries >= 2 and MODEL_CACHE is not None:
                    print("  [MEM-2] Evicting full ModelCache (tier-2 OOM)")
                    MODEL_CACHE.evict_all()

                print_vram_usage("  ")                         # show state after cleanup

                if not auto_reduce or retries > max_retries:
                    if retries > max_retries:
                        print(f"  Generation stopped after {max_retries} OOM retries.")
                        print("  SUGGESTED: reduce CHUNK_FRAMES or switch to t4_ultra_safe.")
                        checkpoint["failed_chunks"].append(idx)
                        save_checkpoint(checkpoint)
                    break

                # Reduce chunk size: 0.75× → 0.5×
                reduction = 0.75 if retries == 1 else 0.5
                raw_reduced = max(9, int(current_num_frames * reduction))
                current_num_frames = normalize_ltx_frame_count(raw_reduced, fps)
                print(f"  Retrying with {current_num_frames} frames ({reduction*100:.0f}%)...")

            except Exception as e:
                print(f"\n  ERROR  {type(e).__name__}: {str(e)[:300]}")
                print(f"  {traceback.format_exc()[:600]}")
                checkpoint["failed_chunks"].append(idx)
                save_checkpoint(checkpoint)
                release_dit_model()
                mem.aggressive_cleanup()
                break

        i += 1

    return completed


# =============================================================================
# SECTION 20 — CHECKPOINT / RESUME SYSTEM
# =============================================================================

def _checkpoint_path() -> str:
    return os.path.join(CONFIG["workspace_dir"], "checkpoint.json")

def init_checkpoint(fps, total_frames, seed, width, height, job_id=None) -> Dict:
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
    checkpoint["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    Path(CONFIG["workspace_dir"]).mkdir(parents=True, exist_ok=True)
    tmp = _checkpoint_path() + ".tmp"
    with open(tmp, "w") as f:
        json.dump(checkpoint, f, indent=2)
    os.replace(tmp, _checkpoint_path())

def load_checkpoint() -> Optional[Dict]:
    p = _checkpoint_path()
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as e:
        print(f"  ⚠ Could not load checkpoint: {e}")
        return None

def get_or_create_checkpoint(fps, total_frames, seed, width, height) -> Dict:
    if CONFIG["resume"]:
        existing = load_checkpoint()
        if existing is not None:
            if (existing.get("fps") == fps
                    and existing.get("total_frames") == total_frames
                    and existing.get("resolution") == [width, height]):
                done = len(existing.get("completed_chunks", []))
                print(f"  ↷ Resuming: {done} chunks already complete.")
                return existing
            else:
                print("  ⚠ Checkpoint config mismatch — starting fresh.")
    cp = init_checkpoint(fps, total_frames, seed, width, height)
    save_checkpoint(cp)
    return cp

# =============================================================================
# SECTION 21 — VIDEO ASSEMBLY  (FFmpeg concat)
# =============================================================================

def assemble_chunks_to_video(completed_chunks: List[Dict], output_path: str, fps: int) -> bool:
    if not completed_chunks:
        print("  ✗ No completed chunks."); return False
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    sorted_chunks   = sorted(completed_chunks, key=lambda c: c["chunk_index"])
    concat_list_path = os.path.join(CONFIG["workspace_dir"], "concat_list.txt")
    with open(concat_list_path, "w") as f:
        for chunk in sorted_chunks:
            safe_path = chunk["path"].replace("'", "'\\''")
            f.write(f"file '{safe_path}'\n")
    print(f"\n  Assembling {len(sorted_chunks)} chunks → {output_path}")

    # Attempt 1: stream-copy (lossless, fastest)
    r = subprocess.run(
        ["ffmpeg","-y","-f","concat","-safe","0","-i",concat_list_path,"-c","copy",output_path],
        capture_output=True, text=True)
    if r.returncode == 0:
        print(f"  ✓ Assembly (stream-copy): {os.path.getsize(output_path)/(1024*1024):.1f} MB")
        return True
    print(f"  Stream-copy failed ({r.stderr.strip()[:120]}), re-encoding...")

    # Attempt 2: re-encode
    r2 = subprocess.run(
        ["ffmpeg","-y","-f","concat","-safe","0","-i",concat_list_path,
         "-vcodec","libx264","-pix_fmt","yuv420p","-crf","18","-preset","fast",
         "-acodec","aac","-b:a","192k",output_path],
        capture_output=True, text=True)
    if r2.returncode == 0:
        print(f"  ✓ Assembly (re-encode): {os.path.getsize(output_path)/(1024*1024):.1f} MB")
        return True
    print(f"  ✗ Assembly failed:\n{r2.stderr.strip()[:300]}")
    return False

# =============================================================================
# SECTION 22 — AUDIO SYNCHRONISATION
# =============================================================================

def assemble_video_with_audio(video_path, audio_path, output_path,
                                fps, total_frames, audio_start_seconds=0.0) -> bool:
    if audio_path is None or not os.path.exists(audio_path):
        if video_path != output_path:
            shutil.copy2(video_path, output_path)
        print(f"  ✓ Output (no external audio): {output_path}")
        return True
    video_duration = total_frames / fps
    print(f"  Muxing audio: start={audio_start_seconds:.3f}s dur={video_duration:.3f}s")
    r = subprocess.run(
        ["ffmpeg","-y","-i",video_path,
         "-ss",str(audio_start_seconds),"-t",str(video_duration),"-i",audio_path,
         "-map","0:v:0","-map","1:a:0","-c:v","copy","-c:a","aac","-b:a","192k",
         "-shortest",output_path],
        capture_output=True, text=True)
    if r.returncode == 0:
        print(f"  ✓ Audio sync: {os.path.getsize(output_path)/(1024*1024):.1f} MB")
        return True
    print(f"  ✗ Audio mux failed: {r.stderr.strip()[:200]}")
    shutil.copy2(video_path, output_path)
    return False


# =============================================================================
# SECTION 23 — FINAL VALIDATION
# =============================================================================

def validate_output_video(output_path: str, expected_frames: int, fps: int) -> bool:
    if not os.path.exists(output_path):
        print(f"  ✗ Output not found: {output_path}"); return False
    size_mb = os.path.getsize(output_path) / (1024*1024)
    try:
        r = subprocess.run(
            ["ffprobe","-v","quiet","-print_format","json",
             "-show_streams","-show_format",output_path],
            capture_output=True, text=True, check=True)
        info = json.loads(r.stdout)
        vs = next((s for s in info.get("streams",[]) if s.get("codec_type")=="video"), None)
        if vs is None:
            print("  ✗ No video stream in output."); return False
        nb_frames = int(vs.get("nb_frames", 0))
        duration  = float(info.get("format",{}).get("duration", 0))
        print(f"\n  Output validation:")
        print(f"    Size    : {size_mb:.1f} MB")
        print(f"    Codec   : {vs.get('codec_name','?')}")
        print(f"    Res     : {vs.get('width',0)}×{vs.get('height',0)}")
        print(f"    Frames  : {nb_frames} (expected ≈{expected_frames})")
        print(f"    Duration: {duration:.2f}s (expected ≈{expected_frames/fps:.2f}s)")
        ok = nb_frames > 0 and duration > 0
        print(f"  {'✓' if ok else '✗'} Validation {'passed' if ok else 'FAILED'}.")
        return ok
    except Exception as e:
        print(f"  ⚠ Validation error: {e} (file={size_mb:.1f} MB)")
        return True

# =============================================================================
# SECTION 24 — PREVIEW MODE
# =============================================================================

def generate_preview(image_path, prompt, fps, width, height, seed, profile,
                      preview_duration=3.0) -> Optional[str]:
    preview_frames = normalize_ltx_frame_count(round(preview_duration * fps), fps)
    preview_out    = os.path.join(CONFIG["output_dir"], "preview.mp4")
    print(f"\n  PREVIEW MODE: {preview_frames} frames ({preview_duration:.1f}s)")
    loaded_image, img_strength, img_bypass = load_input_image(image_path, width, height)
    pos_cond, neg_cond = build_text_conditioning(prompt, fps)
    preview_desc = {"chunk_index":0,"start_frame":0,"num_frames":preview_frames,
                    "fps":fps,"path":None}
    try:
        mem.cleanup()
        result = generate_chunk(
            chunk_desc=preview_desc, pos_cond=pos_cond, neg_cond=neg_cond,
            loaded_image_tuple=loaded_image, image_strength=img_strength,
            image_bypass=img_bypass, width=width, height=height,
            fps=fps, profile=profile, global_seed=seed)
        if result["path"] and os.path.exists(result["path"]):
            Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
            shutil.move(result["path"], preview_out)
            print(f"  ✓ Preview saved: {preview_out}")
            display_video_safe(preview_out)
        else:
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
    if not os.path.exists(video_path):
        print(f"  Video not found: {video_path}"); return
    size_mb = os.path.getsize(video_path) / (1024*1024)
    if size_mb > max_size_mb:
        print(f"  Video {size_mb:.1f} MB — too large for inline display.")
        print(f"  Download: files.download('{video_path}')")
        return
    from base64 import b64encode
    chunks_b64 = []
    with open(video_path, "rb") as f:
        while True:
            block = f.read(65536)
            if not block: break
            chunks_b64.append(block)
    video_b64 = b64encode(b"".join(chunks_b64)).decode()
    del chunks_b64
    display(HTML(f"""
    <video width=640 controls autoplay loop muted>
      <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
    </video>"""))
    del video_b64


# =============================================================================
# SECTION 26 — JOB REPORT
# =============================================================================

def write_job_report(output_path, total_frames, fps, width, height,
                      chunk_size, completed_chunks, failed_chunks,
                      generation_start_time, seed):
    elapsed = time.time() - generation_start_time
    report  = {
        "gpu":                   _GPU_INFO.get("device_name","unknown"),
        "vram_total_gb":         round(_GPU_INFO.get("vram_total_gb",0), 2),
        "torch_version":         torch.__version__,
        "cuda_version":          getattr(torch.version,"cuda","N/A"),
        "models":                {k: MODELS[k] for k in MODELS},
        "lora_strengths":        LORA_STRENGTHS,
        "lora_enabled":          LORA_ENABLED,
        "workflow": {
            "fps":       fps, "sampler": WORKFLOW_SAMPLER_PASS1,
            "scheduler": WORKFLOW_SCHEDULER, "steps": WORKFLOW_STEPS,
            "cfg":       WORKFLOW_CFG, "sigma_mode": CONFIG["sigma_preset_mode"],
        },
        "resolution":            f"{width}x{height}",
        "quality_mode":          CONFIG["quality_mode"],
        "fps":                   fps,
        "seed":                  seed,
        "requested_duration_s":  CONFIG["duration_seconds"],
        "actual_duration_s":     round(total_frames/fps, 3),
        "total_frames":          total_frames,
        "chunk_size_frames":     chunk_size,
        "chunks_completed":      len(completed_chunks),
        "chunks_failed":         len(failed_chunks),
        "failed_chunk_indices":  failed_chunks,
        "peak_gpu_memory_gb":    round(mem.gpu_peak_gb(), 3),
        "generation_time_s":     round(elapsed, 1),
        "generation_time_min":   round(elapsed/60, 2),
        "output_path":           output_path,
        "generated_at":          time.strftime("%Y-%m-%dT%H:%M:%S"),
        "improvements_applied":  ["MEM-1","MEM-2","MEM-3","MEM-4",
                                   "MEM-5","MEM-6","MEM-7","MEM-8"],
    }
    report_path = os.path.join(CONFIG["output_dir"], "job_report.json")
    Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Job report  : {report_path}")
    print(f"  Peak VRAM   : {report['peak_gpu_memory_gb']:.3f} GB")
    print(f"  Gen time    : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Completed   : {report['chunks_completed']} chunks")
    if failed_chunks:
        print(f"  Failed      : {failed_chunks}")

# =============================================================================
# SECTION 27 — FINAL CLEANUP
# =============================================================================

def cleanup_temp_files(completed_chunks: List[Dict], keep_chunks: bool = False):
    if keep_chunks:
        print("  Keeping temp chunks (KEEP_TEMP_CHUNKS=True)."); return
    removed = 0
    for chunk in completed_chunks:
        path = chunk.get("path")
        if path and os.path.exists(path):
            try:
                os.remove(path); removed += 1
            except Exception as e:
                print(f"  ⚠ Could not remove {path}: {e}")
    concat_list = os.path.join(CONFIG["workspace_dir"], "concat_list.txt")
    if os.path.exists(concat_list):
        os.remove(concat_list)
    print(f"  ✓ Removed {removed} temp chunk files.")

def final_memory_report():
    print("\n" + "=" * 62)
    print("FINAL MEMORY REPORT")
    print("=" * 62)
    mem.print_memory()
    print_vram_usage()
    print(f"  Peak GPU: {mem.gpu_peak_gb():.3f} GB")
    print("=" * 62)


# =============================================================================
# SECTION 28 — MAIN ENTRY POINT
# =============================================================================

def print_banner(total_frames, fps, duration_s, width, height, chunk_size, n_chunks):
    print("\n" + "=" * 62)
    print("  LTX-2.3 DIRECTOR 2.0 MV  —  Google Colab T4 Engine  v2.0")
    print("=" * 62)
    print(f"  GPU          : {_GPU_INFO['device_name']}")
    print(f"  VRAM         : {_GPU_INFO['vram_total_gb']:.1f} GB total")
    print_vram_usage("  ")
    print(f"  Resolution   : {width}×{height}")
    print(f"  FPS          : {fps}")
    print(f"  Duration     : {duration_s:.2f}s  ({total_frames} frames)")
    print(f"  Chunk size   : {chunk_size} frames  ({n_chunks} chunks est.)")
    print(f"  Quality mode : {CONFIG['quality_mode']}")
    print(f"  Sigma mode   : {CONFIG['sigma_preset_mode']}")
    print(f"  ModelCache   : {'ON' if CONFIG['use_model_cache'] else 'OFF'}")
    print(f"  FaceRestore  : {'ON' if CONFIG.get('face_restoration') else 'OFF'}")
    print(f"  Resume       : {CONFIG['resume']}")
    print("=" * 62)


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
) -> Optional[str]:
    """
    Complete LTX-2.3 Director 2.0 MV generation pipeline.

    New in v2.0:
      [MEM-3] auto_adjust_settings() applied at startup
      [MEM-7] cleanup_old_cache() run before generation
      [MEM-2] ModelCache evicted after generation loop
      [MEM-5] aggressive_cleanup after conditioning release
    """
    duration_s   = duration_seconds if duration_seconds is not None else CONFIG["duration_seconds"]
    fps          = fps    or CONFIG["fps"]
    width        = width  or CONFIG["width"]
    height       = height or CONFIG["height"]
    seed         = seed   if seed is not None else CONFIG["seed"]
    quality_mode = quality_mode or CONFIG["quality_mode"]

    # [MEM-3] Auto-adjust quality based on detected VRAM
    overrides = auto_adjust_settings()
    if overrides.get("quality_mode"):
        quality_mode = overrides["quality_mode"]
        CONFIG["quality_mode"] = quality_mode
    if "use_model_cache" in overrides:
        CONFIG["use_model_cache"] = overrides["use_model_cache"]

    generation_start = time.time()

    # [MEM-7] Clean stale workspace cache files before starting
    cache_dir = os.path.join(CONFIG["workspace_dir"], "cache")
    cleanup_old_cache(cache_dir, max_age_days=CONFIG.get("cache_max_age_days", 7))

    profile = select_profile(quality_mode)
    width, height = check_resolution_safety(width, height, quality_mode)

    if CONFIG["preview_mode"]:
        return generate_preview(image_path=image_path, prompt=prompt,
                                 fps=fps, width=width, height=height,
                                 seed=seed, profile=profile,
                                 preview_duration=CONFIG.get("preview_duration", 3.0))

    total_frames, actual_duration = calculate_timeline(duration_s, fps)
    chunk_size = estimate_chunk_size(width, height, fps, quality_mode)
    all_chunks = plan_chunks(total_frames, chunk_size, fps)

    print_banner(total_frames, fps, actual_duration, width, height, chunk_size, len(all_chunks))
    run_all_validations(image_path, audio_path, width, height, total_frames)

    setup_comfyui()
    import_custom_nodes()

    checkpoint = get_or_create_checkpoint(fps, total_frames, seed, width, height)

    loaded_image, img_strength, img_bypass = load_input_image(image_path, width, height)

    print("\n  Building text conditioning...")
    mem.cleanup()
    pos_cond, neg_cond = build_text_conditioning(prompt, fps)

    print(f"\n  Starting generation: {len(all_chunks)} chunks...")
    torch.cuda.reset_peak_memory_stats()

    completed_chunks = adaptive_chunk_generator(
        chunks=all_chunks, pos_cond=pos_cond, neg_cond=neg_cond,
        loaded_image_tuple=loaded_image,
        image_strength=img_strength, image_bypass=img_bypass,
        width=width, height=height, fps=fps,
        profile=profile, global_seed=seed, checkpoint=checkpoint,
    )

    # [MEM-5] Full cleanup after generation loop
    del pos_cond, neg_cond, loaded_image
    _CONDITIONING_CACHE.clear()
    LORA_REGISTRY.clear()
    if MODEL_CACHE is not None:
        MODEL_CACHE.evict_all()                                # [MEM-2] evict after all chunks
    mem.aggressive_cleanup()

    if not completed_chunks:
        print("\n  ✗ No chunks completed. Generation aborted.")
        return None

    Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
    assembled_path = os.path.join(CONFIG["output_dir"], "_assembled_no_audio.mp4")
    if not assemble_chunks_to_video(completed_chunks, assembled_path, fps):
        print("\n  ✗ Video assembly failed.")
        return None

    final_output = os.path.join(CONFIG["output_dir"], CONFIG["output_filename"])
    assemble_video_with_audio(video_path=assembled_path, audio_path=audio_path,
                               output_path=final_output, fps=fps,
                               total_frames=total_frames, audio_start_seconds=0.0)

    if os.path.exists(assembled_path) and assembled_path != final_output:
        os.remove(assembled_path)

    validate_output_video(final_output, total_frames, fps)

    write_job_report(output_path=final_output, total_frames=total_frames,
                     fps=fps, width=width, height=height, chunk_size=chunk_size,
                     completed_chunks=completed_chunks,
                     failed_chunks=checkpoint.get("failed_chunks", []),
                     generation_start_time=generation_start, seed=seed)

    cleanup_temp_files(completed_chunks, keep_chunks=CONFIG["keep_temp_chunks"])

    mem.aggressive_cleanup()
    final_memory_report()

    print(f"\n{'='*62}")
    print(f"✓ GENERATION COMPLETE")
    print(f"  Output: {final_output}")
    print(f"{'='*62}\n")
    return final_output


# =============================================================================
# SECTION 29 — COLAB CELL RUNNER
# =============================================================================
# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Install environment (run once per runtime)
# ─────────────────────────────────────────────────────────────────────────────
# install_environment()
#
# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Download models (run once per runtime)
# ─────────────────────────────────────────────────────────────────────────────
# download_all_models()
#
# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Upload reference image and audio
# ─────────────────────────────────────────────────────────────────────────────
# from google.colab import files
# import shutil, os
# os.makedirs('/content/ComfyUI/input', exist_ok=True)
# uploaded = files.upload()
# for fname in uploaded:
#     shutil.move(f'/content/ComfyUI/{fname}', f'/content/ComfyUI/input/{fname}')
#
# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Configure and generate
# ─────────────────────────────────────────────────────────────────────────────
# Adjust @param widgets in SECTION 1, then run:
#
# output = generate_director_mv(
#     image_path = IMAGE_PATH,
#     audio_path = AUDIO_PATH,
#     prompt     = GLOBAL_PROMPT,
# )
# if output:
#     from google.colab import files
#     files.download(output)
#
# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — Memory diagnostic (run anytime to check VRAM / RAM)
# ─────────────────────────────────────────────────────────────────────────────
# mem.print_memory()
# print_vram_usage()
# print(auto_adjust_settings())
#
# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — Force-evict model cache (run if you hit OOM between sessions)
# ─────────────────────────────────────────────────────────────────────────────
# if MODEL_CACHE: MODEL_CACHE.evict_all()
# release_dit_model()
# mem.aggressive_cleanup()
# print_vram_usage()

# =============================================================================
# SECTION 30 — DIRECT EXECUTION GUARD
# =============================================================================

if __name__ == "__main__":
    print("\nRunning LTX23_Director_2_0_MV_Colab_T4_FIXED.py directly...")
    print("Designed for Google Colab. Running setup steps...\n")

    install_environment()
    download_all_models()
    setup_comfyui()
    import_custom_nodes()

    output = generate_director_mv(
        image_path       = IMAGE_PATH,
        audio_path       = AUDIO_PATH,
        prompt           = GLOBAL_PROMPT,
        duration_seconds = CONFIG["duration_seconds"],
        fps              = CONFIG["fps"],
        width            = CONFIG["width"],
        height           = CONFIG["height"],
        seed             = CONFIG["seed"],
        quality_mode     = CONFIG["quality_mode"],
    )

    if output:
        print(f"\nGeneration complete: {output}")
    else:
        print("\nGeneration failed — check logs above.")
