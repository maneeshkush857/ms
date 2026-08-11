# -*- coding: utf-8 -*-
# =============================================================================
# LTX23_Director_2_0_MV_MultiScene.py
#
# LTX-2.3 Director 2.0 — Multi-Scene Infinite Flow Engine
# Crash-Resilient | Scene-JSON Driven | T4-Safe | Auto-Resume
#
# MERGED FROM:
#   ① LTX23_Director_2_0_MV_Colab_T4_FIXED.py  — full Director 2.0 pipeline
#   ② ltx2_ti2v_distilled.py                    — multi-scene loop + resilience
#
# KEY ADDITIONS vs FIXED.py:
#   • SCENE_JSON storyboard  — N scenes × M shots, prompts, camera, emotion
#   • Per-scene checkpoint   — survives ANY crash, auto-resumes from last scene
#   • ModelCache             — keeps DiT in VRAM between scenes (no reload)
#   • Adaptive anchor        — best-frame extraction with character consistency
#   • Adaptive overlap       — overlap scales with shot motion_intensity
#   • OOM retry loop         — 3 attempts per scene, VRAM recovery between tries
#   • Face restoration pass  — YOLO + cv2.detailEnhance per scene
#   • Optical flow morphing  — smooth inter-scene transitions
#   • Shot variations        — generate N seeds, keep best quality score
#   • GPU NVENC encoding     — auto-detected, falls back to libx264
#   • SRT subtitle export    — from SCENE_JSON dialogue_with_timing
#   • LLM prompt expansion   — OpenAI / Gemini API for richer prompts
#   • Background stitch      — partial preview assembled in ThreadPoolExecutor
#
# USAGE (Colab):
#   1. Run CELL 1 (install), CELL 2 (download models) once per runtime.
#   2. Edit SCENE_JSON and @param settings in Section 1.
#   3. Run CELL 3 (generate). Output in /content/ltx23_output/
#   4. On crash: re-run CELL 3 — auto-resumes from last completed scene.
# =============================================================================


# =============================================================================
# SECTION 1 — CONFIGURATION (@param Colab form widgets)
# =============================================================================

# ── Input files ───────────────────────────────────────────────────────────────
IMAGE_PATH = "/content/ComfyUI/input/reference.png"   # @param {type:"string"}
AUDIO_PATH = "/content/ComfyUI/input/audio.mp3"       # @param {type:"string"}

# ── Multi-scene settings ──────────────────────────────────────────────────────
PROJECT_NAME        = "LTX23_MV_MultiScene"  # @param {type:"string"}
SCENE_DURATION_S    = 4.0                    # @param {type:"number"} seconds per scene/shot
FPS                 = 24                     # @param [8,12,16,24,25,30] {type:"raw"}
OUTPUT_WIDTH        = 1280                   # @param {type:"integer"}
OUTPUT_HEIGHT       = 720                    # @param {type:"integer"}
OUTPUT_FILENAME     = "LTX23_MultiScene_Final.mp4"  # @param {type:"string"}

# ── Seed ─────────────────────────────────────────────────────────────────────
SEED        = 123456  # @param {type:"integer"}
RANDOM_SEED = False   # @param {type:"boolean"}

# ── Quality / Memory profile ──────────────────────────────────────────────────
# t4_safe      → 48-frame chunks, 832x480, no CLIP load      ✅ Most stable
# t4_balanced  → 73-frame chunks, 1280x720, moderate offload ⚠️ Moderate
# t4_aggressive→ 97-frame chunks, 1280x720, minimal offload  ❌ OOM risk
QUALITY_MODE = "t4_safe"  # @param ["t4_safe","t4_balanced","t4_aggressive"]

# ── Anchor / overlap ─────────────────────────────────────────────────────────
ANCHOR_STRENGTH_HIGH    = 0.85  # @param {type:"number"}
ANCHOR_STRENGTH_LOW     = 0.70  # @param {type:"number"}
USE_ADAPTIVE_STRENGTH   = True  # @param {type:"boolean"}
USE_ADAPTIVE_OVERLAP    = True  # @param {type:"boolean"}
OVERLAP_FRAMES          = 8     # @param {type:"integer"}

# ── Model cache ───────────────────────────────────────────────────────────────
USE_MODEL_CACHE         = True  # @param {type:"boolean"}

# ── LoRA strengths ────────────────────────────────────────────────────────────
LORA_STRENGTH_DISTILLED  = 0.4   # @param {type:"slider",min:0.0,max:2.0,step:0.05}
LORA_STRENGTH_OMNINFT    = 0.6   # @param {type:"slider",min:0.0,max:2.0,step:0.05}
LORA_STRENGTH_TRANSITION = 0.7   # @param {type:"slider",min:0.0,max:2.0,step:0.05}
LORA_STRENGTH_MVCAMERA   = 0.9   # @param {type:"slider",min:0.0,max:2.0,step:0.05}
ENABLE_LORA_DISTILLED    = True  # @param {type:"boolean"}
ENABLE_LORA_OMNINFT      = True  # @param {type:"boolean"}
ENABLE_LORA_TRANSITION   = False # @param {type:"boolean"}
ENABLE_LORA_MVCAMERA     = False # @param {type:"boolean"}


# ── Sampler ───────────────────────────────────────────────────────────────────
SAMPLER_STEPS     = 8              # @param {type:"slider",min:1,max:30,step:1}
SAMPLER_CFG       = 1.0            # @param {type:"slider",min:1.0,max:10.0,step:0.5}
SAMPLER_NAME      = "euler"        # @param ["euler","euler_ancestral","dpm_2","heun"]
SCHEDULER_NAME    = "linear_quadratic"  # @param ["linear_quadratic","karras","simple"]
IMG_COMPRESSION   = 18             # @param {type:"slider",min:1,max:95,step:1}

# ── OOM / crash protection ────────────────────────────────────────────────────
AUTO_REDUCE_CHUNK_ON_OOM  = True   # @param {type:"boolean"}
MAX_OOM_RETRIES           = 3      # @param {type:"integer"}
GPU_SAFETY_MARGIN_GB      = 1.5    # @param {type:"slider",min:0.5,max:4.0,step:0.25}
MAX_SCENE_RETRIES         = 3      # @param {type:"integer"} retries per scene
ALLOW_AUTO_DOWNGRADE      = True   # @param {type:"boolean"}

# ── Resume ────────────────────────────────────────────────────────────────────
RESUME = True  # @param {type:"boolean"}

# ── Post-processing ───────────────────────────────────────────────────────────
FACE_RESTORATION      = True        # @param {type:"boolean"}
OPTICAL_FLOW_STITCH   = True        # @param {type:"boolean"}
TRANSITION_TYPE       = "crossfade" # @param ["crossfade","fade_black","none"]
GENERATE_SUBTITLES    = True        # @param {type:"boolean"}

# ── Shot variations ───────────────────────────────────────────────────────────
GENERATE_SHOT_VARIATIONS = False  # @param {type:"boolean"}
NUM_VARIATIONS           = 2      # @param {type:"integer"}

# ── LLM prompt expansion ──────────────────────────────────────────────────────
LLM_EXPANSION  = False            # @param {type:"boolean"}
LLM_PROVIDER   = "openai"        # @param ["openai","gemini"]
LLM_API_KEY    = ""              # @param {type:"string"}

# ── Memory / logging ─────────────────────────────────────────────────────────
ENABLE_MEMORY_LOGGING = True   # @param {type:"boolean"}
CLEANUP_AFTER_CHUNK   = True   # @param {type:"boolean"}
CLEANUP_AFTER_STAGE   = True   # @param {type:"boolean"}
KEEP_TEMP_CHUNKS      = False  # @param {type:"boolean"}
CLEANUP_TEMP_FILES    = True   # @param {type:"boolean"}

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKSPACE_DIR = "/content/ltx23_workspace"  # @param {type:"string"}
OUTPUT_DIR    = "/content/ltx23_output"     # @param {type:"string"}
COMFYUI_DIR   = "/content/ComfyUI"          # @param {type:"string"}

# ── Character sheet (for anchor blending) ────────────────────────────────────
USE_CHARACTER_SHEETS  = True   # @param {type:"boolean"}
CHARACTER_SHEET_PATH  = ""     # @param {type:"string"} leave blank to auto-use IMAGE_PATH


# ── Resolve CONFIG ────────────────────────────────────────────────────────────
import random as _random

IMAGE_PATH = IMAGE_PATH.strip() or None
AUDIO_PATH = AUDIO_PATH.strip() or None

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

CONFIG = {
    "fps": FPS, "width": OUTPUT_WIDTH, "height": OUTPUT_HEIGHT,
    "seed": SEED, "quality_mode": QUALITY_MODE,
    "auto_chunk_size": True, "chunk_frames": 48,
    "auto_reduce_chunk_on_oom": AUTO_REDUCE_CHUNK_ON_OOM,
    "max_oom_retries": MAX_OOM_RETRIES,
    "resume": RESUME,
    "gpu_safety_margin_gb": GPU_SAFETY_MARGIN_GB,
    "enable_memory_logging": ENABLE_MEMORY_LOGGING,
    "cleanup_after_chunk": CLEANUP_AFTER_CHUNK,
    "cleanup_after_stage": CLEANUP_AFTER_STAGE,
    "keep_temp_chunks": KEEP_TEMP_CHUNKS,
    "cleanup_temp_files": CLEANUP_TEMP_FILES,
    "allow_auto_downgrade": ALLOW_AUTO_DOWNGRADE,
    "preview_mode": False, "preview_duration": 3,
    "preview_width": 832, "preview_height": 480,
    "workspace_dir": WORKSPACE_DIR,
    "output_dir": OUTPUT_DIR,
    "output_filename": OUTPUT_FILENAME,
    "comfyui_dir": COMFYUI_DIR,
}


# ── Model registry ────────────────────────────────────────────────────────────
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
LORA_STRENGTHS = {k: _LORA_STRENGTHS_OVERRIDE.get(k, 0.5) for k in
                  ["lora_distilled","lora_omninft","lora_transition","lora_mvcamera"]}
LORA_ENABLED   = {k: _LORA_ENABLED_OVERRIDE.get(k, True) for k in LORA_STRENGTHS}

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


T4_PROFILES = {
    "t4_safe": {
        "chunk_frames": 48, "generation_width": 832, "generation_height": 480,
        "offload_models": True, "skip_director": True,
        "img_compression": 33, "longer_edge": 848,
        "description": "Conservative: 48-frame chunks, 832x480, no CLIP",
    },
    "t4_balanced": {
        "chunk_frames": 73, "generation_width": 1280, "generation_height": 720,
        "offload_models": True, "img_compression": 18, "longer_edge": 1312,
        "description": "Balanced: 73-frame chunks, 1280x720",
    },
    "t4_aggressive": {
        "chunk_frames": 97, "generation_width": 1280, "generation_height": 720,
        "offload_models": False, "img_compression": 18, "longer_edge": 1312,
        "description": "Aggressive: 97-frame chunks, OOM risk",
    },
}

WORKFLOW_FPS            = CONFIG["fps"]
WORKFLOW_CFG            = SAMPLER_CFG
WORKFLOW_SAMPLER_PASS1  = SAMPLER_NAME
WORKFLOW_SAMPLER_PASS2  = SAMPLER_NAME
WORKFLOW_SCHEDULER      = SCHEDULER_NAME
WORKFLOW_STEPS          = SAMPLER_STEPS
WORKFLOW_STEPS_PASS2    = 4
WORKFLOW_DENOISE_PASS2  = 0.42
WORKFLOW_IMG_COMPRESSION = IMG_COMPRESSION

# Default global prompt (overridden per-scene by SCENE_JSON)
GLOBAL_PROMPT = (
    "Create a highly realistic cinematic AI music video using the provided reference image. "
    "Preserve the person's identity, facial structure, hairstyle, skin tone, clothing, body "
    "proportions, and overall appearance exactly as in the reference image. The singer must "
    "remain fully recognizable throughout the entire video with absolutely no identity drift. "
    "drclipz, Aggressive cinematic music video camera. Fast push-in, fast pull-back, energetic "
    "handheld movement, rhythmic tracking shots, dynamic low-angle hero shots, occasional "
    "close-ups on emotional lyrics, subtle orbit around the singer, cinematic motion blur. "
    "Photorealistic, blockbuster-quality AI music video, ultra-high facial fidelity."
)


# =============================================================================
# SECTION 2 — SCENE JSON  (Edit this to define your multi-scene video)
# =============================================================================
# Each entry in "shots" becomes one generated video segment.
# camera_movement maps to Director 2.0 LoRA keys (dolly_forward etc.)
# Prompts are auto-enriched by build_shot_prompt() below.
# =============================================================================

SCENE_JSON = {
    "scene_id": "music_video_01",
    "project_name": PROJECT_NAME,
    "video_style": (
        "Photorealistic cinematic music video, ultra-high fidelity, "
        "blockbuster concert performance, premium production quality"
    ),
    "environment": {
        "location":      "Sold-out stadium concert stage with massive LED screen backdrop",
        "time":          "Night performance, spotlight-lit stage",
        "weather":       "Indoor, controlled",
        "mood":          "Explosive energy, charismatic performance",
        "lighting":      "Dynamic concert lighting, neon rims, volumetric haze, dramatic key light",
        "color_palette": "Electric blues, hot pinks, deep blacks, gold highlights",
    },
    "main_characters": [
        {
            "name": "Singer",
            "desc": "World-class pop/hip-hop singer performing live",
            "detailed_appearance": {
                "face":        "Expressive, charismatic, intense performance energy",
                "hair":        "Styled naturally, matches reference image exactly",
                "clothing":    "Stage performance outfit, matches reference image exactly",
                "build":       "Athletic, confident stage presence",
                "skin_tone":   "Matches reference image exactly",
                "accessories": "Wireless microphone, stage lighting"
            },
            "lora_path": None,
        }
    ],
    "story_action": {
        "shots": [
            {
                "time": "0-4s",
                "camera": "Wide establishing shot, slow dolly forward",
                "camera_movement": "dolly_forward",
                "motion_intensity": 0.5,
                "action": "Singer enters from stage right into spotlight. Crowd erupts. Wide shot reveals the full stadium scale.",
                "character_focus": "Singer + crowd",
                "emotion": "energetic_entrance",
                "visual_effects": "Confetti burst, spotlight sweep, crowd waving",
                "prompt_override": ""
            },
            {
                "time": "4-8s",
                "camera": "Medium tracking shot, low-angle hero",
                "camera_movement": "low_angle_hero",
                "motion_intensity": 0.7,
                "action": "Singer moves to front of stage, pointing at crowd, commanding the room. Intense eye contact with camera.",
                "character_focus": "Singer face and body",
                "emotion": "powerful_dominance",
                "visual_effects": "Dynamic lighting shift, heat haze, lens flare",
                "prompt_override": ""
            },
            {
                "time": "8-12s",
                "camera": "Extreme close-up on face, slight handheld",
                "camera_movement": "static_intense",
                "motion_intensity": 0.3,
                "action": "Extreme close-up of singer's face, eyes burning with passion. Lip-sync perfectly matching the beat drop.",
                "character_focus": "Singer face",
                "emotion": "intense_passionate",
                "visual_effects": "Shallow depth of field, rim light, subtle camera shake",
                "prompt_override": ""
            },
            {
                "time": "12-16s",
                "camera": "Fast pull-back to wide, then push-in",
                "camera_movement": "dolly_forward",
                "motion_intensity": 0.8,
                "action": "Fast camera pull-back reveals full stage production. LED walls explode with visuals. Singer raises arms.",
                "character_focus": "Singer + stage",
                "emotion": "triumphant_climax",
                "visual_effects": "Pyrotechnic bursts, LED explosion, crowd reaction",
                "prompt_override": ""
            },
            {
                "time": "16-20s",
                "camera": "360 orbit, starting left",
                "camera_movement": "orbit_left",
                "motion_intensity": 0.6,
                "action": "Camera orbits around singer as they perform center stage. Energy builds toward chorus end.",
                "character_focus": "Singer",
                "emotion": "building_intensity",
                "visual_effects": "Motion blur on orbit, dynamic lighting chase",
                "prompt_override": ""
            },
            {
                "time": "20-24s",
                "camera": "Wide aerial-style tilt down",
                "camera_movement": "tilt_down",
                "motion_intensity": 0.4,
                "action": "Dramatic tilt-down from overhead angle. Singer is small against massive crowd. Scale of performance revealed.",
                "character_focus": "Singer + full venue",
                "emotion": "awe_inspiring_scale",
                "visual_effects": "Deep depth, volumetric haze, crowd lighters",
                "prompt_override": ""
            },
            {
                "time": "24-28s",
                "camera": "Close tracking shot from side",
                "camera_movement": "dolly_right",
                "motion_intensity": 0.5,
                "action": "Side-angle tracking follows singer moving across stage. Background dancers in formation.",
                "character_focus": "Singer side profile",
                "emotion": "confident_swagger",
                "visual_effects": "Neon side lighting, background blur, depth",
                "prompt_override": ""
            },
            {
                "time": "28-32s",
                "camera": "Push in fast on face for final close-up",
                "camera_movement": "zoom_in_fast",
                "motion_intensity": 0.9,
                "action": "Final aggressive camera push-in to singer's face. Maximum emotion. Last lines of the verse delivered.",
                "character_focus": "Singer face",
                "emotion": "maximum_intensity_finale",
                "visual_effects": "Rapid light strobes, emotional peak, mic drop energy",
                "prompt_override": ""
            },
        ]
    },
    "dialogue_with_timing": [
        {"time": 3,  "character": "Singer", "dialogue": "(verse 1 opening line)",
         "emotion": "energetic", "lip_sync_emphasis": "high"},
        {"time": 8,  "character": "Singer", "dialogue": "(pre-chorus build)",
         "emotion": "intense",   "lip_sync_emphasis": "high"},
        {"time": 16, "character": "Singer", "dialogue": "(chorus peak)",
         "emotion": "triumphant","lip_sync_emphasis": "high"},
        {"time": 24, "character": "Singer", "dialogue": "(verse 2 opening)",
         "emotion": "confident", "lip_sync_emphasis": "high"},
        {"time": 28, "character": "Singer", "dialogue": "(final hook)",
         "emotion": "passionate","lip_sync_emphasis": "high"},
    ],
    "audio": {
        "background_music": "High-energy pop/hip-hop with hard beats, synthesizer drops, bass",
        "environment_sfx":  "Stadium crowd roar, echo, reverb",
        "voice_processing": "Slight reverb, live concert mix, compressed vocals"
    },
}


# =============================================================================
# SECTION 3 — IMPORTS & CUDA DETECTION
# =============================================================================

import os, sys, gc, json, time, shutil, hashlib, subprocess, traceback
import math, asyncio, threading, concurrent.futures, warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from functools import lru_cache

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore")

import torch
import numpy as np
import cv2
from PIL import Image

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

from IPython.display import display, HTML, Image as IPImage, clear_output

try:
    import ctypes
    _LIBC = ctypes.CDLL("libc.so.6")
except Exception:
    _LIBC = None

def _malloc_trim():
    if _LIBC is not None:
        try: _LIBC.malloc_trim(0)
        except Exception: pass

def detect_gpu() -> Dict:
    info = {
        "available": torch.cuda.is_available(),
        "device_name": "N/A", "vram_total_gb": 0.0,
        "vram_free_gb": 0.0, "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", "N/A"),
    }
    if not info["available"]: return info
    info["device_name"] = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    info["vram_total_gb"] = props.total_memory / (1024**3)
    free, _ = torch.cuda.mem_get_info(0)
    info["vram_free_gb"] = free / (1024**3)
    return info

_GPU_INFO = detect_gpu()
print(f"PyTorch  : {_GPU_INFO['torch_version']}")
print(f"CUDA     : {_GPU_INFO['cuda_version']}")
print(f"GPU      : {_GPU_INFO['device_name']}")
print(f"VRAM     : {_GPU_INFO['vram_total_gb']:.1f} GB  ({_GPU_INFO['vram_free_gb']:.1f} GB free)")

if not _GPU_INFO["available"]:
    raise RuntimeError(
        "ERROR: No CUDA GPU detected.\n"
        "In Colab: Runtime → Change runtime type → T4 GPU"
    )

DEVICE = torch.device("cuda")


# =============================================================================
# SECTION 4 — MEMORY MANAGER
# =============================================================================

class LTXMemoryManager:
    """VRAM/RAM tracking and cleanup — T4-safe, crash-aware."""

    def __init__(self, safety_margin_gb: float = 1.5, enable_logging: bool = True):
        self.safety_margin_gb = safety_margin_gb
        self.enable_logging   = enable_logging
        self._peak_allocated  = 0.0
        self._chunk_info: Dict = {}
        torch.cuda.reset_peak_memory_stats()

    def gpu_allocated_gb(self) -> float: return torch.cuda.memory_allocated() / (1024**3)
    def gpu_reserved_gb(self)  -> float: return torch.cuda.memory_reserved()  / (1024**3)
    def gpu_free_gb(self)      -> float:
        free, _ = torch.cuda.mem_get_info(0); return free / (1024**3)
    def gpu_total_gb(self)     -> float:
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    def gpu_peak_gb(self)      -> float:
        p = torch.cuda.max_memory_allocated() / (1024**3)
        if p > self._peak_allocated: self._peak_allocated = p
        return self._peak_allocated
    def cpu_used_gb(self)      -> float:
        return psutil.Process().memory_info().rss / (1024**3) if _HAS_PSUTIL else 0.0
    def cpu_available_gb(self) -> float:
        return psutil.virtual_memory().available / (1024**3) if _HAS_PSUTIL else 8.0
    def is_vram_safe(self)     -> bool: return self.gpu_free_gb() > self.safety_margin_gb
    def is_ram_safe(self, required_gb: float = 2.0) -> bool:
        return self.cpu_available_gb() > required_gb

    def soft_cleanup(self):
        gc.collect(); torch.cuda.empty_cache()

    def cleanup(self):
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.ipc_collect()

    def aggressive_cleanup(self):
        for _ in range(3): gc.collect()
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        torch.cuda.ipc_collect(); torch.cuda.reset_peak_memory_stats()
        gc.collect(); _malloc_trim()

    def ram_cleanup(self):
        for _ in range(3): gc.collect()
        _malloc_trim()
        if self.enable_logging:
            print(f"  [mem] RAM cleanup. Available: {self.cpu_available_gb():.2f} GB")

    def release_model(self, model, name: str = "model"):
        if model is None: return
        try:
            if hasattr(model, "to"): model.to("cpu")
        except Exception: pass
        del model; self.cleanup()

    def safe_model_unload(self, model, name: str = "model"):
        if self.enable_logging:
            print(f"  [mem] Unloading {name}  (free: {self.gpu_free_gb():.2f} GB)")
        self.release_model(model, name)
        if self.enable_logging:
            print(f"  [mem] Unloaded  {name}  (free: {self.gpu_free_gb():.2f} GB)")

    def memory_report(self, prefix: str = "") -> str:
        return (f"{prefix}GPU alloc={self.gpu_allocated_gb():.2f}GB "
                f"free={self.gpu_free_gb():.2f}GB peak={self.gpu_peak_gb():.2f}GB | "
                f"RAM used={self.cpu_used_gb():.2f}GB avail={self.cpu_available_gb():.2f}GB")

    def print_memory(self, prefix: str = ""):
        print(self.memory_report(prefix))

    def warn_if_low(self):
        if self.gpu_free_gb() < self.safety_margin_gb:
            print(f"  WARNING: GPU below safety threshold ({self.gpu_free_gb():.2f} GB). Cleaning up.")
            self.aggressive_cleanup()

    def estimate_frame_ram_gb(self, n: int, h: int, w: int) -> float:
        return n * h * w * 3 * 4 / (1024**3)

    def set_chunk_info(self, index: int, frames: int, w: int, h: int):
        self._chunk_info = {"index": index, "frames": frames, "resolution": f"{w}x{h}"}


mem = LTXMemoryManager(
    safety_margin_gb=CONFIG["gpu_safety_margin_gb"],
    enable_logging=CONFIG["enable_memory_logging"],
)


# =============================================================================
# SECTION 5 — MODEL CACHE  (keeps DiT in VRAM between scenes)
# =============================================================================

class MultiSceneModelCache:
    """
    Keeps heavy models resident in VRAM between scene shots to avoid
    re-loading the 12-14 GB DiT GGUF for every single shot.

    Thread-safe via RLock. Evict between full movies, not between shots.
    Call evict_all() only when you want to fully release VRAM.
    """

    def __init__(self):
        self._dit        = None
        self._video_vae  = None
        self._audio_vae  = None
        self._upscaler   = None
        self._lock       = threading.RLock()

    def get_dit(self, loader_fn, force_reload: bool = False):
        with self._lock:
            if self._dit is None or force_reload:
                print("  [cache] Loading DiT into model cache...")
                self._dit = loader_fn()
            else:
                print("  [cache] DiT from cache (no reload)")
            return self._dit

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

    def evict_dit(self):
        """Release only the DiT (e.g. before decode to free VRAM)."""
        with self._lock:
            if self._dit is not None:
                try:
                    if hasattr(self._dit, "to"): self._dit.to("cpu")
                except Exception: pass
                del self._dit
                self._dit = None
                mem.aggressive_cleanup()
                print("  [cache] DiT evicted.")

    def evict_all(self):
        with self._lock:
            for attr in ["_dit","_video_vae","_audio_vae","_upscaler"]:
                obj = getattr(self, attr, None)
                if obj is not None:
                    try:
                        if hasattr(obj, "to"): obj.to("cpu")
                    except Exception: pass
                    del obj
                    setattr(self, attr, None)
        mem.aggressive_cleanup()
        print("  [cache] All models evicted.")

    @property
    def dit_loaded(self)       -> bool: return self._dit is not None
    @property
    def video_vae_loaded(self) -> bool: return self._video_vae is not None
    @property
    def audio_vae_loaded(self) -> bool: return self._audio_vae is not None


_MODEL_CACHE = MultiSceneModelCache() if USE_MODEL_CACHE else None

# Module-level DiT fallback for non-cached path (matches FIXED.py pattern)
_DIT_MODEL_CACHE = None


# =============================================================================
# SECTION 6 — ENVIRONMENT INSTALLATION
# =============================================================================

def install_environment():
    print("=" * 60)
    print("[1/5] Installing Python packages...")
    _run("pip install -q torch torchvision torchaudio", "torch")
    _run("pip install -q torchsde einops diffusers accelerate", "diffusers")
    _run("pip install -q av spandrel albumentations onnx opencv-python onnxruntime", "vision")
    _run("pip install -q psutil nest_asyncio requests moviepy tqdm ipywidgets", "utilities")
    print("\n[2/5] Installing aria2 + ffmpeg...")
    _run("apt-get -y install -qq aria2 ffmpeg", "apt packages")
    print("\n[3/5] Cloning ComfyUI...")
    comfyui_dir = CONFIG["comfyui_dir"]
    if not os.path.exists(comfyui_dir):
        _run(f"git clone -q https://github.com/comfyanonymous/ComfyUI {comfyui_dir}", "ComfyUI")
    else:
        print("  ComfyUI already present.")
    _run(f"pip install -q -r {comfyui_dir}/requirements.txt", "ComfyUI requirements")
    print("\n[4/5] Installing custom nodes...")
    _install_custom_nodes()
    print("\n[5/5] Creating directories...")
    for d in ["chunks","frames","audio","final","logs","scenes"]:
        Path(f"{CONFIG['workspace_dir']}/{d}").mkdir(parents=True, exist_ok=True)
    Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(f"{comfyui_dir}/input").mkdir(parents=True, exist_ok=True)
    print("\n✓ Environment setup complete.")

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
            print(f"  ✓ {name} (present)")
        req = os.path.join(dest, "requirements.txt")
        if os.path.exists(req):
            _run(f"pip install -q -r {req}", f"  req  {name}")


# =============================================================================
# SECTION 7 — COMFYUI SETUP & NODE LOADING
# =============================================================================

_NODES_LOADED = False

def setup_comfyui():
    comfyui_dir = CONFIG["comfyui_dir"]
    if comfyui_dir not in sys.path:
        sys.path.insert(0, comfyui_dir)
    print(f"  ComfyUI path: {comfyui_dir}")

def import_custom_nodes():
    global _NODES_LOADED
    if _NODES_LOADED: return
    import nest_asyncio
    nest_asyncio.apply()
    try:
        from aiohttp import web
        from server import PromptServer
        if not hasattr(PromptServer, "instance") or PromptServer.instance is None:
            PromptServer.instance = PromptServer(asyncio.new_event_loop())
    except Exception: pass
    try:
        import kornia.geometry.transform.pyramid as _kpyr
        if not hasattr(_kpyr, "pad"):
            import torch.nn.functional as F
            _kpyr.pad = F.pad
    except Exception: pass
    from nodes import init_builtin_extra_nodes, init_external_custom_nodes
    async def _loader():
        failed = await init_builtin_extra_nodes()
        await init_external_custom_nodes()
        if failed:
            for n in failed: print(f"  WARNING: node failed: {n}")
    try:
        asyncio.run(_loader())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(_loader())
    _NODES_LOADED = True
    print("  ✓ Custom nodes loaded.")

def get_node(name: str):
    from nodes import NODE_CLASS_MAPPINGS
    if name not in NODE_CLASS_MAPPINGS:
        raise KeyError(f"ComfyUI node '{name}' not found. Ensure the custom node is installed.")
    return NODE_CLASS_MAPPINGS[name]()

def get_node_cls(name: str):
    from nodes import NODE_CLASS_MAPPINGS
    if name not in NODE_CLASS_MAPPINGS:
        raise KeyError(f"ComfyUI node class '{name}' not found.")
    return NODE_CLASS_MAPPINGS[name]

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    try: return obj[index]
    except KeyError: return obj["result"][index]


# =============================================================================
# SECTION 8 — MODEL DOWNLOAD & VALIDATION
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
    cmd = ["aria2c","--console-log-level=error","-c",
           "-x","16","-s","16","-k","1M",
           "--summary-interval=0","--quiet",
           "-d",dest_dir,"-o",filename,url]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("done"); return filename
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {e.stderr.strip()[:200]}"); return None

def download_all_models(skip_loras: bool = False):
    print("\n  Downloading models...")
    keys = [k for k in DOWNLOAD_URLS if not (skip_loras and k.startswith("lora_"))]
    for key in keys:
        model_download(DOWNLOAD_URLS[key], MODEL_DEST_DIRS[key], MODELS[key])

def validate_models() -> bool:
    ok = True
    for key, fname in MODELS.items():
        fp = os.path.join(MODEL_DEST_DIRS[key], fname)
        exists = os.path.exists(fp) and os.path.getsize(fp) > 0
        print(f"  {'✓' if exists else '✗ MISSING':10s} {fname}")
        if not exists: ok = False
    return ok

# =============================================================================
# SECTION 9 — LTX TEMPORAL MATH
# =============================================================================

def _is_valid_ltx_frame_count(n: int, min_frames: int = 9) -> bool:
    return n >= min_frames and (n - 1) % 8 == 0

def normalize_ltx_frame_count(requested: int, fps: int = 24, min_frames: int = 9) -> int:
    if _is_valid_ltx_frame_count(requested, min_frames): return requested
    k = math.ceil((requested - 1) / 8)
    adjusted = k * 8 + 1
    print(f"  LTX frame adjustment: {requested} → {adjusted} ({adjusted/fps:.2f}s)")
    return adjusted

def calculate_timeline(duration_s: float, fps: int) -> Tuple[int, float]:
    raw = round(duration_s * fps)
    valid = normalize_ltx_frame_count(raw, fps)
    return valid, valid / fps

def get_chunk_seed(global_seed: int, chunk_index: int, scene_index: int = 0) -> int:
    return (global_seed + scene_index * 99991 + chunk_index * 1000003) & 0x7FFFFFFF

def plan_chunks(total_frames: int, chunk_size: int, fps: int) -> List[Dict]:
    chunks, start, idx = [], 0, 0
    while start < total_frames:
        raw_size  = min(chunk_size, total_frames - start)
        valid_size = normalize_ltx_frame_count(raw_size, fps)
        if start + valid_size > total_frames:
            valid_size = total_frames - start
            if valid_size < 9:
                if chunks: chunks[-1]["num_frames"] += valid_size
                break
        chunks.append({"chunk_index": idx, "start_frame": start,
                        "num_frames": valid_size, "fps": fps, "path": None})
        idx += 1; start += valid_size
    return chunks

def estimate_chunk_size(w: int, h: int, fps: int, mode: str = "t4_safe") -> int:
    profile = T4_PROFILES.get(mode, T4_PROFILES["t4_safe"])
    if not CONFIG["auto_chunk_size"]:
        return normalize_ltx_frame_count(profile["chunk_frames"])
    free_gb    = max(mem.gpu_free_gb() - CONFIG["gpu_safety_margin_gb"], 1.0)
    lw, lh     = w // 8, h // 8
    bpf        = lw * lh * 128 * 2 * 2
    max_frames = max(9, min(int(free_gb * (1024**3) / bpf), profile["chunk_frames"]))
    return normalize_ltx_frame_count(max_frames, fps)


# =============================================================================
# SECTION 10 — SCENE STORYBOARD BUILDER
# =============================================================================

CAMERA_LORA_MAPPING: Dict[str, str] = {
    "dolly_forward":   "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors",
    "dolly_in":        "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors",
    "zoom_in":         "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors",
    "zoom_in_slow":    "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors",
    "zoom_in_fast":    "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors",
    "dolly_reveal":    "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors",
    "push_in_slow":    "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors",
    "orbit_left":      "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors",
    "dolly_right":     "LTX2.3-MVCamera-drclips.safetensors",
    "pan_right":       "LTX2.3-MVCamera-drclips.safetensors",
    "dolly_left":      "LTX2.3-MVCamera-drclips.safetensors",
    "pan_left":        "LTX2.3-MVCamera-drclips.safetensors",
    "tilt_down":       "LTX2.3-MVCamera-drclips.safetensors",
    "tilt_up":         "LTX2.3-MVCamera-drclips.safetensors",
    "static":          "ltx2.3-transition.safetensors",
    "static_intense":  "ltx2.3-transition.safetensors",
    "static_dramatic": "ltx2.3-transition.safetensors",
    "low_angle_hero":  "ltx2.3-transition.safetensors",
    "handheld_pov":    "ltx2.3-transition.safetensors",
}

def validate_scene_schema(json_data: dict) -> None:
    required = ["scene_id","project_name","story_action","main_characters","environment"]
    missing  = [k for k in required if k not in json_data]
    if missing:
        raise ValueError(f"SCENE_JSON missing keys: {missing}")
    if "shots" not in json_data["story_action"]:
        raise ValueError("'story_action' missing 'shots'")
    for idx, shot in enumerate(json_data["story_action"]["shots"]):
        bad = [k for k in ["time","camera","camera_movement","motion_intensity","action"]
               if k not in shot]
        if bad:
            raise ValueError(f"Shot {idx+1} missing keys: {bad}")
    print(f"  ✓ SCENE_JSON validated — {len(json_data['story_action']['shots'])} shots")

@lru_cache(maxsize=128)
def build_character_prompt(char_json_str: str) -> str:
    d  = json.loads(char_json_str)
    ap = d["detailed_appearance"]
    return (
        f"{d['name']}: {ap['face']}, {ap['hair']}, "
        f"wearing {ap['clothing']}, {ap['build']}, {ap['skin_tone']}, "
        f"{ap['accessories']}. MAINTAIN {d['name']}'s exact appearance. "
    )

def get_character_prefix(scene_json: dict) -> str:
    parts = [build_character_prompt(json.dumps(c, sort_keys=True))
             for c in scene_json["main_characters"]]
    return ("CHARACTER CONSISTENCY CRITICAL: " + " | ".join(parts) +
            " | MAINTAIN EXACT SAME APPEARANCE. NO MORPHING.")

def get_dialogue_for_shot(start_s: float, end_s: float, dialogue_list: list) -> str:
    lines = []
    for e in dialogue_list:
        if start_s <= float(e.get("time", -1)) < end_s:
            char = e["character"]
            text = e["dialogue"]
            emo  = e.get("emotion", "neutral")
            if e.get("lip_sync_emphasis") == "high":
                lines.append(f"LIP SYNC CRITICAL: {char} speaks '{text}' with {emo}. "
                              f"Mouth movements MUST match audio exactly.")
            else:
                lines.append(f"{char} says '{text}' with {emo}.")
    return " | ".join(lines)

def get_motion_guidance_prompt(shot: dict) -> str:
    mi   = shot.get("motion_intensity", 0.5)
    mv   = shot.get("camera_movement", "static").replace("_"," ")
    desc = ("subtle movements, mostly static" if mi < 0.3
            else "moderate motion, natural pace" if mi < 0.6
            else "dynamic motion, energetic action")
    return f"MOTION: {desc}. CAMERA: {mv}. "

def build_shot_prompt(shot: dict, scene_json: dict, shot_index: int) -> str:
    # Parse time range
    try:
        t = shot["time"].replace("s","").split("-")
        start_s, end_s = int(t[0]), int(t[1])
    except Exception:
        start_s, end_s = 0, 5

    # Use override if provided, else auto-build
    if shot.get("prompt_override","").strip():
        base = shot["prompt_override"].strip()
    else:
        char_p  = get_character_prefix(scene_json)
        env     = scene_json["environment"]
        env_p   = (f"ENVIRONMENT: {env['location']}. LIGHTING: {env['lighting']}. "
                   f"MOOD: {env['mood']}. PALETTE: {env['color_palette']}. ")
        dlg_p   = get_dialogue_for_shot(start_s, end_s,
                                         scene_json.get("dialogue_with_timing", []))
        audio_p = (f"AUDIO: {scene_json['audio']['background_music']}. "
                   f"SFX: {scene_json['audio']['environment_sfx']}. ")
        base = (
            f"{char_p} | "
            f"SHOT {shot_index+1}: {shot['action']}. "
            f"CAMERA: {shot['camera']}. "
            f"{get_motion_guidance_prompt(shot)}"
            f"EMOTION: {shot.get('emotion','neutral')}. "
            f"FOCUS: {shot.get('character_focus','scene')}. "
            f"EFFECTS: {shot.get('visual_effects','natural')}. "
            f"STYLE: {scene_json['video_style']}. "
            f"{env_p}{audio_p}"
        )
        if dlg_p: base += dlg_p

    # LLM expansion if enabled
    if LLM_EXPANSION and base:
        base = expand_prompt_via_llm(base, scene_json["video_style"])
    return base

def build_storyboard(scene_json: dict) -> List[Dict]:
    validate_scene_schema(scene_json)
    board = []
    shots = scene_json["story_action"]["shots"]
    for idx, shot in enumerate(shots):
        cm = shot.get("camera_movement","static")
        cam_lora = CAMERA_LORA_MAPPING.get(cm, None)
        board.append({
            "id":          f"shot_{idx+1:03d}",
            "prompt":      build_shot_prompt(shot, scene_json, idx),
            "shot_data":   shot,
            "camera_lora": cam_lora,
            "prev_shot":   shots[idx-1] if idx > 0 else None,
        })
    print(f"  ✓ Storyboard: {len(board)} shots")
    return board


# =============================================================================
# SECTION 11 — ADAPTIVE ANCHOR & OVERLAP (from ltx2_ti2v_distilled.py)
# =============================================================================

def calculate_adaptive_strength(shot: dict, prev_shot: Optional[dict],
                                  prev_shot_success: bool) -> float:
    s = ANCHOR_STRENGTH_HIGH
    if prev_shot:
        mc = abs(shot.get("motion_intensity", 0.5) - prev_shot.get("motion_intensity", 0.5))
        if mc > 0.4:   s -= 0.10
        elif mc < 0.2: s += 0.05
        if shot.get("character_focus") != prev_shot.get("character_focus"): s -= 0.05
    if not prev_shot_success: s -= 0.10
    return max(ANCHOR_STRENGTH_LOW, min(ANCHOR_STRENGTH_HIGH, s))

def calculate_adaptive_overlap(shot: dict) -> int:
    if not USE_ADAPTIVE_OVERLAP: return OVERLAP_FRAMES
    mi  = shot.get("motion_intensity", 0.5)
    adj = max(2, int(OVERLAP_FRAMES * 0.15))
    if mi > 0.7: return OVERLAP_FRAMES + adj
    if mi < 0.3: return OVERLAP_FRAMES - adj
    return OVERLAP_FRAMES

def calculate_motion_score(f1: np.ndarray, f2: np.ndarray) -> float:
    return 1.0 / (1.0 + float(np.mean(cv2.absdiff(f1, f2))))

def extract_overlap_anchor(video_path: str, output_folder: str,
                             scene_idx: int, overlap: int = 8) -> Optional[str]:
    """
    Extract the best anchor frame from the end of a generated clip.
    Scores by brightness * 0.3 + sharpness * 0.1 + motion_stability * 100.
    Optionally blends in character sheet for identity consistency.
    Returns path to saved anchor PNG, or None on failure.
    """
    if not os.path.exists(video_path): return None
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0: cap.release(); return None

    target_point = max(total - 5, total - overlap) if overlap < total else total - 5
    ws = max(0, target_point - 4)
    we = min(total - 1, target_point + 4)

    frames: Dict[int, np.ndarray] = {}
    for fi in range(ws, we + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret: frames[fi] = frame
    cap.release()

    if not frames: return None

    sorted_fi  = sorted(frames.keys())
    best_frame, best_score = None, -1.0
    for i, fi in enumerate(sorted_fi):
        frame = frames[fi]
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bright = float(cv2.mean(gray)[0])
        if bright < 5: continue
        sharp  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mscore = calculate_motion_score(frames[sorted_fi[i-1]], frame) if i > 0 else 1.0
        score  = bright * 0.3 + sharp * 0.1 + mscore * 100
        if score > best_score: best_score = score; best_frame = frame

    if best_frame is None: return None

    # Optional character-sheet consistency blend
    sheet_path = CHARACTER_SHEET_PATH.strip() or (IMAGE_PATH or "")
    if USE_CHARACTER_SHEETS and sheet_path and os.path.exists(sheet_path):
        try:
            sheet = cv2.imread(sheet_path)
            if sheet is not None:
                h, w = best_frame.shape[:2]
                sheet = cv2.resize(sheet, (w, h))
                best_frame = cv2.addWeighted(best_frame, 0.95, sheet, 0.05, 0)
        except Exception as e:
            print(f"  ⚠ Anchor blend failed: {e}")

    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, f"anchor_scene_{scene_idx:04d}.png")
    cv2.imwrite(path, best_frame)
    print(f"  ✓ Anchor extracted (score={best_score:.2f}): {path}")
    return path


# =============================================================================
# SECTION 12 — LLM PROMPT EXPANSION (optional)
# =============================================================================

def expand_prompt_via_llm(action: str, style: str) -> str:
    """
    Expands action description into rich cinematic prompt via OpenAI or Gemini.
    Falls back gracefully if API call fails or keys are missing.
    """
    if not LLM_EXPANSION or not action: return action
    if not LLM_API_KEY:
        return (f"{action}. Cinematic lighting highlights every detail. "
                f"Shot on 35mm lens with shallow depth of field. {style}.")
    if not _HAS_REQUESTS: return action

    sys_prompt = (
        "You are a professional cinematographer. Expand the following action into a "
        "detailed 200-word visual prompt for AI video generation. Focus on lighting, "
        "camera movement, texture, and atmosphere. Do NOT include dialogue or sound. "
        f"Target Style: {style}"
    )
    try:
        if LLM_PROVIDER == "openai":
            r = _requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Content-Type":"application/json",
                         "Authorization":f"Bearer {LLM_API_KEY}"},
                json={"model":"gpt-4o",
                      "messages":[{"role":"system","content":sys_prompt},
                                  {"role":"user","content":action}],
                      "temperature":0.7},
                timeout=15
            )
            if r.status_code == 200:
                print("  🧠 LLM expansion (OpenAI)")
                return r.json()["choices"][0]["message"]["content"]
        elif LLM_PROVIDER == "gemini":
            r = _requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-1.5-pro:generateContent?key={LLM_API_KEY}",
                json={"contents":[{"parts":[{"text":f"{sys_prompt}\n\nACTION: {action}"}]}]},
                timeout=15
            )
            if r.status_code == 200:
                print("  🧠 LLM expansion (Gemini)")
                return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print(f"  ⚠ LLM expansion failed: {e}")
    return action

# =============================================================================
# SECTION 13 — INPUT IMAGE HELPERS
# =============================================================================

def tensor_width_height(image) -> Tuple[int, int]:
    if isinstance(image, (tuple, list)): image = get_value_at_index(image, 0)
    if image.ndim == 4: return int(image.shape[2]), int(image.shape[1])
    if image.ndim == 3: return int(image.shape[1]), int(image.shape[0])
    raise ValueError(f"Unsupported image tensor shape: {getattr(image,'shape',None)}")

def load_input_image(image_path: Optional[str], width: int, height: int) -> Tuple:
    if image_path is not None and os.path.exists(image_path):
        loadimage = get_node("LoadImage")
        loaded    = loadimage.load_image(image=image_path)
        print(f"  ✓ Image loaded: {image_path}")
        return loaded, 1.0, False
    else:
        noise = torch.full((1, height, width, 3), 0.5, dtype=torch.float32)
        print("  ✓ T2V mode — grey placeholder")
        return (noise, None), 0.0, True

def prepare_image_for_chunk(loaded_image_tuple, width: int, height: int,
                              img_compression: int = 18, longer_edge: int = 1312) -> Tuple:
    rimn   = get_node("ResizeImageMaskNode")
    rile   = get_node("ResizeImagesByLongerEdge")
    lprepr = get_node("LTXVPreprocess")
    resized = rimn.EXECUTE_NORMALIZED(
        input=get_value_at_index(loaded_image_tuple, 0), scale_method="lanczos",
        resize_type={"resize_type":"scale dimensions","width":width,"height":height,"crop":"center"})
    rescaled = rile.EXECUTE_NORMALIZED(longer_edge=longer_edge,
                                        images=get_value_at_index(resized, 0))
    preprocessed = lprepr.EXECUTE_NORMALIZED(img_compression=img_compression,
                                              image=get_value_at_index(rescaled, 0))
    resized_w, resized_h = tensor_width_height(get_value_at_index(resized, 0))
    latent_w = max(1, resized_w // 2)
    latent_h = max(1, resized_h // 2)
    del resized, rescaled; mem.soft_cleanup()
    return preprocessed, latent_w, latent_h


# =============================================================================
# SECTION 14 — TEXT CONDITIONING (CPU cache)
# =============================================================================

_CONDITIONING_CACHE: Dict[str, Any] = {}

def build_text_conditioning(prompt: str, fps: int,
                              cache_key: Optional[str] = None) -> Tuple:
    ck = cache_key or hashlib.md5(f"{prompt}|{fps}".encode()).hexdigest()
    if ck in _CONDITIONING_CACHE:
        print("  ✓ Conditioning from cache.")
        return _CONDITIONING_CACHE[ck]

    print("  Loading text encoder (DualCLIPLoader)...")
    dclip = get_node("DualCLIPLoader")
    try:
        clip_result = dclip.load_clip(
            clip_name1=MODELS["text_encoder_1"], clip_name2=MODELS["text_encoder_2"],
            type="ltxv", device="default")
    except Exception as e:
        print(f"  Primary CLIP failed ({e}), trying fp8 fallback...")
        clip_result = dclip.load_clip(
            clip_name1="gemma_3_12B_it_fp8_scaled.safetensors",
            clip_name2="ltx-2.3-22b-dev_embeddings_connectors.safetensors",
            type="ltxv", device="default")

    clip_obj = get_value_at_index(clip_result, 0)
    cte = get_node("CLIPTextEncode")
    pos_enc = cte.encode(text=prompt, clip=clip_obj)
    czo = get_node("ConditioningZeroOut")
    neg_enc = czo.zero_out(conditioning=get_value_at_index(pos_enc, 0))
    del clip_result, clip_obj, dclip, cte; mem.cleanup()

    ltxcond = get_node("LTXVConditioning")
    cond = ltxcond.EXECUTE_NORMALIZED(
        frame_rate=fps,
        positive=get_value_at_index(pos_enc, 0),
        negative=get_value_at_index(neg_enc, 0))
    pos_cond = get_value_at_index(cond, 0)
    neg_cond = get_value_at_index(cond, 1)
    result = (pos_cond, neg_cond)
    _CONDITIONING_CACHE[ck] = result
    del pos_enc, neg_enc, cond; mem.cleanup()
    print("  ✓ Conditioning built and cached.")
    return result

# =============================================================================
# SECTION 15 — MODEL LOADERS
# =============================================================================

def load_dit_model_raw(apply_loras: bool = True) -> Any:
    """Load DiT + LoRAs. Returns the model object (no caching here)."""
    global _DIT_MODEL_CACHE
    if _DIT_MODEL_CACHE is not None:
        print("  DiT from module cache"); return _DIT_MODEL_CACHE
    print("  Loading DiT (UnetLoaderGGUF)...")
    mem.cleanup()
    unetgg = get_node("UnetLoaderGGUF")
    unet_r = unetgg.load_unet(unet_name=MODELS["dit"])
    model  = get_value_at_index(unet_r, 0)
    del unet_r; mem.soft_cleanup()

    if apply_loras:
        from nodes import LoraLoaderModelOnly
        ll = LoraLoaderModelOnly()
        for lora_key, strength in [
            ("lora_distilled",  LORA_STRENGTHS["lora_distilled"]),
            ("lora_omninft",    LORA_STRENGTHS["lora_omninft"]),
            ("lora_transition", LORA_STRENGTHS["lora_transition"]),
            ("lora_mvcamera",   LORA_STRENGTHS["lora_mvcamera"]),
        ]:
            if not LORA_ENABLED.get(lora_key, True):
                print(f"  LoRA disabled: {lora_key}"); continue
            lpath = os.path.join(MODEL_DEST_DIRS[lora_key], MODELS[lora_key])
            if os.path.exists(lpath):
                print(f"  LoRA: {MODELS[lora_key]}  str={strength}")
                model = ll.load_lora_model_only(model, MODELS[lora_key], strength)[0]
                mem.soft_cleanup()
    _DIT_MODEL_CACHE = model
    print("  DiT ready.")
    return model

def load_dit_model(apply_loras: bool = True) -> Any:
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE.get_dit(lambda: load_dit_model_raw(apply_loras))
    return load_dit_model_raw(apply_loras)

def release_dit_model():
    global _DIT_MODEL_CACHE
    if _MODEL_CACHE is not None:
        _MODEL_CACHE.evict_dit(); _DIT_MODEL_CACHE = None
    elif _DIT_MODEL_CACHE is not None:
        del _DIT_MODEL_CACHE; _DIT_MODEL_CACHE = None
        mem.aggressive_cleanup(); print("  DiT released.")

def _load_video_vae_raw() -> Any:
    print("  Loading video VAE..."); vl = get_node("VAELoader")
    return get_value_at_index(vl.load_vae(vae_name=MODELS["video_vae"]), 0)

def _load_audio_vae_raw() -> Any:
    print("  Loading audio VAE...")
    from nodes import NODE_CLASS_MAPPINGS
    if "VAELoaderKJ" in NODE_CLASS_MAPPINGS:
        l = NODE_CLASS_MAPPINGS["VAELoaderKJ"]()
        r = l.load_vae(vae_name=MODELS["audio_vae"],device="main_device",weight_dtype="fp16")
    else:
        l = NODE_CLASS_MAPPINGS["VAELoader"]()
        r = l.load_vae(vae_name=MODELS["audio_vae"])
    return get_value_at_index(r, 0)

def load_video_vae() -> Any:
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE.get_video_vae(_load_video_vae_raw)
    return _load_video_vae_raw()

def load_audio_vae() -> Any:
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE.get_audio_vae(_load_audio_vae_raw)
    return _load_audio_vae_raw()

def load_upscaler_model() -> Any:
    def _raw():
        print("  Loading spatial upscaler...")
        l = get_node("LatentUpscaleModelLoader")
        return get_value_at_index(l.EXECUTE_NORMALIZED(model_name=MODELS["upscaler"]), 0)
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE.get_upscaler(_raw)
    return _raw()


# =============================================================================
# SECTION 16 — DIRECTOR CONDITIONING (from FIXED.py, verbatim)
# =============================================================================

def _build_director_fallback(pos_cond, neg_cond, num_frames: int, fps: int,
                              dit_model=None, audio_vae=None,
                              reason: str = "not found") -> Tuple:
    print(f"  LTXDirector fallback ({reason}) — standard conditioning.")
    if dit_model is None: dit_model = load_dit_model(apply_loras=True)
    else: print("  Reusing pre-loaded DiT model.")
    if audio_vae is None: audio_vae = load_audio_vae()
    else: print("  Reusing pre-loaded audio VAE.")
    ltxvela = get_node("LTXVEmptyLatentAudio")
    audio_lat = ltxvela.EXECUTE_NORMALIZED(
        frames_number=num_frames, frame_rate=fps, batch_size=1, audio_vae=audio_vae)
    return dit_model, pos_cond, None, get_value_at_index(audio_lat, 0), None, None, fps

def build_director_conditioning(pos_cond, neg_cond, image_path, audio_path,
                                  num_frames, fps, width, height,
                                  dit_model=None, audio_vae=None) -> Tuple:
    from nodes import NODE_CLASS_MAPPINGS
    active_profile = T4_PROFILES.get(QUALITY_MODE, {})
    if active_profile.get("skip_director", False):
        return _build_director_fallback(pos_cond, neg_cond, num_frames, fps,
                                        dit_model=dit_model, audio_vae=audio_vae,
                                        reason="t4_safe mode — CLIP skipped")
    if "LTXDirector" in NODE_CLASS_MAPPINGS:
        print("  Using LTXDirector (WhatDreamsCost)...")
        if dit_model is None: dit_model = load_dit_model(apply_loras=True)
        if audio_vae is None: audio_vae = load_audio_vae()
        dclip = get_node("DualCLIPLoader")
        try:
            clip_r = dclip.load_clip(clip_name1=MODELS["text_encoder_1"],
                                      clip_name2=MODELS["text_encoder_2"],
                                      type="ltxv",device="default")
        except Exception as e:
            clip_r = dclip.load_clip(clip_name1="gemma_3_12B_it_fp8_scaled.safetensors",
                                      clip_name2="ltx-2.3-22b-dev_embeddings_connectors.safetensors",
                                      type="ltxv",device="default")
        clip_model = get_value_at_index(clip_r, 0)
        dir_cls = NODE_CLASS_MAPPINGS["LTXDirector"]
        director = dir_cls()
        try:
            input_types = dir_cls.INPUT_TYPES()
        except Exception:
            input_types = {"required":{}, "optional":{}}
        all_accepted = set(input_types.get("required",{}).keys()) | set(input_types.get("optional",{}).keys())
        total_frames  = num_frames
        duration_s    = total_frames / fps
        kwargs = dict(model=dit_model, audio_vae=audio_vae, global_prompt=GLOBAL_PROMPT)
        if not all_accepted or "clip" in all_accepted: kwargs["clip"] = clip_model
        for p, v in [
            ("start_second",0),("end_second",duration_s),("duration_seconds",duration_s),
            ("start_frame",0),("end_frame",total_frames),("duration_frames",total_frames),
            ("timeline_data",json.dumps({"segments":[],"motionSegments":[],"audioSegments":[],
                                         "global_prompt":GLOBAL_PROMPT,"mainTrackEnabled":True,
                                         "audioTrackEnabled":True,"motionTrackEnabled":True})),
            ("local_prompts",""),("segment_lengths",""),("frame_rate",fps),
            ("custom_width",width),("custom_height",height),("divisible_by",32),
        ]:
            if not all_accepted or p in all_accepted: kwargs[p] = v
        try:
            fn_name = getattr(dir_cls,"FUNCTION",None)
            out = getattr(director,fn_name)(**kwargs) if fn_name else director.EXECUTE_NORMALIZED(**kwargs)
        except (TypeError,AttributeError) as e:
            return _build_director_fallback(pos_cond,neg_cond,num_frames,fps,
                                            dit_model=dit_model,audio_vae=audio_vae,reason=str(e))
        return (get_value_at_index(out,0), get_value_at_index(out,1),
                get_value_at_index(out,2), get_value_at_index(out,3),
                get_value_at_index(out,4) if len(out)>4 else None,
                get_value_at_index(out,5) if len(out)>5 else None,
                get_value_at_index(out,6) if len(out)>6 else fps)
    return _build_director_fallback(pos_cond,neg_cond,num_frames,fps,
                                    dit_model=dit_model,audio_vae=audio_vae)


# =============================================================================
# SECTION 17 — DIRECTOR GUIDE & CROP GUIDES (from FIXED.py)
# =============================================================================

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
        it = guide_cls.INPUT_TYPES()
    except Exception:
        it = {"required":{}, "optional":{}}
    all_accepted = set(it.get("required",{}).keys()) | set(it.get("optional",{}).keys())
    inputs = dict(positive=pos_cond, negative=neg_cond, vae=video_vae,
                  latent=latent, model=model)
    for p, v in [("upscale_factor",upscale_factor),("interpolation","bicubic"),
                 ("blend_radius",1),("use_tiling",True),("tile_size",256),("tile_stride",64)]:
        if not all_accepted or p in all_accepted: inputs[p] = v
    if not all_accepted or "guide_data" in all_accepted: inputs["guide_data"] = guide_data
    if motion_guide_data is not None: inputs["motion_guide_data"] = motion_guide_data
    try:
        fn = getattr(guide_cls,"FUNCTION",None)
        out = getattr(guide_node,fn)(**inputs) if fn else guide_node.EXECUTE_NORMALIZED(**inputs)
    except (TypeError,AttributeError) as e:
        print(f"  LTXDirectorGuide ({node_id}) failed: {e} — passthrough.")
        return pos_cond, neg_cond, latent, model
    pos_out   = get_value_at_index(out, 0)
    neg_out   = get_value_at_index(out, 1)
    lat_out   = get_value_at_index(out, 2)
    model_out = get_value_at_index(out, 3) if len(out) > 3 else model
    return pos_out, neg_out, lat_out, model_out

def run_director_crop_guides(pos_cond, neg_cond, latent,
                               prefer_standard: bool = False) -> Tuple:
    from nodes import NODE_CLASS_MAPPINGS
    use_dir = ("LTXDirectorCropGuides" in NODE_CLASS_MAPPINGS) and not prefer_standard
    if not use_dir:
        if "LTXVCropGuides" in NODE_CLASS_MAPPINGS:
            cn = NODE_CLASS_MAPPINGS["LTXVCropGuides"]()
            out = cn.EXECUTE_NORMALIZED(positive=pos_cond, negative=neg_cond, latent=latent)
        else:
            return pos_cond, neg_cond, latent
    else:
        crop_cls = NODE_CLASS_MAPPINGS["LTXDirectorCropGuides"]
        crop_n   = crop_cls()
        try:
            fn  = getattr(crop_cls,"FUNCTION",None)
            out = getattr(crop_n,fn)(positive=pos_cond,negative=neg_cond,latent=latent) if fn \
                  else crop_n.EXECUTE_NORMALIZED(positive=pos_cond,negative=neg_cond,latent=latent)
        except (TypeError,AttributeError) as e:
            print(f"  LTXDirectorCropGuides failed: {e} — passthrough.")
            return pos_cond, neg_cond, latent
    pos_out = get_value_at_index(out,0) or pos_cond
    neg_out = get_value_at_index(out,1) or neg_cond
    lat_out = get_value_at_index(out,2)
    return pos_out, neg_out, lat_out


# =============================================================================
# SECTION 18 — TWO-PASS SAMPLING (from FIXED.py)
# =============================================================================

def build_empty_latents(num_frames, latent_w, latent_h, fps,
                         image_preprocessed, image_strength, image_bypass,
                         video_vae, audio_vae) -> Tuple:
    eltxv = get_node("EmptyLTXVLatentVideo")
    empty_vid = eltxv.EXECUTE_NORMALIZED(width=latent_w,height=latent_h,
                                          length=num_frames,batch_size=1)
    i2v = get_node("LTXVImgToVideoInplace")
    img_cond = i2v.EXECUTE_NORMALIZED(
        strength=image_strength, bypass=image_bypass, vae=video_vae,
        image=get_value_at_index(image_preprocessed,0),
        latent=get_value_at_index(empty_vid,0))
    elalat = get_node("LTXVEmptyLatentAudio")
    empty_aud = elalat.EXECUTE_NORMALIZED(
        frames_number=num_frames, frame_rate=fps, batch_size=1, audio_vae=audio_vae)
    catav = get_node("LTXVConcatAVLatent")
    vsrc  = get_value_at_index(img_cond,0) if not image_bypass else get_value_at_index(empty_vid,0)
    av    = catav.EXECUTE_NORMALIZED(video_latent=vsrc,
                                      audio_latent=get_value_at_index(empty_aud,0))
    del empty_vid, empty_aud; mem.soft_cleanup()
    return av, img_cond

def run_sampling_pass(model, pos_cond, neg_cond, latent,
                       noise_seed: int, steps: int = WORKFLOW_STEPS,
                       cfg: float = WORKFLOW_CFG, denoise: float = 1.0,
                       pass_name: str = "Pass1") -> Any:
    print(f"  Sampling {pass_name} ({steps} steps, denoise={denoise}, seed={noise_seed})...")
    ksel = get_node("KSamplerSelect")
    sampler = ksel.EXECUTE_NORMALIZED(sampler_name=WORKFLOW_SAMPLER_PASS1)
    rn   = get_node("RandomNoise")
    noise = rn.EXECUTE_NORMALIZED(noise_seed=noise_seed)
    bsched = get_node("BasicScheduler")
    sigmas = bsched.EXECUTE_NORMALIZED(model=model, scheduler=WORKFLOW_SCHEDULER,
                                        steps=steps, denoise=denoise)
    cfg_node = get_node("CFGGuider")
    guider = cfg_node.EXECUTE_NORMALIZED(cfg=cfg, model=model,
                                          positive=pos_cond, negative=neg_cond)
    sca = get_node("SamplerCustomAdvanced")
    result = sca.EXECUTE_NORMALIZED(
        noise=get_value_at_index(noise,0), guider=get_value_at_index(guider,0),
        sampler=get_value_at_index(sampler,0), sigmas=get_value_at_index(sigmas,0),
        latent_image=latent)
    del noise, sampler, sigmas, guider; mem.soft_cleanup()
    return result

def separate_av_latent(sampler_output, output_index: int = 0) -> Tuple:
    sep = get_node("LTXVSeparateAVLatent")
    out = sep.EXECUTE_NORMALIZED(av_latent=get_value_at_index(sampler_output, output_index))
    return get_value_at_index(out,0), get_value_at_index(out,1)

def upsample_video_latent(video_latent, upscaler_model, video_vae) -> Any:
    print("  Upsampling latent (2x)...")
    lup = get_node("LTXVLatentUpsampler")
    return get_value_at_index(
        lup.upsample_latent(samples=video_latent,
                             upscale_model=upscaler_model, vae=video_vae), 0)

def recondition_image_on_upscaled(upscaled_latent, image_preprocessed,
                                    image_strength, image_bypass,
                                    video_vae, audio_lat_pass1) -> Any:
    i2v = get_node("LTXVImgToVideoInplace")
    if not image_bypass:
        rec = i2v.EXECUTE_NORMALIZED(
            strength=image_strength, bypass=image_bypass, vae=video_vae,
            image=get_value_at_index(image_preprocessed,0), latent=upscaled_latent)
        vid_for_p2 = get_value_at_index(rec,0)
    else:
        vid_for_p2 = upscaled_latent
    catav = get_node("LTXVConcatAVLatent")
    return catav.EXECUTE_NORMALIZED(video_latent=vid_for_p2,
                                     audio_latent=audio_lat_pass1)


# =============================================================================
# SECTION 19 — VAE DECODING & CHUNK SAVING (from FIXED.py)
# =============================================================================

def decode_video_latent(video_latent, video_vae, max_batch_frames: int = 0) -> Any:
    print("  VAE decoding video latent...")
    vaedec = get_node("VAEDecode")
    lat    = video_latent
    samples = lat["samples"] if isinstance(lat, dict) else lat
    if torch.is_tensor(samples) and samples.ndim == 5:
        t_lat   = samples.shape[2]
        est_ram = mem.estimate_frame_ram_gb(t_lat*8, samples.shape[3]*8, samples.shape[4]*8)
        if max_batch_frames > 0 or (mem.cpu_available_gb() < est_ram + 2.0):
            batch_t = max_batch_frames if max_batch_frames > 0 else 8
            print(f"  Sub-batch decode (batch_t={batch_t}, RAM avail={mem.cpu_available_gb():.2f}GB)")
            all_frames = []
            for t0 in range(0, t_lat, batch_t):
                t1 = min(t0 + batch_t, t_lat)
                sub = {"samples": samples[:,:,t0:t1,:,:]}
                dec = vaedec.decode(samples=sub, vae=video_vae)
                fr  = get_value_at_index(dec,0).detach().to("cpu", non_blocking=False)
                torch.cuda.synchronize()
                all_frames.append(fr)
                del fr, dec, sub; mem.cleanup()
            frames_cpu = torch.cat(all_frames, dim=0); del all_frames; mem.soft_cleanup()
            return frames_cpu
    dec = vaedec.decode(samples=lat, vae=video_vae)
    fr  = get_value_at_index(dec,0).detach().to("cpu", non_blocking=False)
    torch.cuda.synchronize(); del dec; mem.cleanup()
    return fr

def decode_audio_latent(audio_latent, audio_vae) -> Any:
    print("  VAE decoding audio latent...")
    aud_dec = get_node("LTXVAudioVAEDecode")
    dec = aud_dec.EXECUTE_NORMALIZED(samples=audio_latent, audio_vae=audio_vae)
    aud = get_value_at_index(dec,0)
    if torch.is_tensor(aud): aud = aud.detach().cpu()
    elif isinstance(aud,dict) and "waveform" in aud and torch.is_tensor(aud["waveform"]):
        aud = {**aud, "waveform": aud["waveform"].detach().cpu()}
    del dec; mem.cleanup(); return aud

def save_chunk_to_disk(frames_cpu, audio_cpu, chunk_index, fps, width, height) -> str:
    chunks_dir = os.path.join(CONFIG["workspace_dir"],"chunks")
    Path(chunks_dir).mkdir(parents=True, exist_ok=True)
    chunk_path = os.path.join(chunks_dir, f"chunk_{chunk_index:04d}.mp4")
    if not mem.is_ram_safe(required_gb=4.0):
        _write_chunk_via_ffmpeg(frames_cpu, audio_cpu, chunk_path, fps, width, height)
        return chunk_path
    try:
        from nodes import NODE_CLASS_MAPPINGS
        import folder_paths
        from comfy_api.latest import Types
        if "CreateVideo" in NODE_CLASS_MAPPINGS:
            cv  = NODE_CLASS_MAPPINGS["CreateVideo"]()
            vob = cv.EXECUTE_NORMALIZED(fps=fps,images=frames_cpu,audio=audio_cpu)
            vid = get_value_at_index(vob,0)
            w   = frames_cpu.shape[2] if frames_cpu.ndim==4 else width
            h   = frames_cpu.shape[1] if frames_cpu.ndim==4 else height
            full_folder,fname,counter,_,_ = folder_paths.get_save_image_path(
                f"chunk_{chunk_index:04d}", folder_paths.get_output_directory(), w, h)
            ext  = Types.VideoContainer.get_extension("auto")
            tmp  = os.path.join(full_folder,f"{fname}_{counter:05d}_.{ext}")
            vid.save_to(tmp,format=Types.VideoContainer("auto"),codec="auto",metadata=None)
            shutil.move(tmp, chunk_path)
            del vob, vid; mem.soft_cleanup()
            print(f"  ✓ Chunk {chunk_index:04d} saved (CreateVideo)")
            return chunk_path
    except Exception as e:
        print(f"  CreateVideo failed ({e}), falling back to ffmpeg pipe...")
    _write_chunk_via_ffmpeg(frames_cpu, audio_cpu, chunk_path, fps, width, height)
    return chunk_path

def _write_chunk_via_ffmpeg(frames_cpu, audio_cpu, out_path, fps, w, h):
    if torch.is_tensor(frames_cpu):
        n, fh, fw = frames_cpu.shape[0], frames_cpu.shape[1], frames_cpu.shape[2]
    else:
        n, fh, fw = frames_cpu.shape[0], frames_cpu.shape[1], frames_cpu.shape[2]
    cmd = ["ffmpeg","-y","-f","rawvideo","-vcodec","rawvideo",
           "-s",f"{fw}x{fh}","-pix_fmt","rgb24","-r",str(fps),"-i","pipe:0",
           "-vcodec","libx264","-pix_fmt","yuv420p","-crf","18","-preset","fast",out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for i in range(n):
        frame = (frames_cpu[i].clamp(0,1)*255).byte().numpy() if torch.is_tensor(frames_cpu) \
                else frames_cpu[i]
        proc.stdin.write(frame.tobytes()); del frame
        if i % 16 == 0: gc.collect()
    proc.stdin.close(); proc.wait()
    print(f"  ✓ Chunk saved (ffmpeg pipe): {out_path}")


# =============================================================================
# SECTION 20 — SINGLE CHUNK GENERATION (full Director 2.0 pipeline from FIXED.py)
# =============================================================================

def generate_chunk(chunk_desc: Dict, pos_cond, neg_cond,
                    loaded_image_tuple, image_strength: float, image_bypass: bool,
                    width: int, height: int, fps: int, profile: Dict,
                    global_seed: int, scene_index: int = 0) -> Dict:
    idx         = chunk_desc["chunk_index"]
    start_frame = chunk_desc["start_frame"]
    num_frames  = chunk_desc["num_frames"]
    chunk_seed  = get_chunk_seed(global_seed, idx, scene_index)
    img_compress = profile.get("img_compression", WORKFLOW_IMG_COMPRESSION)
    longer_edge  = profile.get("longer_edge", 1312)

    mem.set_chunk_info(idx, num_frames, width, height)
    if CONFIG["enable_memory_logging"]:
        print(f"\n  GPU before chunk {idx}: {mem.gpu_free_gb():.2f} GB free")

    with torch.inference_mode():
        preprocessed, latent_w, latent_h = prepare_image_for_chunk(
            loaded_image_tuple, width, height, img_compress, longer_edge)

        active_profile = T4_PROFILES.get(QUALITY_MODE, {})
        if active_profile.get("skip_director", False):
            video_vae = None; audio_vae = load_audio_vae()
        else:
            video_vae = load_video_vae(); audio_vae = load_audio_vae()

        upscaler = None

        director_result = build_director_conditioning(
            pos_cond=pos_cond, neg_cond=neg_cond,
            image_path=None, audio_path=None,
            num_frames=num_frames, fps=fps, width=width, height=height,
            audio_vae=audio_vae)
        (dir_model, dir_positive, dir_video_latent, dir_audio_latent,
         dir_guide_data, dir_motion_guide_data, dir_frame_rate) = director_result
        no_director_data = (dir_guide_data is None and dir_motion_guide_data is None)
        base_model = dir_model

        if dir_video_latent is not None:
            video_latent_pass1     = dir_video_latent
            audio_latent_for_concat = dir_audio_latent
        else:
            if video_vae is None: video_vae = load_video_vae()
            av_lat1, img_cond_lat = build_empty_latents(
                num_frames, latent_w, latent_h, fps,
                preprocessed, image_strength, image_bypass, video_vae, audio_vae)
            video_latent_pass1 = get_value_at_index(img_cond_lat, 0)
            elalat = get_node("LTXVEmptyLatentAudio")
            fresh_aud = elalat.EXECUTE_NORMALIZED(
                frames_number=num_frames, frame_rate=fps, batch_size=1, audio_vae=audio_vae)
            audio_latent_for_concat = get_value_at_index(fresh_aud, 0)
            del av_lat1, img_cond_lat, fresh_aud; mem.soft_cleanup()

        if dir_positive is not None:
            czo = get_node("ConditioningZeroOut")
            neg_from_dir = czo.zero_out(conditioning=dir_positive)
            ltxcond = get_node("LTXVConditioning")
            dcond = ltxcond.EXECUTE_NORMALIZED(frame_rate=dir_frame_rate,
                                                positive=dir_positive,
                                                negative=get_value_at_index(neg_from_dir,0))
            cond_pos = get_value_at_index(dcond,0)
            cond_neg = get_value_at_index(dcond,1)
        else:
            cond_pos, cond_neg = pos_cond, neg_cond

        if video_vae is None: video_vae = load_video_vae()
        pos_g1, neg_g1, lat_g1, model_g1 = run_director_guide(
            cond_pos, cond_neg, video_vae, video_latent_pass1,
            dir_guide_data, dir_motion_guide_data, base_model,
            upscale_factor=0.5, node_id="pass1")

        if audio_latent_for_concat is not None:
            catav = get_node("LTXVConcatAVLatent")
            av_c1 = catav.EXECUTE_NORMALIZED(video_latent=lat_g1,
                                              audio_latent=audio_latent_for_concat)
            lat_samp1 = get_value_at_index(av_c1,0); del av_c1
        else:
            lat_samp1 = lat_g1

        sample_out_1 = run_sampling_pass(
            model_g1, pos_g1, neg_g1, lat_samp1,
            noise_seed=chunk_seed, steps=WORKFLOW_STEPS,
            cfg=WORKFLOW_CFG, denoise=1.0, pass_name=f"Pass1(chunk {idx})")
        del lat_samp1; mem.cleanup(); mem.warn_if_low()

        video_lat_p1, audio_lat_p1 = separate_av_latent(sample_out_1, output_index=0)
        del sample_out_1; mem.soft_cleanup()

        pos_c55, neg_c55, lat_c55 = run_director_crop_guides(
            pos_g1, neg_g1, video_lat_p1, prefer_standard=no_director_data)
        del video_lat_p1, pos_g1, neg_g1, model_g1; mem.soft_cleanup()

        upscaler = load_upscaler_model()
        upscaled_lat = upsample_video_latent(lat_c55, upscaler, video_vae)
        del lat_c55
        # Do NOT evict upscaler from cache here — it will be reused next chunk
        mem.cleanup()

        pos_g2, neg_g2, lat_g2, model_g2 = run_director_guide(
            pos_c55, neg_c55, video_vae, upscaled_lat,
            dir_guide_data, dir_motion_guide_data, base_model,
            upscale_factor=1.0, node_id="pass2")
        del pos_c55, neg_c55, upscaled_lat; mem.cleanup(); mem.warn_if_low()

        for _dv in ["dir_model","dir_guide_data","dir_motion_guide_data",
                    "dir_positive","dir_video_latent","dir_audio_latent"]:
            try: del locals()[_dv]
            except KeyError: pass
        mem.soft_cleanup()

        catav2 = get_node("LTXVConcatAVLatent")
        av_c2  = catav2.EXECUTE_NORMALIZED(video_latent=lat_g2,
                                            audio_latent=audio_lat_p1)
        lat_samp2 = get_value_at_index(av_c2,0)
        del av_c2, audio_lat_p1, lat_g2; mem.soft_cleanup()

        sample_out_2 = run_sampling_pass(
            model_g2, pos_g2, neg_g2, lat_samp2,
            noise_seed=0, steps=WORKFLOW_STEPS_PASS2,
            cfg=WORKFLOW_CFG, denoise=WORKFLOW_DENOISE_PASS2,
            pass_name=f"Pass2(chunk {idx})")
        del lat_samp2, model_g2

        # CRITICAL: release DiT before decode if model cache is NOT in use,
        # because the decode VAE needs the VRAM freed by the DiT.
        if not USE_MODEL_CACHE:
            try: del base_model
            except NameError: pass
            release_dit_model()
        mem.cleanup()

        final_video_lat, final_audio_lat = separate_av_latent(sample_out_2, output_index=0)
        del sample_out_2; mem.soft_cleanup()

        pos_c54, neg_c54, lat_c54 = run_director_crop_guides(
            pos_g2, neg_g2, final_video_lat, prefer_standard=no_director_data)
        del pos_g2, neg_g2, final_video_lat, pos_c54, neg_c54
        mem.aggressive_cleanup(); mem.ram_cleanup()

        frames_cpu = decode_video_latent(lat_c54, video_vae); del lat_c54
        audio_cpu  = decode_audio_latent(final_audio_lat, audio_vae)
        del final_audio_lat

        if not USE_MODEL_CACHE:
            del video_vae, audio_vae
        mem.cleanup()

    chunk_path = save_chunk_to_disk(frames_cpu, audio_cpu, idx, fps, width, height)
    del frames_cpu, audio_cpu; gc.collect()

    if CONFIG["enable_memory_logging"]:
        print(f"  GPU after  chunk {idx}: {mem.gpu_free_gb():.2f} GB free")
    if CONFIG["cleanup_after_chunk"]:
        mem.aggressive_cleanup()

    return {"chunk_index": idx, "start_frame": start_frame,
            "num_frames": num_frames, "fps": fps, "path": chunk_path}


# =============================================================================
# SECTION 21 — SCENE-LEVEL CHECKPOINT SYSTEM
# =============================================================================
# Per-scene checkpoint: each scene (shot) is checkpointed independently.
# Survives ANY crash — on restart, completed scenes are skipped automatically.
# =============================================================================

def _scene_checkpoint_path(project_name: str) -> str:
    return os.path.join(CONFIG["workspace_dir"], f"{project_name}_scene_checkpoint.json")

def init_scene_checkpoint(project_name: str, num_scenes: int, seed: int,
                            width: int, height: int) -> Dict:
    return {
        "project_name":      project_name,
        "num_scenes":        num_scenes,
        "seed":              seed,
        "resolution":        [width, height],
        "completed_scenes":  [],   # list of scene indices (ints)
        "failed_scenes":     [],
        "clip_paths":        {},   # {str(scene_idx): clip_path}
        "anchor_paths":      {},   # {str(scene_idx): anchor_path}
        "created_at":        time.strftime("%Y-%m-%dT%H:%M:%S"),
        "updated_at":        time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

def save_scene_checkpoint(cp: Dict, project_name: str):
    cp["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    Path(CONFIG["workspace_dir"]).mkdir(parents=True, exist_ok=True)
    p   = _scene_checkpoint_path(project_name)
    tmp = p + ".tmp"
    with open(tmp, "w") as f: json.dump(cp, f, indent=2)
    os.replace(tmp, p)

def load_scene_checkpoint(project_name: str) -> Optional[Dict]:
    p = _scene_checkpoint_path(project_name)
    if not os.path.exists(p): return None
    try:
        with open(p) as f: return json.load(f)
    except Exception as e:
        print(f"  ⚠ Scene checkpoint load failed: {e}"); return None

def get_or_create_scene_checkpoint(project_name: str, num_scenes: int,
                                    seed: int, width: int, height: int) -> Dict:
    if RESUME:
        existing = load_scene_checkpoint(project_name)
        if existing is not None:
            if (existing.get("num_scenes") == num_scenes
                    and existing.get("resolution") == [width, height]):
                done = len(existing.get("completed_scenes", []))
                print(f"  ↷ Resuming: {done}/{num_scenes} scenes already done.")
                return existing
            else:
                print("  ⚠ Checkpoint config mismatch — starting fresh.")
    cp = init_scene_checkpoint(project_name, num_scenes, seed, width, height)
    save_scene_checkpoint(cp, project_name)
    return cp

# =============================================================================
# SECTION 22 — OOM-RECOVERY CHUNK GENERATOR (from FIXED.py)
# =============================================================================

def adaptive_chunk_generator(chunks, pos_cond, neg_cond,
                               loaded_image_tuple, image_strength, image_bypass,
                               width, height, fps, profile, global_seed,
                               chunk_checkpoint: Dict,
                               scene_index: int = 0) -> List[Dict]:
    max_retries = CONFIG["max_oom_retries"]
    auto_reduce = CONFIG["auto_reduce_chunk_on_oom"]
    completed   = []
    current_chunks = list(chunks)
    i = 0
    while i < len(current_chunks):
        chunk_desc = current_chunks[i]
        idx = chunk_desc["chunk_index"]

        if idx in chunk_checkpoint.get("completed_chunks", []):
            ep = os.path.join(CONFIG["workspace_dir"],"chunks",f"chunk_{idx:04d}.mp4")
            if os.path.exists(ep) and os.path.getsize(ep) > 0:
                print(f"  ↷ Chunk {idx:04d} already done — skipping.")
                completed.append({"chunk_index":idx,"start_frame":chunk_desc["start_frame"],
                                   "num_frames":chunk_desc["num_frames"],"fps":fps,"path":ep})
                i += 1; continue

        retries, success, cur_frames = 0, False, chunk_desc["num_frames"]
        while retries <= max_retries and not success:
            try:
                print(f"\n{'='*60}\n[Chunk {idx+1}]  frames {chunk_desc['start_frame']}–"
                      f"{chunk_desc['start_frame']+cur_frames-1}  ({cur_frames} frames)\n{'='*60}")
                if CONFIG["enable_memory_logging"]: mem.print_memory("  ")
                result = generate_chunk(
                    chunk_desc={**chunk_desc,"num_frames":cur_frames},
                    pos_cond=pos_cond, neg_cond=neg_cond,
                    loaded_image_tuple=loaded_image_tuple,
                    image_strength=image_strength, image_bypass=image_bypass,
                    width=width, height=height, fps=fps, profile=profile,
                    global_seed=global_seed, scene_index=scene_index)
                completed.append(result)
                chunk_checkpoint.setdefault("completed_chunks",[]).append(idx)
                success = True
                print(f"  ✓ Chunk {idx:04d} complete.")
            except torch.cuda.OutOfMemoryError as oom:
                retries += 1
                print(f"\n  OOM on chunk {idx} | GPU free: {mem.gpu_free_gb():.2f}GB | "
                      f"retry {retries}/{max_retries}")
                release_dit_model(); mem.aggressive_cleanup()
                if not auto_reduce or retries > max_retries:
                    chunk_checkpoint.setdefault("failed_chunks",[]).append(idx); break
                reduction  = 0.75 if retries == 1 else 0.5
                cur_frames = normalize_ltx_frame_count(max(9, int(cur_frames*reduction)), fps)
                print(f"  Reducing chunk to {cur_frames} frames and retrying...")
            except Exception as e:
                print(f"\n  ERROR (chunk {idx}): {type(e).__name__}: {str(e)[:300]}")
                print(traceback.format_exc())
                chunk_checkpoint.setdefault("failed_chunks",[]).append(idx)
                release_dit_model(); mem.aggressive_cleanup(); break
        i += 1
    return completed


# =============================================================================
# SECTION 23 — POST-PROCESSING (Face Restore + Optical Flow)
# =============================================================================

def apply_face_restoration(video_path: str) -> str:
    """
    Per-frame face detection + cv2.detailEnhance.
    Falls back gracefully if ultralytics not installed.
    """
    if not FACE_RESTORATION: return video_path
    print("  ✨ Face restoration pass...")
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n-face.pt")
    except Exception:
        print("  ⚠ Ultralytics not available — skipping face restore."); return video_path

    cap = cv2.VideoCapture(video_path)
    fps_ = cap.get(cv2.CAP_PROP_FPS)
    w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = video_path.replace(".mp4","_facefixed.mp4")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret: break
        try:
            results = model(frame, verbose=False)
            for r in results:
                for box in r.boxes:
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    face = frame[y1:y2, x1:x2]
                    if face.size == 0: continue
                    frame[y1:y2, x1:x2] = cv2.detailEnhance(face, sigma_s=10, sigma_r=0.15)
        except Exception: pass
        out.write(frame)

    cap.release(); out.release()
    if os.path.exists(out_path):
        os.remove(video_path); os.rename(out_path, video_path)
        print(f"  ✓ Face restoration done: {video_path}")
    return video_path

def apply_optical_flow_morph(img1: np.ndarray, img2: np.ndarray,
                               steps: int = 5) -> List[np.ndarray]:
    """Generate morphed transition frames between two anchor images."""
    prev_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None,
                                         0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = img1.shape[:2]
    morphed = []
    for i in range(1, steps + 1):
        alpha = i / (steps + 1)
        fc    = flow * alpha
        mx, my = np.meshgrid(np.arange(w), np.arange(h))
        mx = (mx + fc[...,0]).astype(np.float32)
        my = (my + fc[...,1]).astype(np.float32)
        warped  = cv2.remap(img1, mx, my, interpolation=cv2.INTER_LINEAR)
        blended = cv2.addWeighted(warped, 1-alpha, img2, alpha, 0)
        morphed.append(blended)
    return morphed

# =============================================================================
# SECTION 24 — SHOT QUALITY METRICS
# =============================================================================

def calculate_shot_metrics(video_path: str) -> Tuple[float, Dict]:
    """Compute overall quality score for best-variation selection."""
    try:
        cap   = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step  = max(1, total // 10)
        sharp_v, bright_v, motion_v = [], [], []
        prev_gray = None
        for fi in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret: continue
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
        score = (min(1.0, metrics["sharpness"]/1000)
                 + min(1.0, metrics["brightness"]/200)
                 + max(0.0, 1.0 - metrics["motion_std"]/50)) / 3.0
        return score, metrics
    except Exception as e:
        print(f"  ⚠ Metrics error: {e}"); return 0.0, {}


# =============================================================================
# SECTION 25 — SINGLE SCENE GENERATION (wraps chunk generator)
# =============================================================================

def generate_scene(scene_entry: Dict, scene_index: int,
                    image_path: Optional[str], width: int, height: int,
                    fps: int, profile: Dict, global_seed: int,
                    scene_cp: Dict) -> Optional[str]:
    """
    Generate one scene (storyboard shot) as an MP4 clip.
    Internally uses plan_chunks + adaptive_chunk_generator (OOM recovery).
    Returns path to saved scene clip, or None on failure.
    """
    prompt    = scene_entry["prompt"]
    shot_data = scene_entry["shot_data"]

    print(f"\n{'='*60}")
    print(f"  SCENE {scene_index+1:03d}/{len(STORYBOARD):03d}  id={scene_entry['id']}")
    print(f"  Prompt: {prompt[:120]}...")
    print(f"{'='*60}")

    # Determine frame count from shot timing
    try:
        t = shot_data["time"].replace("s","").split("-")
        scene_dur = float(t[1]) - float(t[0])
    except Exception:
        scene_dur = SCENE_DURATION_S

    total_frames, actual_dur = calculate_timeline(scene_dur, fps)
    chunk_size = estimate_chunk_size(width, height, fps, QUALITY_MODE)
    chunks     = plan_chunks(total_frames, chunk_size, fps)
    print(f"  {total_frames} frames ({actual_dur:.2f}s), {len(chunks)} chunk(s)")

    # Load input image for this scene (anchor from previous scene if available)
    loaded_image, img_strength, img_bypass = load_input_image(image_path, width, height)

    # Build (or retrieve from cache) text conditioning
    mem.cleanup()
    pos_cond, neg_cond = build_text_conditioning(prompt, fps)

    # Per-scene chunk checkpoint (isolated from scene checkpoint)
    chunk_cp: Dict = {"completed_chunks": [], "failed_chunks": []}

    completed_chunks = adaptive_chunk_generator(
        chunks=chunks, pos_cond=pos_cond, neg_cond=neg_cond,
        loaded_image_tuple=loaded_image,
        image_strength=img_strength, image_bypass=img_bypass,
        width=width, height=height, fps=fps, profile=profile,
        global_seed=global_seed, chunk_checkpoint=chunk_cp,
        scene_index=scene_index)

    del pos_cond, neg_cond, loaded_image
    _CONDITIONING_CACHE.clear()
    mem.aggressive_cleanup()

    if not completed_chunks:
        print(f"  ✗ Scene {scene_index+1} — no chunks completed.")
        return None

    # Assemble chunks into scene clip
    scene_dir  = os.path.join(CONFIG["workspace_dir"], "scenes")
    Path(scene_dir).mkdir(parents=True, exist_ok=True)
    scene_clip = os.path.join(scene_dir, f"scene_{scene_index:04d}.mp4")

    assembled = _assemble_chunks(completed_chunks, scene_clip, fps)
    if not assembled: return None

    # Face restoration
    scene_clip = apply_face_restoration(scene_clip)

    # Clean up per-scene chunk files
    if CONFIG["cleanup_temp_files"]:
        for c in completed_chunks:
            if c.get("path") and os.path.exists(c["path"]):
                try: os.remove(c["path"])
                except Exception: pass

    print(f"  ✓ Scene {scene_index+1} complete: {scene_clip}")
    return scene_clip

def _assemble_chunks(completed_chunks: List[Dict], output_path: str, fps: int) -> bool:
    """FFmpeg concat: stream-copy chunks into a single clip."""
    if not completed_chunks: return False
    sorted_c = sorted(completed_chunks, key=lambda c: c["chunk_index"])
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    concat_f = output_path + "_concat.txt"
    with open(concat_f, "w") as f:
        for c in sorted_c:
            safe = c["path"].replace("'","'\\''")
            f.write(f"file '{safe}'\n")
    cmd = ["ffmpeg","-y","-f","concat","-safe","0","-i",concat_f,"-c","copy",output_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        print(f"  ✓ Scene assembled (stream-copy): {output_path}")
        os.remove(concat_f); return True
    # Re-encode fallback
    cmd2 = ["ffmpeg","-y","-f","concat","-safe","0","-i",concat_f,
            "-vcodec","libx264","-pix_fmt","yuv420p","-crf","18","-preset","fast",
            "-acodec","aac","-b:a","192k",output_path]
    r2 = subprocess.run(cmd2, capture_output=True, text=True)
    if r2.returncode == 0:
        os.remove(concat_f); return True
    print(f"  ✗ Assembly failed: {r2.stderr.strip()[:200]}")
    return False


# =============================================================================
# SECTION 26 — FINAL VIDEO ASSEMBLY (overlap stitch + transitions + audio)
# =============================================================================

def _detect_nvenc() -> bool:
    try:
        out = subprocess.run(["ffmpeg","-encoders"], capture_output=True, text=True, timeout=10)
        return "h264_nvenc" in out.stdout
    except Exception: return False

def stitch_scenes_to_movie(scene_clips: List[str], output_path: str,
                             fps: int, overlap_frames: int = 8) -> Optional[str]:
    """
    Concatenate scene clips with optional overlap trim.
    Uses ffmpeg stream-copy where possible, re-encode as fallback.
    """
    valid = [p for p in scene_clips if p and os.path.exists(p)]
    if not valid:
        print("  ✗ No valid scene clips to stitch."); return None

    print(f"\n  Stitching {len(valid)} scene clips → {output_path}")
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

    # Trim overlap from end of all-but-last clips using ffmpeg
    trimmed_clips = []
    for i, clip_path in enumerate(valid):
        if i < len(valid) - 1 and overlap_frames > 0:
            probe_cmd = ["ffprobe","-v","quiet","-print_format","json",
                         "-show_streams","-show_format",clip_path]
            try:
                pr = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
                info = json.loads(pr.stdout)
                vs   = next((s for s in info.get("streams",[]) if s.get("codec_type")=="video"),None)
                if vs:
                    dur  = float(info.get("format",{}).get("duration",0))
                    trim = dur - overlap_frames / fps
                    if trim > 0:
                        tp = clip_path.replace(".mp4",f"_trim{i}.mp4")
                        cmd_trim = ["ffmpeg","-y","-i",clip_path,
                                    "-t",f"{trim:.4f}","-c","copy",tp]
                        rt = subprocess.run(cmd_trim, capture_output=True)
                        if rt.returncode == 0:
                            trimmed_clips.append(tp); continue
            except Exception: pass
        trimmed_clips.append(clip_path)

    concat_f = output_path + "_final_concat.txt"
    with open(concat_f,"w") as f:
        for cp in trimmed_clips:
            safe = cp.replace("'","'\\''")
            f.write(f"file '{safe}'\n")

    use_nvenc = _detect_nvenc()
    codec     = "h264_nvenc" if use_nvenc else "libx264"
    preset    = "p4"  if use_nvenc else "medium"
    print(f"  Encoding: {'GPU NVENC' if use_nvenc else 'CPU libx264'}")

    cmd = ["ffmpeg","-y","-f","concat","-safe","0","-i",concat_f,
           "-vcodec",codec,"-pix_fmt","yuv420p","-crf","18",
           "-preset",preset,"-acodec","aac","-b:a","192k",output_path]
    r = subprocess.run(cmd, capture_output=True, text=True)

    # Cleanup trimmed temps
    for tp in trimmed_clips:
        if "_trim" in tp and os.path.exists(tp):
            try: os.remove(tp)
            except Exception: pass
    try: os.remove(concat_f)
    except Exception: pass

    if r.returncode == 0:
        sz = os.path.getsize(output_path) / (1024*1024)
        print(f"  ✓ Final movie: {output_path} ({sz:.1f} MB)")
        return output_path
    print(f"  ✗ Final stitch failed: {r.stderr.strip()[:300]}")
    return None

def mux_audio(video_path: str, audio_path: Optional[str],
               output_path: str, fps: int, total_frames: int,
               audio_start_s: float = 0.0) -> bool:
    if not audio_path or not os.path.exists(audio_path):
        if video_path != output_path: shutil.copy2(video_path, output_path)
        print(f"  ✓ Output (no external audio): {output_path}")
        return True
    dur = total_frames / fps
    cmd = ["ffmpeg","-y","-i",video_path,
           "-ss",str(audio_start_s),"-t",str(dur),"-i",audio_path,
           "-map","0:v:0","-map","1:a:0","-c:v","copy","-c:a","aac","-b:a","192k",
           "-shortest",output_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        print(f"  ✓ Audio muxed: {output_path}"); return True
    print(f"  ✗ Audio mux failed: {r.stderr.strip()[:200]}")
    shutil.copy2(video_path, output_path); return False

# =============================================================================
# SECTION 27 — SUBTITLE EXPORT
# =============================================================================

def format_time_srt(seconds: float) -> str:
    ms = int((seconds - int(seconds)) * 1000)
    h, m, s = int(seconds//3600), int((seconds%3600)//60), int(seconds%60)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def generate_srt(dialogue_list: list, output_path: str) -> None:
    try:
        with open(output_path,"w",encoding="utf-8") as f:
            for i, e in enumerate(dialogue_list, 1):
                start = float(e.get("time",0))
                end   = start + float(e.get("duration",4.0))
                f.write(f"{i}\n{format_time_srt(start)} --> {format_time_srt(end)}\n")
                f.write(f"[{e.get('character','?')}] {e.get('dialogue','')}\n\n")
        print(f"  ✓ Subtitles: {output_path}")
    except Exception as e:
        print(f"  ⚠ Subtitle export failed: {e}")


# =============================================================================
# SECTION 28 — DISPLAY HELPERS
# =============================================================================

def display_video_safe(video_path: str, max_size_mb: float = 50.0):
    if not os.path.exists(video_path): return
    size_mb = os.path.getsize(video_path) / (1024*1024)
    if size_mb > max_size_mb:
        print(f"  Video {size_mb:.1f} MB — too large for inline display.")
        print(f"  Download: files.download('{video_path}')"); return
    from base64 import b64encode
    chunks_b64 = []
    with open(video_path,"rb") as f:
        while True:
            block = f.read(65536)
            if not block: break
            chunks_b64.append(block)
    vid64 = b64encode(b"".join(chunks_b64)).decode(); del chunks_b64
    display(HTML(f'<video width=800 controls autoplay loop muted>'
                 f'<source src="data:video/mp4;base64,{vid64}" type="video/mp4"></video>'))
    del vid64

def print_vram_usage():
    used  = torch.cuda.memory_allocated() / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    bar   = "█" * int(20*used/total) + "░" * (20-int(20*used/total))
    print(f"  VRAM [{bar}] {used:.1f}/{total:.1f}GB ({used/total*100:.1f}%)")

# =============================================================================
# SECTION 29 — MAIN MULTI-SCENE ORCHESTRATOR
# =============================================================================

def generate_multiscene_mv(
    scene_json:   Optional[Dict] = None,
    image_path:   Optional[str]  = None,
    audio_path:   Optional[str]  = None,
    quality_mode: Optional[str]  = None,
    seed:         Optional[int]  = None,
    fps:          Optional[int]  = None,
    width:        Optional[int]  = None,
    height:       Optional[int]  = None,
) -> Optional[str]:
    """
    Complete multi-scene Director 2.0 MV pipeline.

    Pipeline:
        ① Parse SCENE_JSON → STORYBOARD (N shots)
        ② For each shot (auto-resume skips completed):
              • Build shot-specific prompt
              • Set anchor image (previous scene's last frame)
              • Generate clip via Director 2.0 two-pass pipeline
              • OOM retry × MAX_SCENE_RETRIES with VRAM recovery
              • Extract anchor from clip end
              • Save scene clip to workspace/scenes/
        ③ Stitch all clips → final movie (with overlap trim)
        ④ Mux original audio track
        ⑤ Export SRT subtitles
        ⑥ Display final video inline

    Returns final output path, or None on failure.
    """
    sj         = scene_json  or SCENE_JSON
    img_path   = image_path  or IMAGE_PATH
    aud_path   = audio_path  or AUDIO_PATH
    q_mode     = quality_mode or QUALITY_MODE
    global_seed = seed  if seed  is not None else SEED
    out_fps    = fps   or FPS
    out_w      = width  or OUTPUT_WIDTH
    out_h      = height or OUTPUT_HEIGHT

    generation_start = time.time()
    print("\n" + "="*60)
    print("LTX-2.3 DIRECTOR 2.0  —  MULTI-SCENE MV ENGINE")
    print("="*60)
    print(f"  GPU       : {_GPU_INFO['device_name']}")
    print(f"  VRAM      : {_GPU_INFO['vram_total_gb']:.1f} GB  ({mem.gpu_free_gb():.1f} GB free)")
    print(f"  Quality   : {q_mode}")
    print(f"  Resolution: {out_w}×{out_h} @ {out_fps}fps")
    print(f"  Seed      : {global_seed}")
    print(f"  Resume    : {RESUME}")
    print(f"  ModelCache: {USE_MODEL_CACHE}")
    print("="*60)

    # ── Build storyboard ──────────────────────────────────────────────────────
    global STORYBOARD
    STORYBOARD = build_storyboard(sj)
    num_scenes = len(STORYBOARD)
    print(f"\n  {num_scenes} shots in storyboard")

    # ── Profile ───────────────────────────────────────────────────────────────
    profile = T4_PROFILES.get(q_mode, T4_PROFILES["t4_safe"])
    gen_w, gen_h = profile["generation_width"], profile["generation_height"]
    if out_w > gen_w or out_h > gen_h:
        if CONFIG["allow_auto_downgrade"]:
            print(f"  Auto-downgrade {out_w}×{out_h} → {gen_w}×{gen_h}")
            out_w, out_h = gen_w, gen_h

    # ── ComfyUI setup ─────────────────────────────────────────────────────────
    setup_comfyui()
    import_custom_nodes()

    # ── Scene checkpoint ──────────────────────────────────────────────────────
    scene_cp = get_or_create_scene_checkpoint(
        PROJECT_NAME, num_scenes, global_seed, out_w, out_h)

    # ── Workspace dirs ────────────────────────────────────────────────────────
    scenes_dir  = os.path.join(CONFIG["workspace_dir"], "scenes")
    input_dir   = os.path.join(COMFYUI_DIR, "input")
    cache_dir   = os.path.join(CONFIG["output_dir"], f"{PROJECT_NAME}_cache")
    for d in [scenes_dir, input_dir, cache_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)
    Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)

    generated_clips: List[str] = []
    current_input_image = img_path or ""

    # ── Restore from checkpoint ───────────────────────────────────────────────
    # Reload clips and anchors from completed scenes
    start_index = 0
    for i in range(num_scenes):
        clip_p   = scene_cp["clip_paths"].get(str(i))
        anchor_p = scene_cp["anchor_paths"].get(str(i))
        if (i in scene_cp["completed_scenes"]
                and clip_p and os.path.exists(clip_p)):
            generated_clips.append(clip_p)
            if anchor_p and os.path.exists(anchor_p):
                current_input_image = anchor_p
            start_index = i + 1
        else:
            break

    if start_index > 0:
        print(f"  ↷ Restored {start_index} completed scenes from checkpoint.")

    prev_shot_success = True

    # ── Background stitch executor ────────────────────────────────────────────
    executor     = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    stitch_future = None

    try:
        for i in range(start_index, num_scenes):
            scene       = STORYBOARD[i]
            shot_data   = scene["shot_data"]
            mem.aggressive_cleanup()
            print_vram_usage()

            # Check background stitch result
            if stitch_future and stitch_future.done():
                try: print(f"  Background stitch result: {stitch_future.result()}")
                except Exception as e: print(f"  Background stitch error: {e}")

            # Adaptive image strength and overlap
            strength    = (calculate_adaptive_strength(shot_data, scene.get("prev_shot"),
                                                       prev_shot_success)
                           if i > 0 else 0.0)
            overlap_val = calculate_adaptive_overlap(shot_data)

            # Scene-level retry loop (MAX_SCENE_RETRIES)
            success   = False
            clip_path = None

            for attempt in range(1, MAX_SCENE_RETRIES + 1):
                seed_for_scene = global_seed + i * 9973 + attempt * 197
                print(f"\n  Shot {i+1}/{num_scenes}  attempt {attempt}/{MAX_SCENE_RETRIES}"
                      f"  seed={seed_for_scene}  strength={strength:.2f}  overlap={overlap_val}")

                try:
                    if GENERATE_SHOT_VARIATIONS and NUM_VARIATIONS > 1:
                        clip_path = _generate_shot_with_variations(
                            scene, current_input_image, strength,
                            seed_for_scene, out_w, out_h, out_fps,
                            profile, i)
                    else:
                        clip_path = generate_scene(
                            scene_entry=scene,
                            scene_index=i,
                            image_path=current_input_image or None,
                            width=out_w, height=out_h, fps=out_fps,
                            profile=profile,
                            global_seed=seed_for_scene,
                            scene_cp=scene_cp)

                    if clip_path and os.path.exists(clip_path):
                        # Extract anchor for next scene
                        anchor = extract_overlap_anchor(
                            clip_path, output_folder=input_dir,
                            scene_idx=i, overlap=overlap_val)

                        # Cache clip
                        cached = os.path.join(cache_dir, f"scene_{i:04d}.mp4")
                        shutil.copy(clip_path, cached)
                        generated_clips.append(cached)

                        # Update checkpoint
                        scene_cp["completed_scenes"].append(i)
                        scene_cp["clip_paths"][str(i)]   = cached
                        if anchor:
                            scene_cp["anchor_paths"][str(i)] = anchor
                            current_input_image = anchor
                        save_scene_checkpoint(scene_cp, PROJECT_NAME)

                        prev_shot_success = True
                        success = True
                        print(f"  ✓ Shot {i+1} complete! ({len(generated_clips)}/{num_scenes})")

                        # Trigger background partial stitch
                        if len(generated_clips) > 1:
                            partial_out = os.path.join(CONFIG["output_dir"],
                                                        f"{PROJECT_NAME}_PARTIAL.mp4")
                            stitch_future = executor.submit(
                                stitch_scenes_to_movie,
                                list(generated_clips), partial_out,
                                out_fps, overlap_val)
                        break
                    else:
                        print(f"  ⚠ Shot {i+1} attempt {attempt} — no clip output.")

                except torch.cuda.OutOfMemoryError:
                    print(f"  OOM on shot {i+1} attempt {attempt}")
                    release_dit_model(); mem.aggressive_cleanup()
                    prev_shot_success = False
                except Exception as e:
                    print(f"  ERROR shot {i+1} attempt {attempt}: {type(e).__name__}: {str(e)[:200]}")
                    traceback.print_exc()
                    release_dit_model(); mem.aggressive_cleanup()
                    prev_shot_success = False

            if not success:
                print(f"  ⚠ Shot {i+1} failed after {MAX_SCENE_RETRIES} attempts — "
                      f"skipping and continuing.")
                scene_cp["failed_scenes"].append(i)
                save_scene_checkpoint(scene_cp, PROJECT_NAME)
                prev_shot_success = False

    except KeyboardInterrupt:
        print("\n  Interrupted — stitching available clips...")
    except Exception as e:
        print(f"\n  Critical error: {e}")
        traceback.print_exc()
        print("  Attempting recovery stitch from completed clips...")
    finally:
        executor.shutdown(wait=False)
        if _MODEL_CACHE: _MODEL_CACHE.evict_all()

    # ── Final assembly ────────────────────────────────────────────────────────
    if not generated_clips:
        print("\n  ✗ No clips generated."); return None

    print(f"\n  Stitching {len(generated_clips)}/{num_scenes} scene clips...")
    asm_path = os.path.join(CONFIG["output_dir"], f"{PROJECT_NAME}_no_audio.mp4")
    total_frames = sum(
        round(SCENE_DURATION_S * out_fps) for _ in generated_clips)

    final_assembled = stitch_scenes_to_movie(
        generated_clips, asm_path, out_fps, OVERLAP_FRAMES)

    if not final_assembled:
        print("  ✗ Final stitch failed."); return None

    # ── Mux audio ─────────────────────────────────────────────────────────────
    final_output = os.path.join(CONFIG["output_dir"], OUTPUT_FILENAME)
    mux_audio(asm_path, aud_path, final_output, out_fps, total_frames)

    if os.path.exists(asm_path) and asm_path != final_output:
        os.remove(asm_path)

    # ── Subtitles ─────────────────────────────────────────────────────────────
    if GENERATE_SUBTITLES:
        srt_path = final_output.replace(".mp4",".srt")
        generate_srt(sj.get("dialogue_with_timing",[]), srt_path)

    # ── Job summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - generation_start
    print(f"\n{'='*60}")
    print(f"  GENERATION COMPLETE")
    print(f"  Output       : {final_output}")
    print(f"  Scenes done  : {len(scene_cp['completed_scenes'])}/{num_scenes}")
    print(f"  Scenes failed: {scene_cp.get('failed_scenes',[])}")
    print(f"  Elapsed      : {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"  Peak VRAM    : {mem.gpu_peak_gb():.3f} GB")
    print(f"{'='*60}\n")

    mem.aggressive_cleanup()
    display_video_safe(final_output)
    return final_output


# =============================================================================
# SECTION 30 — SHOT VARIATIONS HELPER
# =============================================================================

def _generate_shot_with_variations(scene: Dict, image_path: Optional[str],
                                     image_strength: float, base_seed: int,
                                     width: int, height: int, fps: int,
                                     profile: Dict, scene_index: int) -> Optional[str]:
    """Generate NUM_VARIATIONS clips for one shot, return the best quality one."""
    best_path, best_score = None, -1.0
    for vi in range(NUM_VARIATIONS):
        seed = base_seed + vi * 1000
        print(f"  Variation {vi+1}/{NUM_VARIATIONS}  seed={seed}")
        try:
            clip = generate_scene(
                scene_entry=scene, scene_index=scene_index,
                image_path=image_path, width=width, height=height,
                fps=fps, profile=profile, global_seed=seed,
                scene_cp={"completed_scenes":[],"clip_paths":{},"anchor_paths":{}})
            if clip:
                score, m = calculate_shot_metrics(clip)
                print(f"    Score={score:.3f}  {m}")
                if score > best_score:
                    best_score = score; best_path = clip
        except Exception as e:
            print(f"    Variation {vi+1} failed: {e}")
    if best_path:
        print(f"  Best variation score={best_score:.3f}")
    return best_path

# =============================================================================
# SECTION 31 — COLAB CELL RUNNERS
# =============================================================================
#
# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Install (run ONCE per runtime)
# ─────────────────────────────────────────────────────────────────────────────
# install_environment()
#
# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Download models (run ONCE per runtime or after /content cleared)
# ─────────────────────────────────────────────────────────────────────────────
# download_all_models()
#
# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Upload reference image & audio
# ─────────────────────────────────────────────────────────────────────────────
# from google.colab import files
# import shutil, os
# os.makedirs('/content/ComfyUI/input', exist_ok=True)
# up = files.upload()
# for fname in up:
#     shutil.move(f'/content/{fname}', f'/content/ComfyUI/input/{fname}')
#
# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Edit SCENE_JSON above (optional), then run this cell to generate
# ─────────────────────────────────────────────────────────────────────────────
# output = generate_multiscene_mv()
# if output:
#     from google.colab import files
#     files.download(output)
#
# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — After a crash, JUST RE-RUN CELL 4.
#   • RESUME=True (default) means completed scenes are automatically skipped.
#   • The scene checkpoint is stored at:
#       /content/ltx23_workspace/{PROJECT_NAME}_scene_checkpoint.json
#   • Nothing is regenerated unless you set RESUME=False.
# ─────────────────────────────────────────────────────────────────────────────

# =============================================================================
# SECTION 32 — DIRECT EXECUTION GUARD
# =============================================================================

if __name__ == "__main__":
    print("\nRunning LTX23_Director_2_0_MV_MultiScene.py directly...")
    print("Designed for Google Colab T4 GPU. Running setup...\n")

    # Step 1: Install environment
    install_environment()

    # Step 2: Download all required models
    download_all_models()

    # Step 3: Setup ComfyUI and load custom nodes
    setup_comfyui()
    import_custom_nodes()

    # Step 4: Generate multi-scene music video
    # Edit SCENE_JSON (Section 2) and @param settings (Section 1) before running.
    output = generate_multiscene_mv(
        scene_json  = SCENE_JSON,
        image_path  = IMAGE_PATH,
        audio_path  = AUDIO_PATH,
        quality_mode= QUALITY_MODE,
        seed        = SEED,
        fps         = FPS,
        width       = OUTPUT_WIDTH,
        height      = OUTPUT_HEIGHT,
    )

    if output:
        print(f"\n✓ Multi-scene generation complete: {output}")
    else:
        print("\n✗ Generation failed — review errors above.")
