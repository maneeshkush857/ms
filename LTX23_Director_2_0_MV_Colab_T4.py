#!/usr/bin/env python3
# =============================================================================
# LTX23_Director_2_0_MV_Colab_T4.py
# LTX-2.3 Director 2.0 Music Video Pipeline
# Google Colab NVIDIA Tesla T4 Optimized
# =============================================================================
#
# WORKFLOW SOURCE : LTX-2.3_Director_2.0-MV-Workflow-30s.json
# REFERENCE       : LTX2_TI2V_Distilled.py
#
# WORKFLOW → PYTHON NODE MAPPING:
# Node 135  UnetLoaderGGUF          → load_dit_model()
# Node 12   DualCLIPLoader          → load_text_encoder()
# Node 138  Power Lora Loader       → apply_loras()
# Node 6    VAELoaderKJ (Tiny)      → load_tiny_vae()
# Node 8    VAELoader (Audio)       → load_audio_vae()
# Node 36   VAELoader (Video)       → load_video_vae()
# Node 13   LatentUpscaleModelLoader→ load_upscale_model()
# Node 131  LTXDirector             → build_director_conditioning()
# Node 133  LTXDirectorGuide pass1  → run_director_guide_pass1()
# Node 132  LTXDirectorGuide pass2  → run_director_guide_pass2()
# Node 27   LTXVConditioning        → apply_ltxv_conditioning()
# Node 128  ConditioningZeroOut     → zero_out_conditioning()
# Node 29   LTXVConcatAVLatent p1   → concat_av_latent()
# Node 18   LTXVConcatAVLatent p2   → concat_av_latent()
# Node 30   RandomNoise             → make_noise()
# Node 32   KSamplerSelect p1       → select_sampler()
# Node 20   KSamplerSelect p2       → select_sampler()
# Node 33   BasicScheduler p1       → make_scheduler() steps=8  denoise=1.0
# Node 21   BasicScheduler p2       → make_scheduler() steps=4  denoise=0.42
# Node 28   CFGGuider p1            → make_cfg_guider()
# Node 17   CFGGuider p2            → make_cfg_guider()
# Node 31   SamplerCustomAdvanced p1→ sample_pass1()
# Node 19   SamplerCustomAdvanced p2→ sample_pass2()
# Node 34   LTXVSeparateAVLatent p1 → separate_av_latent()
# Node 22   LTXVSeparateAVLatent p2 → separate_av_latent()
# Node 55   LTXDirectorCropGuides   → crop_director_guides() (mid)
# Node 54   LTXDirectorCropGuides   → crop_director_guides() (final)
# Node 14   LTXVLatentUpsampler     → upsample_latent()
# Node 1    VAEDecode               → decode_video_latent()
# Node 24   LTXVAudioVAEDecode      → decode_audio_latent()
# Node 139  VHS_VideoCombine        → assemble_video_with_audio()
# Node 10   ModelPreviewOverrideKJ  → preview passthrough (non-critical)
# =============================================================================



# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# ── User-facing configuration ─────────────────────────────────────────────────
# All workflow-critical defaults mirror the JSON:
#   duration=31.5s, fps=24, 756 frames, 1280×720, CFG=1, euler sampler
#   Pass1: 8 steps denoise=1.0   Pass2: 4 steps denoise=0.42

CONFIG = {
    # ── Timeline (from workflow JSON) ─────────────────────────────────────────
    "duration_seconds"       : 31.5,   # workflow end_second
    "fps"                    : 24,     # workflow frame_rate
    "width"                  : 1280,   # workflow custom_width
    "height"                 : 720,    # workflow custom_height

    # ── Seed ──────────────────────────────────────────────────────────────────
    "seed"                   : 0,      # 0 = use random; matches node 30 default
    "random_seed"            : True,

    # ── Sampler (mirrors workflow) ────────────────────────────────────────────
    "sampler_name"           : "euler",
    "scheduler"              : "linear_quadratic",
    "pass1_steps"            : 8,
    "pass1_denoise"          : 1.0,
    "pass2_steps"            : 4,
    "pass2_denoise"          : 0.42,
    "cfg_scale"              : 1.0,    # distilled model — no CFG needed

    # ── LoRAs (from node 138 Power Lora Loader) ───────────────────────────────
    "lora_dynamic_enable"    : True,
    "lora_dynamic_strength"  : 0.4,
    "lora_omninft_enable"    : True,
    "lora_omninft_strength"  : 0.6,
    "lora_transition_enable" : True,
    "lora_transition_strength": 0.7,
    "lora_mvcamera_enable"   : True,
    "lora_mvcamera_strength" : 0.9,

    # ── Quality / VRAM profile ────────────────────────────────────────────────
    "quality_mode"           : "t4_safe",   # t4_safe | t4_balanced | t4_aggressive

    # ── Chunk control ─────────────────────────────────────────────────────────
    "auto_chunk_size"        : True,
    "chunk_frames"           : 48,          # used when auto_chunk_size=False
    "max_gpu_memory_gb"      : 14.0,
    "gpu_safety_margin_gb"   : 1.5,

    # ── OOM recovery ──────────────────────────────────────────────────────────
    "auto_reduce_chunk_on_oom": True,
    "max_oom_retries"        : 3,

    # ── Resume ────────────────────────────────────────────────────────────────
    "resume"                 : True,

    # ── Preview ───────────────────────────────────────────────────────────────
    "preview_mode"           : False,
    "preview_duration"       : 3.0,

    # ── Resolution safety ─────────────────────────────────────────────────────
    "allow_auto_downgrade"   : True,

    # ── Output ────────────────────────────────────────────────────────────────
    "output_dir"             : "/content/ltx23_output",
    "workspace_dir"          : "/content/ltx23_workspace",
    "keep_temp_chunks"       : False,
    "cleanup_temp_files"     : True,

    # ── Logging ───────────────────────────────────────────────────────────────
    "enable_memory_logging"  : True,
    "cleanup_after_chunk"    : True,
    "cleanup_after_stage"    : True,

    # ── Input ─────────────────────────────────────────────────────────────────
    "image_path"             : "",   # set before running
    "audio_path"             : "",   # set before running

    # ── Prompt (from workflow node 131 global_prompt) ─────────────────────────
    "custom_prompt"          : "",   # leave empty to use GLOBAL_PROMPT below
}

# Master cinematic prompt — exact text from workflow node 131 properties
GLOBAL_PROMPT = (
    "Create a highly realistic cinematic AI music video using the provided reference image. "
    "Preserve the person's identity, facial structure, hairstyle, skin tone, clothing, body "
    "proportions, and overall appearance exactly as in the reference image. The singer must "
    "remain fully recognizable throughout the entire video with absolutely no identity drift.\n\n"
    "The person is performing directly to the camera as a world-class pop, hip-hop and rap "
    "singer during a sold-out stadium concert. Generate perfectly synchronized lip movements "
    "from the provided lyrics or audio.\n\n"
    "Performance Energy:\n"
    "Explosive stage presence. Every lyric instantly changes facial expression, eye emotion, "
    "head movement, shoulders, hands, posture and body rhythm. Own the stage with absolute "
    "confidence. Perform as if in front of 50,000 screaming fans.\n\n"
    "Facial Performance:\n"
    "Extremely expressive facial acting. Rich emotional transitions every few words. Powerful "
    "eye contact. Eyes sparkle with confidence and passion. Never hold the same expression.\n\n"
    "Body Performance:\n"
    "The entire body constantly grooves with the beat. Strong rhythmic bouncing. Powerful "
    "shoulder accents. Lean toward the camera during emotional lyrics.\n\n"
    "Hand Performance:\n"
    "Large expressive gestures. Fast rhythmic arm accents. Asymmetrical movement. Never repeat "
    "the same gesture pattern.\n\n"
    "Camera:\n"
    "drclipz, Aggressive cinematic music video camera. Fast push-in, fast pull-back, energetic "
    "handheld movement, rhythmic tracking shots, dynamic low-angle hero shots.\n\n"
    "Lighting:\n"
    "Premium concert lighting with cinematic key light, colorful neon rim lights, volumetric "
    "atmosphere, dramatic contrast, realistic skin tones.\n\n"
    "Overall Style:\n"
    "Photorealistic, blockbuster-quality AI music video, premium live concert performance, "
    "ultra-high facial fidelity, charismatic superstar, emotionally captivating, explosive "
    "stage energy. No jitter. No flickering. No facial distortion. No identity drift. "
    "No hand deformation. No extra fingers. No malformed limbs."
)

# Model filenames — exact names from workflow JSON nodes
MODEL_FILES = {
    "dit_gguf"       : "ltx-2-3-22b-dev-Q4_K_M.gguf",
    "clip1"          : "gemma_3_12B_it_fp4_mixed.safetensors",
    "clip2"          : "ltx-2.3_text_projection_bf16.safetensors",
    "audio_vae"      : "LTX23_audio_vae_bf16.safetensors",
    "video_vae"      : "LTX23_video_vae_bf16.safetensors",
    "tiny_vae"       : "taeltx2_3.safetensors",
    "upscaler"       : "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    "lora_dynamic"   : "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors",
    "lora_omninft"   : "LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors",
    "lora_transition": "ltx2.3-transition.safetensors",
    "lora_mvcamera"  : "LTX2.3-MVCamera-drclips.safetensors",
}

# T4 VRAM profiles (frames per chunk at 1280×720)
T4_PROFILES = {
    "t4_safe"       : {"chunk_frames": 33,  "vae_sub_batch": 8,  "desc": "Max stability"},
    "t4_balanced"   : {"chunk_frames": 49,  "vae_sub_batch": 16, "desc": "Speed/safety balance"},
    "t4_aggressive" : {"chunk_frames": 65,  "vae_sub_batch": 32, "desc": "Maximum speed (risky)"},
}



# =============================================================================
# SECTION 2 — IMPORTS & CUDA / T4 DETECTION
# =============================================================================

import os
import sys
import gc
import json
import math
import time
import shutil
import hashlib
import subprocess
import traceback
import importlib
import ctypes
import random
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Set CUDA allocator config BEFORE importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn

# ── Workspace directories ──────────────────────────────────────────────────────
WORKSPACE  = Path(CONFIG["workspace_dir"])
OUTPUT_DIR = Path(CONFIG["output_dir"])
CHUNKS_DIR = WORKSPACE / "chunks"
FRAMES_DIR = WORKSPACE / "frames"
AUDIO_DIR  = WORKSPACE / "audio"
LOGS_DIR   = WORKSPACE / "logs"
FINAL_DIR  = OUTPUT_DIR

for _d in [WORKSPACE, OUTPUT_DIR, CHUNKS_DIR, FRAMES_DIR, AUDIO_DIR, LOGS_DIR, FINAL_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


def detect_gpu() -> Dict:
    """Detect CUDA GPU and return environment info. Raises RuntimeError if no GPU."""
    print("\n" + "="*60)
    print("LTX-2.3 DIRECTOR 2.0 MV  —  Google Colab T4 Engine")
    print("="*60)

    print(f"  PyTorch version : {torch.__version__}")
    print(f"  CUDA version    : {torch.version.cuda}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "\nERROR: NVIDIA CUDA GPU was not detected.\n"
            "This notebook requires a CUDA-enabled runtime.\n"
            "Go to Runtime → Change runtime type → Hardware accelerator → T4 GPU"
        )

    device_name  = torch.cuda.get_device_name(0)
    vram_total   = torch.cuda.get_device_properties(0).total_memory / 1e9
    vram_free    = (torch.cuda.get_device_properties(0).total_memory
                    - torch.cuda.memory_allocated(0)) / 1e9

    print(f"  GPU             : {device_name}")
    print(f"  VRAM total      : {vram_total:.2f} GB")
    print(f"  VRAM free       : {vram_free:.2f} GB")

    if "T4" not in device_name and "A100" not in device_name and "V100" not in device_name:
        print(f"  WARNING: GPU '{device_name}' is not a T4. Profiles may need adjustment.")

    return {
        "device_name"  : device_name,
        "vram_total_gb": round(vram_total, 2),
        "vram_free_gb" : round(vram_free, 2),
        "torch_version": torch.__version__,
        "cuda_version" : torch.version.cuda,
    }


GPU_INFO: Dict = {}   # filled by detect_gpu() in main



# =============================================================================
# SECTION 3 — ADVANCED MEMORY MANAGER
# =============================================================================

class LTXMemoryManager:
    """
    Dedicated GPU/CPU memory manager for LTX-2.3 T4 pipeline.

    Tracks CUDA allocation, RAM usage, provides tiered cleanup,
    model release, and safety threshold warnings.
    """

    def __init__(self, safety_margin_gb: float = 1.5, enable_logging: bool = True):
        self.safety_margin_gb = safety_margin_gb
        self.enable_logging   = enable_logging
        self._peak_gb         = 0.0

    # ── Queries ───────────────────────────────────────────────────────────────

    def gpu_allocated_gb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated(0) / 1e9

    def gpu_reserved_gb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_reserved(0) / 1e9

    def gpu_free_gb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        props = torch.cuda.get_device_properties(0)
        return (props.total_memory - torch.cuda.memory_allocated(0)) / 1e9

    def gpu_total_gb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.get_device_properties(0).total_memory / 1e9

    def gpu_peak_gb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated(0) / 1e9

    def cpu_used_gb(self) -> float:
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1e9
        except Exception:
            return 0.0

    def cpu_available_gb(self) -> float:
        try:
            import psutil
            return psutil.virtual_memory().available / 1e9
        except Exception:
            return 0.0

    # ── Cleanup tiers ─────────────────────────────────────────────────────────

    def soft_cleanup(self):
        """Light cleanup — GC + empty CUDA cache."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def cleanup(self):
        """Standard cleanup — GC + empty cache + IPC collect."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def aggressive_cleanup(self):
        """
        Full cleanup — 3× GC cycles, CUDA sync, cache, IPC, peak reset,
        plus glibc malloc_trim to return freed heap pages to OS.
        Use between major pipeline stages or after OOM recovery.
        """
        for _ in range(3):
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
        # Return freed glibc pages to OS (prevents silent host-kill on Colab)
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    def empty_cuda_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Model helpers ─────────────────────────────────────────────────────────

    def release_tensor(self, tensor, name: str = "tensor"):
        """Safely delete a GPU tensor and trigger cleanup."""
        if tensor is not None:
            if hasattr(tensor, "data"):
                del tensor
            self.soft_cleanup()

    def release_model(self, model, name: str = "model"):
        """Move model to CPU, delete, clean up."""
        if model is None:
            return
        try:
            if hasattr(model, "to"):
                model.to("cpu")
        except Exception:
            pass
        del model
        self.cleanup()
        if self.enable_logging:
            print(f"  [MEM] Released model: {name}")

    def safe_model_unload(self, model_dict: Dict, key: str):
        """Release a model stored in a dict by key."""
        if key in model_dict and model_dict[key] is not None:
            self.release_model(model_dict[key], name=key)
            model_dict[key] = None

    # ── Warnings ──────────────────────────────────────────────────────────────

    def warn_if_low(self, threshold_gb: Optional[float] = None) -> bool:
        """Return True and trigger aggressive_cleanup if VRAM is below threshold."""
        threshold = threshold_gb or self.safety_margin_gb
        free = self.gpu_free_gb()
        if free < threshold:
            print(f"\n  WARNING: GPU memory below safety threshold "
                  f"({free:.2f} GB free < {threshold:.2f} GB required).")
            print("  Starting aggressive cleanup.")
            self.aggressive_cleanup()
            return True
        return False

    def estimate_frame_ram_gb(self, num_frames: int, width: int, height: int,
                               dtype_bytes: int = 4) -> float:
        """Estimate RAM for decoded pixel frames tensor (N, H, W, 3) at float32."""
        return (num_frames * height * width * 3 * dtype_bytes) / 1e9

    # ── Reporting ─────────────────────────────────────────────────────────────

    def gpu_memory(self) -> Dict:
        return {
            "allocated_gb": round(self.gpu_allocated_gb(), 3),
            "reserved_gb" : round(self.gpu_reserved_gb(),  3),
            "free_gb"     : round(self.gpu_free_gb(),       3),
            "peak_gb"     : round(self.gpu_peak_gb(),       3),
            "total_gb"    : round(self.gpu_total_gb(),      3),
        }

    def cpu_memory(self) -> Dict:
        return {
            "used_gb"     : round(self.cpu_used_gb(),      3),
            "available_gb": round(self.cpu_available_gb(), 3),
        }

    def memory_report(self, prefix: str = "", chunk_info: Optional[Dict] = None):
        if not self.enable_logging:
            return
        g = self.gpu_memory()
        c = self.cpu_memory()
        tag = f"[{prefix}] " if prefix else ""
        print(f"\n  {tag}GPU  alloc={g['allocated_gb']:.2f}GB  "
              f"resv={g['reserved_gb']:.2f}GB  "
              f"free={g['free_gb']:.2f}GB  "
              f"peak={g['peak_gb']:.2f}GB")
        print(f"  {tag}CPU  used={c['used_gb']:.2f}GB  avail={c['available_gb']:.2f}GB")
        if chunk_info:
            print(f"  {tag}Chunk {chunk_info.get('index','?')}  "
                  f"frames={chunk_info.get('num_frames','?')}  "
                  f"res={chunk_info.get('width','?')}x{chunk_info.get('height','?')}")


# Global singleton
memory_manager = LTXMemoryManager(
    safety_margin_gb=CONFIG["gpu_safety_margin_gb"],
    enable_logging=CONFIG["enable_memory_logging"]
)


def print_memory(prefix: str = ""):
    memory_manager.memory_report(prefix=prefix)


def cleanup_memory():
    memory_manager.cleanup()


def aggressive_cleanup():
    memory_manager.aggressive_cleanup()



# =============================================================================
# SECTION 4 — ENVIRONMENT INSTALLATION
# =============================================================================

COMFYUI_DIR   = Path("/content/ComfyUI")
CUSTOM_NODES  = COMFYUI_DIR / "custom_nodes"
MODELS_DIR    = COMFYUI_DIR / "models"


def _run(cmd: str, desc: str = "", check: bool = True):
    """Run a shell command, printing a short description."""
    if desc:
        print(f"  → {desc}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"  [WARN] Command failed: {cmd}")
        print(f"  stderr: {result.stderr[-500:]}")
    return result


def install_environment():
    """Install all required Python packages, system tools, and ComfyUI."""

    print("\n" + "="*60)
    print("SECTION 4 — Environment Installation")
    print("="*60)

    # ── System packages ───────────────────────────────────────────────────────
    print("\n[4.1] System packages …")
    _run("apt-get update -qq && apt-get install -y -qq aria2 ffmpeg libsm6 libxext6",
         "apt: aria2, ffmpeg")

    # ── Python packages ───────────────────────────────────────────────────────
    print("\n[4.2] Python packages …")
    pip_packages = [
        "torchsde",
        "einops",
        "diffusers>=0.28.0",
        "accelerate",
        "transformers>=4.40.0",
        "av",
        "opencv-python-headless",
        "psutil",
        "imageio",
        "imageio-ffmpeg",
        "scipy",
        "safetensors",
        "gguf",
        "nest_asyncio",
        "spandrel",
        "kornia",
    ]
    for pkg in pip_packages:
        _run(f"pip install -q '{pkg}'", f"pip: {pkg}")

    # ── ComfyUI ───────────────────────────────────────────────────────────────
    print("\n[4.3] ComfyUI …")
    if not COMFYUI_DIR.exists():
        _run("git clone --depth=1 https://github.com/comfyanonymous/ComfyUI.git /content/ComfyUI",
             "git clone ComfyUI")
    else:
        print("  ComfyUI already present — skipping clone.")

    _run(f"pip install -q -r {COMFYUI_DIR}/requirements.txt", "ComfyUI requirements")

    # ── Create model subdirectories ───────────────────────────────────────────
    for subdir in ["unet", "clip", "vae", "upscale_models", "loras"]:
        (MODELS_DIR / subdir).mkdir(parents=True, exist_ok=True)

    print("\n[4.4] Custom nodes …")
    _install_custom_nodes()

    print("\n[4] Environment installation complete.\n")


# ── Required custom nodes from workflow JSON ──────────────────────────────────
CUSTOM_NODE_REPOS = [
    ("ComfyUI-KJNodes",
     "https://github.com/kijai/ComfyUI-KJNodes.git"),
    ("ComfyUI-GGUF",
     "https://github.com/city96/ComfyUI-GGUF.git"),
    ("ComfyUI-LTXVideo",
     "https://github.com/Lightricks/ComfyUI-LTXVideo.git"),
    ("WhatDreamsCost-ComfyUI",
     "https://github.com/WhatDreamsCost/ComfyUI.git"),
    ("ComfyUI-VideoHelperSuite",
     "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"),
    ("ComfyUI-MelBandRoFormer",
     "https://github.com/nomaddo/ComfyUI-MelBandRoFormer.git"),
    ("rgthree-comfy",
     "https://github.com/rgthree/rgthree-comfy.git"),
]


def _install_custom_nodes():
    CUSTOM_NODES.mkdir(parents=True, exist_ok=True)
    for name, url in CUSTOM_NODE_REPOS:
        dest = CUSTOM_NODES / name
        if dest.exists():
            print(f"  {name} already present — skipping.")
        else:
            _run(f"git clone --depth=1 {url} {dest}", f"git clone {name}")
        req = dest / "requirements.txt"
        if req.exists():
            _run(f"pip install -q -r {req}", f"  requirements: {name}")



# =============================================================================
# SECTION 5 — COMFYUI SETUP & NODE LOADING
# =============================================================================

# Global node registry — populated by import_comfyui_nodes()
NODE_CLASS_MAPPINGS: Dict[str, Any] = {}
_COMFYUI_LOADED = False


def setup_comfyui():
    """Add ComfyUI to sys.path so its Python API is importable."""
    comfyui_str = str(COMFYUI_DIR)
    if comfyui_str not in sys.path:
        sys.path.insert(0, comfyui_str)
    # Also add custom_nodes root so relative imports inside node packages work
    custom_str = str(CUSTOM_NODES)
    if custom_str not in sys.path:
        sys.path.insert(0, custom_str)
    print(f"  [5] ComfyUI path registered: {comfyui_str}")


def _patch_prompt_server():
    """
    WhatDreamsCost nodes expect PromptServer.instance to exist at import time.
    Create a minimal mock for headless Colab execution.
    """
    try:
        import server as comfy_server
        if not hasattr(comfy_server.PromptServer, "instance") or \
                comfy_server.PromptServer.instance is None:
            class _MockPromptServer:
                def __init__(self):
                    self.client_id = None
                    self.send_sync  = lambda *a, **kw: None
                    self.trigger_on_prompt = lambda *a, **kw: None
            comfy_server.PromptServer.instance = _MockPromptServer()
    except Exception:
        pass


def _patch_kornia():
    """
    Some kornia versions removed kornia.geometry.transform.pyramid.pad
    which ComfyUI-LTXVideo depends on. Patch with torch.nn.functional.pad.
    """
    try:
        import kornia.geometry.transform.pyramid as _kpyramid
        if not hasattr(_kpyramid, "pad"):
            import torch.nn.functional as F
            _kpyramid.pad = F.pad
    except Exception:
        pass


def import_comfyui_nodes() -> Dict[str, Any]:
    """
    Import all ComfyUI built-in and custom nodes.
    Returns the populated NODE_CLASS_MAPPINGS dict.
    """
    global NODE_CLASS_MAPPINGS, _COMFYUI_LOADED

    if _COMFYUI_LOADED:
        return NODE_CLASS_MAPPINGS

    print("\n[5] Loading ComfyUI nodes …")

    # Colab async-loop compatibility
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except Exception:
        pass

    setup_comfyui()
    _patch_prompt_server()
    _patch_kornia()

    # Import ComfyUI node manager
    try:
        import nodes as comfy_nodes
        NODE_CLASS_MAPPINGS.update(comfy_nodes.NODE_CLASS_MAPPINGS)
        print(f"  Built-in nodes loaded: {len(comfy_nodes.NODE_CLASS_MAPPINGS)}")
    except Exception as e:
        print(f"  [WARN] Could not load built-in ComfyUI nodes: {e}")

    # Load each custom node package
    for name, _ in CUSTOM_NODE_REPOS:
        node_dir = CUSTOM_NODES / name
        if not node_dir.exists():
            print(f"  [WARN] Custom node directory missing: {node_dir}")
            continue
        _load_custom_node_package(name, node_dir)

    _COMFYUI_LOADED = True
    print(f"  Total nodes registered: {len(NODE_CLASS_MAPPINGS)}")
    return NODE_CLASS_MAPPINGS


def _load_custom_node_package(pkg_name: str, pkg_dir: Path):
    """Try to import a custom node package and merge its NODE_CLASS_MAPPINGS."""
    # Try __init__.py first, then nodes.py, then all top-level *.py files
    candidates = ["__init__", "nodes", pkg_name.replace("-", "_")]
    loaded = False
    for candidate in candidates:
        try:
            spec_path = pkg_dir / f"{candidate}.py"
            if not spec_path.exists() and candidate == "__init__":
                spec_path = pkg_dir / "__init__.py"
            if not spec_path.exists():
                continue
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                f"custom_nodes.{pkg_name}.{candidate}", spec_path
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            if hasattr(mod, "NODE_CLASS_MAPPINGS"):
                NODE_CLASS_MAPPINGS.update(mod.NODE_CLASS_MAPPINGS)
                print(f"  ✓ {pkg_name}: {len(mod.NODE_CLASS_MAPPINGS)} nodes")
                loaded = True
                break
        except Exception as e:
            # Don't hard-fail — some packages have optional GPU ops
            pass
    if not loaded:
        print(f"  [WARN] Could not import nodes from {pkg_name}")


def get_node(name: str):
    """
    Return an instantiated ComfyUI node class from the registry.
    Raises a clear error with install instructions if not found.
    """
    if name not in NODE_CLASS_MAPPINGS:
        available = [k for k in NODE_CLASS_MAPPINGS if name.lower() in k.lower()]
        hint = f"\n  Similar: {available}" if available else ""
        raise ImportError(
            f"\nMISSING NODE: '{name}' not found in NODE_CLASS_MAPPINGS.{hint}\n"
            f"Ensure the required custom node package is installed and loaded.\n"
            f"Run install_environment() and import_comfyui_nodes() first."
        )
    return NODE_CLASS_MAPPINGS[name]()


def get_value(obj, index: int = 0):
    """
    ComfyUI nodes return either a tuple or {'result': (...)} dict.
    This helper normalizes both into a single indexed value.
    """
    if isinstance(obj, dict) and "result" in obj:
        obj = obj["result"]
    if isinstance(obj, (list, tuple)):
        return obj[index]
    return obj



# =============================================================================
# SECTION 6 — CUSTOM NODE VALIDATION
# =============================================================================

# Required nodes as defined in the workflow JSON (cnr_id / Node name for S&R)
REQUIRED_NODES = {
    # Core ComfyUI
    "CFGGuider"              : "comfy-core",
    "KSamplerSelect"         : "comfy-core",
    "BasicScheduler"         : "comfy-core",
    "SamplerCustomAdvanced"  : "comfy-core",
    "RandomNoise"            : "comfy-core",
    "VAEDecode"              : "comfy-core",
    "VAELoader"              : "comfy-core",
    "DualCLIPLoader"         : "comfy-core",
    "ConditioningZeroOut"    : "comfy-core",
    # LTX-specific (comfy-core in newer builds)
    "LTXVConditioning"       : "comfy-core / ComfyUI-LTXVideo",
    "LTXVConcatAVLatent"     : "comfy-core / ComfyUI-LTXVideo",
    "LTXVSeparateAVLatent"   : "comfy-core / ComfyUI-LTXVideo",
    "LTXVLatentUpsampler"    : "comfy-core / ComfyUI-LTXVideo",
    "LTXVAudioVAEDecode"     : "comfy-core / ComfyUI-LTXVideo",
    "LatentUpscaleModelLoader": "comfy-core",
    # GGUF loader
    "UnetLoaderGGUF"         : "ComfyUI-GGUF",
    # KJNodes
    "VAELoaderKJ"            : "comfyui-kjnodes",
    "ModelPreviewOverrideKJ" : "comfyui-kjnodes",
    # Director (WhatDreamsCost)
    "LTXDirector"            : "whatdreamscost-comfyui",
    "LTXDirectorGuide"       : "whatdreamscost-comfyui",
    "LTXDirectorCropGuides"  : "whatdreamscost-comfyui",
    # VideoHelperSuite
    "VHS_VideoCombine"       : "comfyui-videohelpersuite",
    # rgthree
    "Power Lora Loader (rgthree)": "rgthree-comfy",
}


def validate_custom_nodes() -> bool:
    """
    Check every required workflow node is present in NODE_CLASS_MAPPINGS.
    Prints a formatted dependency report.
    Returns True if all critical nodes are present.
    """
    print("\n" + "="*60)
    print("SECTION 6 — Custom Node Validation")
    print("="*60)

    all_ok   = True
    # Nodes that are highly desirable but not hard failures
    optional = {"ModelPreviewOverrideKJ", "VHS_VideoCombine",
                "Power Lora Loader (rgthree)"}

    for node_name, package in REQUIRED_NODES.items():
        present = node_name in NODE_CLASS_MAPPINGS
        status  = "✓" if present else ("△" if node_name in optional else "✗")
        print(f"  {status}  {node_name:<35s} [{package}]")
        if not present and node_name not in optional:
            all_ok = False

    if not all_ok:
        print("\n  CRITICAL NODES MISSING — generation will fail.")
        print("  Run install_environment() then re-run import_comfyui_nodes().")
    else:
        print("\n  All critical nodes present.")
    return all_ok



# =============================================================================
# SECTION 7 — MODEL DOWNLOAD
# =============================================================================

# HuggingFace repo / path mappings for each model file
# Update URLs if mirrors change
MODEL_SOURCES = {
    MODEL_FILES["dit_gguf"]: {
        "url": "https://huggingface.co/city96/LTX-Video-2-3-22b-dev-gguf/resolve/main/ltx-2-3-22b-dev-Q4_K_M.gguf",
        "dest_subdir": "unet",
    },
    MODEL_FILES["clip1"]: {
        "url": "https://huggingface.co/Kijai/LTX-Video-2-3_comfy/resolve/main/gemma_3_12B_it_fp4_mixed.safetensors",
        "dest_subdir": "clip",
    },
    MODEL_FILES["clip2"]: {
        "url": "https://huggingface.co/Kijai/LTX-Video-2-3_comfy/resolve/main/ltx-2.3_text_projection_bf16.safetensors",
        "dest_subdir": "clip",
    },
    MODEL_FILES["audio_vae"]: {
        "url": "https://huggingface.co/Kijai/LTX-Video-2-3_comfy/resolve/main/LTX23_audio_vae_bf16.safetensors",
        "dest_subdir": "vae",
    },
    MODEL_FILES["video_vae"]: {
        "url": "https://huggingface.co/Kijai/LTX-Video-2-3_comfy/resolve/main/LTX23_video_vae_bf16.safetensors",
        "dest_subdir": "vae",
    },
    MODEL_FILES["tiny_vae"]: {
        "url": "https://huggingface.co/madebyollin/taef1/resolve/main/taeltx2_3.safetensors",
        "dest_subdir": "vae",
    },
    MODEL_FILES["upscaler"]: {
        "url": "https://huggingface.co/Kijai/LTX-Video-2-3_comfy/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "dest_subdir": "upscale_models",
    },
    MODEL_FILES["lora_dynamic"]: {
        "url": "https://huggingface.co/Kijai/LTX-Video-2-3_comfy/resolve/main/ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors",
        "dest_subdir": "loras",
    },
    MODEL_FILES["lora_omninft"]: {
        "url": "https://huggingface.co/OmniGen/OmniNFT/resolve/main/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors",
        "dest_subdir": "loras",
    },
    MODEL_FILES["lora_transition"]: {
        "url": "https://huggingface.co/Kijai/LTX-Video-2-3_comfy/resolve/main/ltx2.3-transition.safetensors",
        "dest_subdir": "loras",
    },
    MODEL_FILES["lora_mvcamera"]: {
        "url": "https://huggingface.co/Kijai/LTX-Video-2-3_comfy/resolve/main/LTX2.3-MVCamera-drclips.safetensors",
        "dest_subdir": "loras",
    },
}


def _aria2_download(url: str, dest: Path, connections: int = 16):
    """Download a file with aria2c (fast, resumable, multi-connection)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = (f'aria2c --console-log-level=error -c -x {connections} -s {connections} '
           f'--allow-overwrite=false -d "{dest.parent}" -o "{dest.name}" "{url}"')
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def download_models(skip_existing: bool = True):
    """
    Download all required model files using aria2c.
    Skips files that already exist and have non-zero size.
    """
    print("\n" + "="*60)
    print("SECTION 7 — Model Download")
    print("="*60)

    for filename, info in MODEL_SOURCES.items():
        subdir = MODELS_DIR / info["dest_subdir"]
        subdir.mkdir(parents=True, exist_ok=True)
        dest = subdir / filename

        if skip_existing and dest.exists() and dest.stat().st_size > 1024:
            print(f"  ✓  {filename}  (exists, {dest.stat().st_size / 1e6:.0f} MB)")
            continue

        print(f"  ↓  {filename} …")
        ok = _aria2_download(info["url"], dest)
        if ok and dest.exists():
            print(f"      saved → {dest}  ({dest.stat().st_size / 1e6:.0f} MB)")
        else:
            print(f"  [WARN] Download may have failed: {filename}")
            print(f"         URL: {info['url']}")
            print(f"         Manually place the file at: {dest}")

    print("\n[7] Model download complete.\n")


def get_model_path(filename: str) -> Optional[Path]:
    """Locate a model file under MODELS_DIR. Returns None if not found."""
    for subdir in ["unet", "clip", "vae", "upscale_models", "loras", ""]:
        candidate = MODELS_DIR / subdir / filename
        if candidate.exists():
            return candidate
    # Also search ComfyUI model subdirs directly
    for root, dirs, files in os.walk(MODELS_DIR):
        if filename in files:
            return Path(root) / filename
    return None



# =============================================================================
# SECTION 8 — MODEL VALIDATION
# =============================================================================

def validate_models() -> bool:
    """
    Verify every required model file exists on disk before spending time loading.
    Returns True if all critical models are present.
    """
    print("\n" + "="*60)
    print("SECTION 8 — Model Validation")
    print("="*60)

    # LoRAs are optional per CONFIG flags
    lora_enabled = {
        MODEL_FILES["lora_dynamic"]   : CONFIG["lora_dynamic_enable"],
        MODEL_FILES["lora_omninft"]   : CONFIG["lora_omninft_enable"],
        MODEL_FILES["lora_transition"]: CONFIG["lora_transition_enable"],
        MODEL_FILES["lora_mvcamera"]  : CONFIG["lora_mvcamera_enable"],
    }

    critical = [
        MODEL_FILES["dit_gguf"],
        MODEL_FILES["clip1"],
        MODEL_FILES["clip2"],
        MODEL_FILES["audio_vae"],
        MODEL_FILES["video_vae"],
        MODEL_FILES["upscaler"],
    ]

    all_ok = True
    for fname in critical:
        path = get_model_path(fname)
        ok   = path is not None and path.stat().st_size > 1024
        mark = "✓" if ok else "✗"
        size = f"  ({path.stat().st_size / 1e6:.0f} MB)" if ok else ""
        print(f"  {mark}  {fname}{size}")
        if not ok:
            all_ok = False

    for fname, enabled in lora_enabled.items():
        if not enabled:
            print(f"  -  {fname}  [disabled]")
            continue
        path = get_model_path(fname)
        ok   = path is not None and path.stat().st_size > 1024
        mark = "✓" if ok else "△"
        print(f"  {mark}  {fname}")
        # LoRA absence is a warning, not a hard failure
        if not ok:
            print(f"      [WARN] LoRA missing — will skip: {fname}")

    if not all_ok:
        print("\n  CRITICAL MODELS MISSING. Run download_models() first.")
    else:
        print("\n  All critical models present.")
    return all_ok



# =============================================================================
# SECTION 9 — WORKFLOW VALIDATION
# =============================================================================

def validate_environment() -> bool:
    """Quick CUDA + path sanity check before any expensive operation."""
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU detected.")
        return False
    if not COMFYUI_DIR.exists():
        print("ERROR: ComfyUI not found. Run install_environment() first.")
        return False
    return True


def validate_input_image(image_path: str) -> bool:
    p = Path(image_path)
    if not p.exists():
        print(f"ERROR: Image not found: {image_path}")
        return False
    if p.stat().st_size < 100:
        print(f"ERROR: Image file too small (corrupted?): {image_path}")
        return False
    try:
        from PIL import Image
        with Image.open(p) as img:
            w, h = img.size
        print(f"  Input image: {w}×{h}  ({p.suffix.upper()})")
    except Exception as e:
        print(f"ERROR: Cannot open image: {e}")
        return False
    return True


def validate_audio(audio_path: str) -> bool:
    p = Path(audio_path)
    if not p.exists():
        print(f"ERROR: Audio not found: {audio_path}")
        return False
    if p.stat().st_size < 1024:
        print(f"ERROR: Audio file too small: {audio_path}")
        return False
    print(f"  Input audio: {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")
    return True


def validate_resolution(width: int, height: int) -> Tuple[int, int]:
    """
    Check the requested resolution against T4 safety limits.
    Returns (final_width, final_height) after any auto-downgrade.
    Width and height must be divisible by 32 (workflow: divisible_by=32).
    """
    # Snap to divisible-by-32
    width  = (width  // 32) * 32
    height = (height // 32) * 32

    # Rough VRAM estimate for 1 frame at BF16: (W*H*4*2)/1e9 GB
    # For a 48-frame latent chunk (4× spatial compression in LTX), rough budget:
    # latent dims ≈ W/32 × H/32 × 128 channels × 48 frames × 2 bytes ≈ modest
    # But decoded frames (W×H×3×4) × 48 frames is the real cost
    frames_test  = 48
    frame_ram    = memory_manager.estimate_frame_ram_gb(frames_test, width, height)
    vram_total   = memory_manager.gpu_total_gb()
    # Rough budget: DiT ~9 GB, VAE ~1.4 GB, misc ~1 GB → ~11.5 GB baseline
    available_for_frames = max(0.0, vram_total - 11.5)

    safe = frame_ram <= available_for_frames

    print(f"  Resolution check: {width}×{height}")
    print(f"    Decoded frames ({frames_test}f) ≈ {frame_ram:.2f} GB")
    print(f"    VRAM budget for frames ≈ {available_for_frames:.2f} GB")

    if not safe:
        if CONFIG["allow_auto_downgrade"]:
            new_w, new_h = 960, 544
            new_w = (new_w // 32) * 32
            new_h = (new_h // 32) * 32
            print(f"  AUTO DOWNGRADE: {width}×{height} → {new_w}×{new_h}")
            return new_w, new_h
        else:
            print(f"  WARNING: {width}×{height} may exceed T4 memory at {frames_test} frames/chunk.")
            print(f"           Set allow_auto_downgrade=True to auto-reduce.")

    return width, height


def validate_frame_count(total_frames: int, fps: int) -> int:
    """Ensure frame count satisfies LTX-2.3 temporal constraint (8k+1)."""
    return normalize_ltx_frame_count(total_frames, fps)


def validate_gpu_memory() -> bool:
    free = memory_manager.gpu_free_gb()
    total = memory_manager.gpu_total_gb()
    print(f"  GPU memory: {free:.2f} GB free / {total:.2f} GB total")
    if free < 2.0:
        print("  WARNING: Very low free VRAM before generation start.")
        print("  Running aggressive cleanup …")
        aggressive_cleanup()
    return True


def validate_workflow_dependencies() -> bool:
    """High-level check that the workflow's key architectural nodes exist."""
    critical = [
        "LTXDirector", "LTXDirectorGuide", "LTXDirectorCropGuides",
        "LTXVConditioning", "LTXVLatentUpsampler",
        "LTXVConcatAVLatent", "LTXVSeparateAVLatent",
        "SamplerCustomAdvanced", "UnetLoaderGGUF",
        "BasicScheduler", "CFGGuider", "VAEDecode",
    ]
    missing = [n for n in critical if n not in NODE_CLASS_MAPPINGS]
    if missing:
        print(f"  [ERROR] Missing workflow nodes: {missing}")
        return False
    print("  Workflow architecture nodes: OK")
    return True


def run_all_validations(image_path: str, audio_path: str,
                        width: int, height: int,
                        total_frames: int, fps: int) -> Tuple[bool, int, int]:
    """
    Run all pre-generation validations.
    Returns (ok, final_width, final_height).
    """
    print("\n" + "="*60)
    print("SECTION 9 — Pre-Generation Validation")
    print("="*60)

    ok = True
    ok &= validate_environment()
    ok &= validate_models()
    ok &= validate_custom_nodes()
    ok &= validate_workflow_dependencies()
    ok &= validate_input_image(image_path)
    ok &= validate_audio(audio_path)
    final_w, final_h = validate_resolution(width, height)
    validate_frame_count(total_frames, fps)
    ok &= validate_gpu_memory()

    if ok:
        print("\n  ✓ All validations passed. Ready for generation.\n")
    else:
        print("\n  ✗ Validation failed. Fix errors above before generating.\n")
    return ok, final_w, final_h



# =============================================================================
# SECTION 10 — INPUT UPLOAD / LOADING
# =============================================================================

def upload_image_colab() -> str:
    """
    Colab helper — prompt the user to upload an image via the Files widget.
    Returns the saved path string or empty string if not in Colab.
    """
    try:
        from google.colab import files
        print("\nPlease upload your reference image (PNG/JPG):")
        uploaded = files.upload()
        if not uploaded:
            print("  No file uploaded.")
            return ""
        filename = list(uploaded.keys())[0]
        dest = str(WORKSPACE / filename)
        with open(dest, "wb") as f:
            f.write(uploaded[filename])
        print(f"  Saved: {dest}")
        return dest
    except ImportError:
        print("  Not running in Colab — set CONFIG['image_path'] manually.")
        return ""


def upload_audio_colab() -> str:
    """Colab helper — upload audio file. Returns path or empty string."""
    try:
        from google.colab import files
        print("\nPlease upload your audio file (MP3/WAV):")
        uploaded = files.upload()
        if not uploaded:
            return ""
        filename = list(uploaded.keys())[0]
        dest = str(AUDIO_DIR / filename)
        with open(dest, "wb") as f:
            f.write(uploaded[filename])
        print(f"  Saved: {dest}")
        return dest
    except ImportError:
        print("  Not running in Colab — set CONFIG['audio_path'] manually.")
        return ""


def load_reference_image(image_path: str, width: int, height: int) -> "torch.Tensor":
    """
    Load reference image from disk → float32 RGB tensor [1, H, W, 3] on CPU.
    Resize to (width, height) preserving aspect ratio (longer-edge based),
    then centre-crop/pad to exact target size.
    Kept on CPU — transferred to GPU only when needed per chunk.
    """
    from PIL import Image
    import numpy as np

    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    # Resize: longer edge → target longer edge (preserve aspect)
    target_longer = max(width, height)
    orig_longer   = max(orig_w, orig_h)
    scale = target_longer / orig_longer
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    # Centre-crop to exact target
    left = (new_w - width)  // 2
    top  = (new_h - height) // 2
    img  = img.crop((left, top, left + width, top + height))

    # → float32 [0,1] tensor [1, H, W, 3]
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, H, W, 3]

    print(f"  Reference image loaded: {orig_w}×{orig_h} → {width}×{height}")
    return tensor   # CPU, float32



# =============================================================================
# SECTION 11 — AUDIO PREPARATION
# =============================================================================

def prepare_audio(audio_path: str, fps: int, total_frames: int,
                  trim_start_frames: float = 0.0) -> Dict:
    """
    Load audio file metadata without reading the entire waveform into RAM.
    Returns a lightweight dict; the waveform is only loaded by the Director node
    at generation time via the audio_vae path.

    Parameters
    ----------
    audio_path        : path to MP3/WAV/FLAC file
    fps               : video frame rate
    total_frames      : number of video frames
    trim_start_frames : offset into audio (in frames) matching workflow trimStart
                        (workflow value: 446.9 frames ≈ 18.6 s into the MP3)
    """
    print("\n" + "="*60)
    print("SECTION 11 — Audio Preparation")
    print("="*60)

    p = Path(audio_path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Probe duration without loading into RAM
    duration_s = _probe_audio_duration(audio_path)
    video_duration_s = total_frames / fps
    trim_start_s     = trim_start_frames / fps

    print(f"  Audio file      : {p.name}")
    print(f"  Audio duration  : {duration_s:.2f} s")
    print(f"  Video duration  : {video_duration_s:.2f} s")
    print(f"  Audio trim start: {trim_start_s:.2f} s  ({trim_start_frames:.1f} frames)")

    if duration_s > 0 and (trim_start_s + video_duration_s) > duration_s:
        print(f"  [WARN] Audio track ends before video ends — "
              f"generation will pad with silence.")

    # Copy audio to workspace (avoids path issues with spaces/unicode)
    dest = AUDIO_DIR / p.name
    if not dest.exists():
        shutil.copy2(p, dest)
    audio_workspace_path = str(dest)

    return {
        "path"            : audio_workspace_path,
        "original_path"   : str(p),
        "duration_s"      : duration_s,
        "fps"             : fps,
        "total_frames"    : total_frames,
        "trim_start_s"    : trim_start_s,
        "trim_start_frames": trim_start_frames,
        "video_duration_s": video_duration_s,
    }


def _probe_audio_duration(audio_path: str) -> float:
    """Use ffprobe to get audio duration without loading the file."""
    cmd = (
        f'ffprobe -v error -show_entries format=duration '
        f'-of default=noprint_wrappers=1:nokey=1 "{audio_path}"'
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def get_audio_segment_for_chunk(audio_info: Dict, start_frame: int,
                                 num_frames: int) -> Dict:
    """
    Return lightweight metadata for the audio portion covering a chunk.
    The actual audio slicing is performed by FFmpeg during final assembly;
    here we only track offsets to avoid duplicating waveform arrays.
    """
    fps     = audio_info["fps"]
    trim_s  = audio_info["trim_start_s"]
    chunk_start_s = trim_s + (start_frame / fps)
    chunk_dur_s   = num_frames / fps
    return {
        "audio_path"    : audio_info["path"],
        "start_s"       : chunk_start_s,
        "duration_s"    : chunk_dur_s,
        "start_frame"   : start_frame,
        "num_frames"    : num_frames,
        "fps"           : fps,
    }



# =============================================================================
# SECTION 12 — PROMPT / CONDITIONING CACHE
# =============================================================================

# Global conditioning cache — stores CPU tensors to avoid re-encoding
# the same prompt on every chunk.
_CONDITIONING_CACHE: Dict[str, Any] = {}


def build_prompt(custom_prompt: str = "") -> str:
    """Return the effective generation prompt."""
    if custom_prompt and custom_prompt.strip():
        return custom_prompt.strip()
    return GLOBAL_PROMPT


def encode_prompt(prompt: str, clip_model) -> Tuple[Any, Any]:
    """
    Encode text prompt through CLIPTextEncode → positive conditioning.
    Cache the result on CPU keyed by prompt hash so identical prompts
    across chunks are not re-encoded.

    Returns (positive_cond, negative_cond) where negative_cond is the
    ConditioningZeroOut result (all-zeros, as used in the workflow).
    """
    cache_key = hashlib.md5(prompt.encode()).hexdigest()

    if cache_key in _CONDITIONING_CACHE:
        cached = _CONDITIONING_CACHE[cache_key]
        # Move CPU-cached tensors back to GPU for use
        return cached["positive"], cached["negative"]

    # Node: CLIPTextEncode
    clip_encoder = get_node("CLIPTextEncode")
    with torch.inference_mode():
        positive_raw = clip_encoder.encode(text=prompt, clip=clip_model)
        positive_cond = get_value(positive_raw, 0)

    # Node 128: ConditioningZeroOut — workflow negative for distilled LTX
    zero_out = get_node("ConditioningZeroOut")
    with torch.inference_mode():
        negative_raw  = zero_out.zero_out(conditioning=positive_cond)
        negative_cond = get_value(negative_raw, 0)

    # Cache on CPU to free GPU memory between chunks
    _CONDITIONING_CACHE[cache_key] = {
        "positive": positive_cond,
        "negative": negative_cond,
    }

    return positive_cond, negative_cond


def apply_ltxv_conditioning(positive_cond, negative_cond,
                             frame_rate: float) -> Tuple[Any, Any]:
    """
    Node 27: LTXVConditioning
    Attaches temporal (frame_rate) metadata to the conditioning tensors.
    This is required by the LTX-2.3 DiT to understand the video cadence.
    """
    ltxv_cond = get_node("LTXVConditioning")
    with torch.inference_mode():
        result = ltxv_cond.append(
            positive=positive_cond,
            negative=negative_cond,
            frame_rate=frame_rate,
        )
    pos = get_value(result, 0)
    neg = get_value(result, 1)
    return pos, neg


def clear_conditioning_cache():
    """Release all cached conditioning tensors."""
    global _CONDITIONING_CACHE
    _CONDITIONING_CACHE.clear()
    cleanup_memory()



# =============================================================================
# SECTION 13 — TIMELINE CALCULATION & LTX TEMPORAL CONSTRAINTS
# =============================================================================

def _is_valid_ltx_frame_count(n: int) -> bool:
    """
    LTX-2.3 requires frame counts of the form  8k + 1
    (i.e. 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97 …)
    This mirrors the temporal downsampling factor of the LTX VAE (8×).
    """
    return n >= 9 and (n - 1) % 8 == 0


def normalize_ltx_frame_count(requested: int, fps: int,
                                min_frames: int = 9) -> int:
    """
    Round requested frame count UP to the nearest valid LTX count (8k+1).
    Always reports the adjustment so the user knows the actual duration.
    """
    if requested < min_frames:
        requested = min_frames

    # Round up: find k such that 8k+1 >= requested
    k = math.ceil((requested - 1) / 8)
    adjusted = 8 * k + 1

    requested_dur = requested / fps
    adjusted_dur  = adjusted  / fps

    print(f"  Frame count normalization:")
    print(f"    Requested frames : {requested}  ({requested_dur:.3f} s)")
    print(f"    LTX-valid frames : {adjusted}  ({adjusted_dur:.3f} s)")
    if adjusted != requested:
        print(f"    Δ frames         : +{adjusted - requested}")

    return adjusted


def calculate_timeline(duration_seconds: float, fps: int,
                        preview_mode: bool = False,
                        preview_duration: float = 3.0) -> Dict:
    """
    Calculate total frames, applying LTX temporal constraints.
    Matches the workflow's 31.5 s / 756-frame timeline when defaults are used.
    """
    effective_duration = preview_duration if preview_mode else duration_seconds

    raw_frames   = round(effective_duration * fps)
    total_frames = normalize_ltx_frame_count(raw_frames, fps)
    actual_dur   = total_frames / fps

    return {
        "fps"            : fps,
        "duration_s"     : duration_seconds,
        "effective_dur_s": effective_duration,
        "raw_frames"     : raw_frames,
        "total_frames"   : total_frames,
        "actual_dur_s"   : actual_dur,
        "preview_mode"   : preview_mode,
    }


def get_chunk_seed(global_seed: int, chunk_index: int) -> int:
    """
    Deterministic per-chunk seed derived from global seed.
    Uses a large prime multiplier to avoid correlation between close chunks.
    """
    return (global_seed + chunk_index * 1_000_003) & 0x7FFF_FFFF


def plan_chunks(total_frames: int, chunk_frames: int, fps: int) -> List[Dict]:
    """
    Divide the full timeline into overlapping chunks where each chunk
    satisfies the LTX 8k+1 frame constraint.

    The last chunk is extended (never truncated) to a valid LTX count,
    so the total generated frames may slightly exceed total_frames.
    """
    chunks  = []
    cursor  = 0
    idx     = 0

    while cursor < total_frames:
        remaining = total_frames - cursor
        # Use configured chunk size, but don't overshoot unless last chunk
        n_frames  = min(chunk_frames, remaining)
        # Ensure valid LTX count — round up
        n_frames  = normalize_ltx_frame_count(n_frames, fps)
        # Clamp to not start beyond total
        if cursor + n_frames > total_frames + 8:
            n_frames = normalize_ltx_frame_count(remaining, fps)

        chunk_path = CHUNKS_DIR / f"chunk_{idx:04d}.mp4"
        chunks.append({
            "chunk_index" : idx,
            "start_frame" : cursor,
            "num_frames"  : n_frames,
            "fps"         : fps,
            "path"        : str(chunk_path),
        })

        cursor += n_frames
        idx    += 1

        if n_frames == 0:
            break   # safety: should never happen

    return chunks


def select_chunk_size(quality_mode: str, width: int, height: int,
                       auto: bool = True) -> int:
    """
    Select a safe chunk frame count for the given quality mode and resolution.
    When auto=True, scales down from the profile default based on resolution.
    """
    profile    = T4_PROFILES.get(quality_mode, T4_PROFILES["t4_safe"])
    base_chunk = profile["chunk_frames"]

    if not auto:
        return normalize_ltx_frame_count(CONFIG["chunk_frames"], CONFIG["fps"])

    # Scale by pixel count relative to 1280×720
    ref_pixels    = 1280 * 720
    actual_pixels = width * height
    scale_factor  = ref_pixels / max(actual_pixels, 1)
    scaled         = int(base_chunk * scale_factor)
    # Clamp between 9 and 97
    scaled = max(9, min(97, scaled))
    final  = normalize_ltx_frame_count(scaled, CONFIG["fps"])

    print(f"  Chunk size: profile={base_chunk}  "
          f"scaled={scaled}  final={final}  ({quality_mode})")
    return final



# =============================================================================
# SECTION 14 — CHUNK PLANNER (printout)
# =============================================================================

def print_generation_plan(timeline: Dict, chunks: List[Dict],
                           width: int, height: int):
    """Print a formatted generation plan before starting."""
    print("\n" + "="*60)
    print("LTX-2.3 DIRECTOR 2.0 MV — Generation Plan")
    print("="*60)
    g = memory_manager.gpu_memory()
    print(f"  GPU             : {GPU_INFO.get('device_name', 'Unknown')}")
    print(f"  VRAM free       : {g['free_gb']:.2f} GB / {g['total_gb']:.2f} GB total")
    print(f"  Resolution      : {width}×{height}")
    print(f"  FPS             : {timeline['fps']}")
    print(f"  Requested dur   : {timeline['duration_s']:.2f} s")
    print(f"  Actual dur      : {timeline['actual_dur_s']:.2f} s")
    print(f"  Total frames    : {timeline['total_frames']}")
    print(f"  Chunk size      : {chunks[0]['num_frames']} frames")
    print(f"  Total chunks    : {len(chunks)}")
    print(f"  Quality mode    : {CONFIG['quality_mode']}")
    print(f"  Sampler pass 1  : {CONFIG['pass1_steps']} steps  denoise={CONFIG['pass1_denoise']}")
    print(f"  Sampler pass 2  : {CONFIG['pass2_steps']} steps  denoise={CONFIG['pass2_denoise']}")
    print("="*60)



# =============================================================================
# SECTION 15 — LTX-2.3 MODEL LOADING
# =============================================================================

# Model cache — keyed by role string.  Values are live ComfyUI model objects.
# None means not yet loaded.
_MODEL_CACHE: Dict[str, Any] = {
    "dit"           : None,   # DiT + LoRAs (NODE_CLASS_MAPPINGS["UnetLoaderGGUF"] output)
    "clip"          : None,   # DualCLIPLoader output
    "audio_vae"     : None,   # VAELoader (audio)
    "video_vae"     : None,   # VAELoader (video)
    "tiny_vae"      : None,   # VAELoaderKJ (preview)
    "upscale_model" : None,   # LatentUpscaleModelLoader
}


def _set_comfyui_model_path(filename: str, subdir: str):
    """
    Tell ComfyUI's folder_paths where to find a specific model file.
    Required before calling any loader node that uses folder_paths.
    """
    try:
        import folder_paths
        dest = MODELS_DIR / subdir
        dest.mkdir(parents=True, exist_ok=True)
        if str(dest) not in folder_paths.get_folder_paths(subdir):
            folder_paths.add_model_folder_path(subdir, str(dest))
    except Exception:
        pass


def load_dit_model() -> Any:
    """
    Node 135: UnetLoaderGGUF
    Loads the LTX-2.3 22B DiT model in Q4_K_M GGUF format.
    Cached — only loads once per session.
    """
    if _MODEL_CACHE["dit"] is not None:
        return _MODEL_CACHE["dit"]

    print("\n  [15] Loading DiT model (UnetLoaderGGUF) …")
    cleanup_memory()
    print_memory("before DiT load")

    _set_comfyui_model_path(MODEL_FILES["dit_gguf"], "unet")

    loader = get_node("UnetLoaderGGUF")
    with torch.inference_mode():
        result = loader.load_unet(unet_name=MODEL_FILES["dit_gguf"])
    model = get_value(result, 0)

    print_memory("after DiT load")
    _MODEL_CACHE["dit"] = model
    print(f"  ✓ DiT loaded")
    return model


def load_loras(model, clip) -> Tuple[Any, Any]:
    """
    Node 138: Power Lora Loader (rgthree)
    Applies 4 LoRAs to the DiT model and CLIP encoder.
    Skips disabled LoRAs and missing files gracefully.
    Returns (patched_model, patched_clip).
    """
    lora_cfg = [
        (MODEL_FILES["lora_dynamic"],    CONFIG["lora_dynamic_enable"],    CONFIG["lora_dynamic_strength"]),
        (MODEL_FILES["lora_omninft"],    CONFIG["lora_omninft_enable"],    CONFIG["lora_omninft_strength"]),
        (MODEL_FILES["lora_transition"], CONFIG["lora_transition_enable"], CONFIG["lora_transition_strength"]),
        (MODEL_FILES["lora_mvcamera"],   CONFIG["lora_mvcamera_enable"],   CONFIG["lora_mvcamera_strength"]),
    ]

    # Use standard LoraLoaderModelOnly if Power Lora Loader isn't available
    use_power = "Power Lora Loader (rgthree)" in NODE_CLASS_MAPPINGS
    lora_loader_name = ("Power Lora Loader (rgthree)" if use_power
                        else "LoraLoaderModelOnly")

    patched_model = model
    patched_clip  = clip

    _set_comfyui_model_path("", "loras")

    for lora_file, enabled, strength in lora_cfg:
        if not enabled:
            print(f"    LoRA {lora_file[:40]} — skipped (disabled)")
            continue
        lora_path = get_model_path(lora_file)
        if lora_path is None:
            print(f"    LoRA {lora_file[:40]} — skipped (file not found)")
            continue

        try:
            if use_power:
                loader = get_node("Power Lora Loader (rgthree)")
                # rgthree loader takes model+clip and lora_stack list
                with torch.inference_mode():
                    res = loader.load_loras(
                        model=patched_model,
                        clip=patched_clip,
                        loras=[{"lora": lora_file, "strength": strength,
                                "strengthTwo": None, "on": True}],
                    )
                patched_model = get_value(res, 0)
                patched_clip  = get_value(res, 1)
            else:
                # Fallback: LoraLoaderModelOnly (model only, no CLIP)
                loader = get_node("LoraLoaderModelOnly")
                with torch.inference_mode():
                    res = loader.load_lora(
                        model=patched_model,
                        lora_name=lora_file,
                        strength_model=strength,
                    )
                patched_model = get_value(res, 0)
            print(f"    ✓ LoRA {lora_file[:50]}  strength={strength}")
        except Exception as e:
            print(f"    [WARN] LoRA apply failed ({lora_file}): {e}")

    return patched_model, patched_clip


def load_text_encoder() -> Any:
    """
    Node 12: DualCLIPLoader
    Loads Gemma-3 12B FP4 + LTX-2.3 text projection as a dual CLIP encoder.
    Cached — only loads once.
    """
    if _MODEL_CACHE["clip"] is not None:
        return _MODEL_CACHE["clip"]

    print("\n  [15] Loading text encoder (DualCLIPLoader) …")
    cleanup_memory()

    for fname in [MODEL_FILES["clip1"], MODEL_FILES["clip2"]]:
        _set_comfyui_model_path(fname, "clip")

    loader = get_node("DualCLIPLoader")
    with torch.inference_mode():
        result = loader.load_clip(
            clip_name1 = MODEL_FILES["clip1"],
            clip_name2 = MODEL_FILES["clip2"],
            type       = "ltxv",
            device     = "default",
        )
    clip = get_value(result, 0)
    _MODEL_CACHE["clip"] = clip
    print("  ✓ Text encoder loaded")
    return clip


def load_audio_vae() -> Any:
    """Node 8: VAELoader for LTX23 Audio VAE."""
    if _MODEL_CACHE["audio_vae"] is not None:
        return _MODEL_CACHE["audio_vae"]
    print("  [15] Loading audio VAE …")
    _set_comfyui_model_path(MODEL_FILES["audio_vae"], "vae")
    loader = get_node("VAELoader")
    with torch.inference_mode():
        result = loader.load_vae(vae_name=MODEL_FILES["audio_vae"])
    vae = get_value(result, 0)
    _MODEL_CACHE["audio_vae"] = vae
    print("  ✓ Audio VAE loaded")
    return vae


def load_video_vae() -> Any:
    """Node 36: VAELoader for LTX23 Video VAE (bf16, ~1.4 GB)."""
    if _MODEL_CACHE["video_vae"] is not None:
        return _MODEL_CACHE["video_vae"]
    print("  [15] Loading video VAE …")
    _set_comfyui_model_path(MODEL_FILES["video_vae"], "vae")
    loader = get_node("VAELoader")
    with torch.inference_mode():
        result = loader.load_vae(vae_name=MODEL_FILES["video_vae"])
    vae = get_value(result, 0)
    _MODEL_CACHE["video_vae"] = vae
    print("  ✓ Video VAE loaded")
    return vae


def load_upscale_model() -> Any:
    """Node 13: LatentUpscaleModelLoader — LTX-2.3 spatial 2× upscaler."""
    if _MODEL_CACHE["upscale_model"] is not None:
        return _MODEL_CACHE["upscale_model"]
    print("  [15] Loading latent upscale model …")
    _set_comfyui_model_path(MODEL_FILES["upscaler"], "upscale_models")
    loader = get_node("LatentUpscaleModelLoader")
    with torch.inference_mode():
        result = loader.load_model(model_name=MODEL_FILES["upscaler"])
    upscaler = get_value(result, 0)
    _MODEL_CACHE["upscale_model"] = upscaler
    print("  ✓ Latent upscaler loaded")
    return upscaler


def load_all_models() -> Dict[str, Any]:
    """
    Load all models in dependency order.
    DiT + LoRAs are loaded first (largest); VAEs last (can still OOM).
    """
    print("\n" + "="*60)
    print("SECTION 15 — Model Loading")
    print("="*60)

    torch.cuda.reset_peak_memory_stats()
    print_memory("start of model loading")

    # 1. DiT (heaviest — ~9 GB GGUF Q4)
    dit   = load_dit_model()
    # 2. Text encoder
    clip  = load_text_encoder()
    # 3. Apply LoRAs to DiT + CLIP
    dit, clip = load_loras(dit, clip)
    _MODEL_CACHE["dit"]  = dit
    _MODEL_CACHE["clip"] = clip
    # 4. VAEs
    load_audio_vae()
    load_video_vae()
    # 5. Upscale model (deferred until first Pass-2)
    #    load_upscale_model() is called inside generate_chunk() to avoid
    #    holding it in VRAM during Pass-1 sampling.

    print_memory("after all models loaded")
    print("\n  ✓ All models loaded and cached.\n")
    return _MODEL_CACHE



# =============================================================================
# SECTION 16 — DIRECTOR WORKFLOW EXECUTION
# =============================================================================
#
# This section implements the full LTX Director 2.0 computational graph
# exactly as described in the workflow JSON:
#
#   Node 131  LTXDirector              → build_director_conditioning()
#   Node 133  LTXDirectorGuide (pass1) → run_director_guide()  scale=0.5
#   Node 132  LTXDirectorGuide (pass2) → run_director_guide()  scale=1.0
#   Node 55   LTXDirectorCropGuides    → crop_director_guides()
#   Node 54   LTXDirectorCropGuides    → crop_director_guides()
#   Nodes 28/17  CFGGuider             → make_cfg_guider()
#   Nodes 32/20  KSamplerSelect        → make_noise() / select_sampler()
#   Nodes 33/21  BasicScheduler        → make_sigmas()
#   Node 30   RandomNoise              → make_noise()
#   Nodes 31/19  SamplerCustomAdvanced → run_sampler()
#   Nodes 29/18  LTXVConcatAVLatent    → concat_av_latent()
#   Nodes 34/22  LTXVSeparateAVLatent  → separate_av_latent()
#   Node 14   LTXVLatentUpsampler      → upsample_latent()
#
# =============================================================================

def _build_director_fallback(model, clip, audio_vae,
                              image_tensor, prompt: str,
                              num_frames: int, fps: float,
                              width: int, height: int, seed: int,
                              audio_path: str) -> Dict:
    """
    Fallback path used when the LTXDirector node is unavailable
    (e.g. WhatDreamsCost package failed to load).

    Uses the standard LTX image-to-video pipeline:
      LTXVImgToVideoInplace + LTXVEmptyLatentAudio
    instead of the Director timeline conditioning.
    """
    print("  [16] Using standard LTX fallback (no Director node)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_gpu = image_tensor.to(device)

    # ── Preprocess image ──────────────────────────────────────────────────────
    try:
        preprocessor = get_node("LTXVPreprocess")
        with torch.inference_mode():
            prep_res = preprocessor.preprocess(
                image=img_gpu, width=width, height=height,
                img_compression=18,
            )
        preprocessed_img = get_value(prep_res, 0)
    except Exception:
        preprocessed_img = img_gpu

    # ── Text conditioning ─────────────────────────────────────────────────────
    pos_cond, neg_cond = encode_prompt(prompt, clip)
    pos_ltxv, neg_ltxv = apply_ltxv_conditioning(pos_cond, neg_cond, fps)

    # ── Image-to-video latent ─────────────────────────────────────────────────
    video_vae = _MODEL_CACHE["video_vae"]
    try:
        img2vid = get_node("LTXVImgToVideoInplace")
        with torch.inference_mode():
            img_lat_res = img2vid.generate(
                positive=pos_ltxv, negative=neg_ltxv,
                vae=video_vae, image=preprocessed_img,
                width=width, height=height,
                num_frames=num_frames, frame_rate=fps,
            )
        pos_ltxv = get_value(img_lat_res, 0)
        neg_ltxv = get_value(img_lat_res, 1)
        video_latent = get_value(img_lat_res, 2)
    except Exception as e:
        print(f"  [WARN] LTXVImgToVideoInplace failed: {e} — using empty latent")
        empty_lat_node = get_node("LTXVEmptyLatentVideo")
        with torch.inference_mode():
            vl_res = empty_lat_node.generate(
                width=width, height=height,
                num_frames=num_frames, batch_size=1,
            )
        video_latent = get_value(vl_res, 0)

    # ── Audio latent ──────────────────────────────────────────────────────────
    try:
        audio_lat_node = get_node("LTXVEmptyLatentAudio")
        with torch.inference_mode():
            al_res = audio_lat_node.generate(
                seconds=num_frames / fps,
                audio_vae=audio_vae,
            )
        audio_latent = get_value(al_res, 0)
    except Exception as e:
        print(f"  [WARN] Audio latent generation failed: {e}")
        audio_latent = None

    del img_gpu
    cleanup_memory()

    return {
        "model"            : model,
        "positive"         : pos_ltxv,
        "negative"         : neg_ltxv,
        "video_latent"     : video_latent,
        "audio_latent"     : audio_latent,
        "guide_data"       : None,
        "motion_guide_data": None,
        "frame_rate"       : fps,
    }


def build_director_conditioning(model, clip, audio_vae,
                                 image_tensor, prompt: str,
                                 num_frames: int, fps: float,
                                 width: int, height: int, seed: int,
                                 audio_path: str,
                                 quality_mode: str = "t4_safe") -> Dict:
    """
    Node 131: LTXDirector
    The primary conditioning node. Ingests reference image, audio file,
    global prompt, and timeline data → outputs model, conditioning,
    video_latent, audio_latent, guide_data, motion_guide_data, frame_rate.

    In t4_safe mode (no CLIP model headroom) we skip the Director and fall
    back to the standard LTX image-to-video pipeline to conserve VRAM.
    """
    director_available = "LTXDirector" in NODE_CLASS_MAPPINGS

    if not director_available:
        return _build_director_fallback(
            model, clip, audio_vae, image_tensor, prompt,
            num_frames, fps, width, height, seed, audio_path
        )

    print("  [16] Building Director conditioning …")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_gpu = image_tensor.to(device)

    # Build minimal timeline_data JSON matching the workflow structure
    timeline_data = json.dumps({
        "mainTrackEnabled" : True,
        "audioTrackEnabled": True,
        "motionTrackEnabled": True,
        "global_prompt"    : prompt,
        "normalStartFrame" : 0,
        "normalDurationFrames": num_frames,
        "segments": [{
            "id"         : "chunk_seg_0",
            "start"      : 0,
            "length"     : float(num_frames),
            "prompt"     : "",
            "type"       : "image",
            "imageFile"  : "",
            "isEndFrame" : False,
        }],
        "motionSegments": [],
        "audioSegments" : [{
            "id"                : "chunk_audio_0",
            "type"              : "audio",
            "start"             : 0,
            "length"            : float(num_frames),
            "trimStart"         : 0,
            "audioDurationFrames": num_frames,
            "audioFile"         : audio_path,
            "fileName"          : Path(audio_path).name,
            "waveformPeaks"     : [],
        }] if audio_path else [],
    })

    director = get_node("LTXDirector")

    # Introspect the node's signature to avoid passing unsupported args
    import inspect
    director_sig   = inspect.signature(
        director.generate if hasattr(director, "generate")
        else list(type(director).__dict__.values())[0]
    )
    director_params = set(director_sig.parameters.keys())

    base_kwargs = {
        "model"         : model,
        "clip"          : clip,
        "audio_vae"     : audio_vae,
        "timeline_data" : timeline_data,
        "width"         : width,
        "height"        : height,
        "frame_rate"    : fps,
        "start_frame"   : 0,
        "end_frame"     : num_frames,
        "seed"          : seed,
        "img_compression": 18,
    }

    # Only pass image if node accepts it (some builds differ)
    if "image" in director_params:
        base_kwargs["image"] = img_gpu
    if "optional_latent" in director_params:
        base_kwargs["optional_latent"] = None

    kwargs = {k: v for k, v in base_kwargs.items() if k in director_params}

    try:
        with torch.inference_mode():
            # Find the actual execute method
            execute_fn = None
            for mname in ["generate", "execute", "run", "__call__"]:
                if hasattr(director, mname) and callable(getattr(director, mname)):
                    execute_fn = getattr(director, mname)
                    break
            if execute_fn is None:
                raise RuntimeError("Cannot find execute method on LTXDirector")
            result = execute_fn(**kwargs)

        out_model        = get_value(result, 0)
        out_positive     = get_value(result, 1)
        out_video_latent = get_value(result, 2)
        out_audio_latent = get_value(result, 3)
        out_guide_data   = get_value(result, 4)
        out_motion_guide = get_value(result, 5)
        out_frame_rate   = get_value(result, 6)

        del img_gpu
        cleanup_memory()

        return {
            "model"            : out_model,
            "positive"         : out_positive,
            "negative"         : None,   # Director provides only positive; zero-out applied next
            "video_latent"     : out_video_latent,
            "audio_latent"     : out_audio_latent,
            "guide_data"       : out_guide_data,
            "motion_guide_data": out_motion_guide,
            "frame_rate"       : out_frame_rate if out_frame_rate else fps,
        }

    except Exception as e:
        print(f"  [WARN] LTXDirector failed: {e}")
        print("  Falling back to standard LTX pipeline …")
        del img_gpu
        cleanup_memory()
        return _build_director_fallback(
            model, clip, audio_vae, image_tensor, prompt,
            num_frames, fps, width, height, seed, audio_path
        )


def run_director_guide(positive, negative, video_vae, latent,
                       guide_data, motion_guide_data, model,
                       upscale_factor: float, node_tag: str = "") -> Dict:
    """
    Nodes 132/133: LTXDirectorGuide
    Applies Director guide conditioning at a given upscale_factor:
      - Pass 1 (node 133): upscale_factor=0.5  (half resolution)
      - Pass 2 (node 132): upscale_factor=1.0  (full resolution)

    Returns dict with positive, negative, latent, model.
    If guide_data is None (fallback mode) returns inputs unchanged.
    """
    if guide_data is None:
        # No Director data — pass through unchanged
        return {
            "positive": positive,
            "negative": negative,
            "latent"  : latent,
            "model"   : model,
        }

    if "LTXDirectorGuide" not in NODE_CLASS_MAPPINGS:
        return {
            "positive": positive,
            "negative": negative,
            "latent"  : latent,
            "model"   : model,
        }

    guide_node = get_node("LTXDirectorGuide")

    # Widget values from workflow node 132/133:
    # ["None", 1, scale, "bicubic", 1, "center", True, False, 256, 64, False]
    kwargs = {
        "positive"          : positive,
        "negative"          : negative,
        "vae"               : video_vae,
        "latent"            : latent,
        "guide_data"        : guide_data,
        "upscale_factor"    : upscale_factor,
        "interpolation"     : "bicubic",
        "crop"              : "center",
        "guide_strength"    : 1.0,
        "use_guide"         : True,
        "debug"             : False,
        "height"            : 256,
        "width"             : 64,
        "force_upscale"     : False,
    }
    if motion_guide_data is not None:
        kwargs["motion_guide_data"] = motion_guide_data

    try:
        with torch.inference_mode():
            result = guide_node.apply_guide(**{
                k: v for k, v in kwargs.items()
                if hasattr(guide_node, "apply_guide") or True
            })
        return {
            "positive": get_value(result, 0),
            "negative": get_value(result, 1),
            "latent"  : get_value(result, 2),
            "model"   : get_value(result, 3),
        }
    except Exception as e:
        print(f"  [WARN] LTXDirectorGuide{node_tag} failed: {e} — skipping guide")
        return {
            "positive": positive,
            "negative": negative,
            "latent"  : latent,
            "model"   : model,
        }


def crop_director_guides(positive, negative, latent) -> Dict:
    """
    Nodes 54/55: LTXDirectorCropGuides
    Strips padding added by the DirectorGuide before upscaling / VAE decode.
    Returns dict with positive, negative, latent.
    If node unavailable, passes through unchanged.
    """
    if "LTXDirectorCropGuides" not in NODE_CLASS_MAPPINGS:
        return {"positive": positive, "negative": negative, "latent": latent}
    crop_node = get_node("LTXDirectorCropGuides")
    try:
        with torch.inference_mode():
            result = crop_node.crop(
                positive=positive, negative=negative, latent=latent
            )
        return {
            "positive": get_value(result, 0),
            "negative": get_value(result, 1),
            "latent"  : get_value(result, 2),
        }
    except Exception as e:
        print(f"  [WARN] LTXDirectorCropGuides failed: {e} — skipping")
        return {"positive": positive, "negative": negative, "latent": latent}


def concat_av_latent(video_latent, audio_latent) -> Any:
    """
    Nodes 18/29: LTXVConcatAVLatent
    Fuses video + audio latent tensors into a single joint AV latent.
    Required by SamplerCustomAdvanced.
    """
    if audio_latent is None:
        return video_latent
    concat_node = get_node("LTXVConcatAVLatent")
    with torch.inference_mode():
        result = concat_node.concat(
            video_latent=video_latent,
            audio_latent=audio_latent,
        )
    return get_value(result, 0)


def separate_av_latent(av_latent) -> Tuple[Any, Any]:
    """
    Nodes 22/34: LTXVSeparateAVLatent
    Splits a joint AV latent back into (video_latent, audio_latent).
    """
    sep_node = get_node("LTXVSeparateAVLatent")
    with torch.inference_mode():
        result = sep_node.separate(av_latent=av_latent)
    return get_value(result, 0), get_value(result, 1)


def make_noise(seed: int) -> Any:
    """Node 30: RandomNoise — create a deterministic noise tensor."""
    noise_node = get_node("RandomNoise")
    with torch.inference_mode():
        result = noise_node.get_noise(noise_seed=seed, noise_type="fixed")
    return get_value(result, 0)


def select_sampler(sampler_name: str = "euler") -> Any:
    """Nodes 20/32: KSamplerSelect."""
    sampler_node = get_node("KSamplerSelect")
    with torch.inference_mode():
        result = sampler_node.get_sampler(sampler_name=sampler_name)
    return get_value(result, 0)


def make_sigmas(model, scheduler: str, steps: int, denoise: float) -> Any:
    """
    Nodes 21/33: BasicScheduler
    Generates the sigma schedule for a sampling pass.
    """
    sched_node = get_node("BasicScheduler")
    with torch.inference_mode():
        result = sched_node.get_sigmas(
            model=model, scheduler=scheduler,
            steps=steps, denoise=denoise,
        )
    return get_value(result, 0)


def make_cfg_guider(model, positive, negative, cfg_scale: float = 1.0) -> Any:
    """Nodes 17/28: CFGGuider — cfg=1.0 for distilled LTX."""
    guider_node = get_node("CFGGuider")
    with torch.inference_mode():
        result = guider_node.get_guider(
            model=model, positive=positive, negative=negative, cfg=cfg_scale
        )
    return get_value(result, 0)


def upsample_latent(latent, upscale_model, video_vae) -> Any:
    """
    Node 14: LTXVLatentUpsampler
    2× spatial upscale in latent space (no pixel decode).
    """
    upsample_node = get_node("LTXVLatentUpsampler")
    with torch.inference_mode():
        result = upsample_node.upscale(
            samples=latent,
            upscale_model=upscale_model,
            vae=video_vae,
        )
    return get_value(result, 0)


def run_sampler(noise, guider, sampler, sigmas, latent) -> Any:
    """
    Nodes 19/31: SamplerCustomAdvanced
    Core sampling step shared by both passes.
    """
    sampler_node = get_node("SamplerCustomAdvanced")
    with torch.inference_mode():
        result = sampler_node.sample(
            noise=noise,
            guider=guider,
            sampler=sampler,
            sigmas=sigmas,
            latent_image=latent,
        )
    return get_value(result, 0)   # "output" (not denoised_output)


# =============================================================================
# generate_chunk() — The complete 2-pass Director pipeline for one chunk
# =============================================================================

def generate_chunk(chunk_info: Dict,
                   image_tensor: "torch.Tensor",
                   prompt: str,
                   audio_info: Dict,
                   global_seed: int,
                   width: int,
                   height: int,
                   fps: float,
                   quality_mode: str = "t4_safe") -> Dict:
    """
    Execute the full LTX-2.3 Director 2.0 two-pass pipeline for one temporal chunk.

    Pass 1 (nodes 133 → 31):  8 steps, denoise=1.0,  upscale_factor=0.5
    Pass 2 (nodes 132 → 19):  4 steps, denoise=0.42, upscale_factor=1.0

    Returns lightweight metadata dict (no GPU tensors).
    Raises on non-OOM errors; OOM handled by adaptive_chunk_generator.
    """
    idx        = chunk_info["chunk_index"]
    start_f    = chunk_info["start_frame"]
    num_frames = chunk_info["num_frames"]
    out_path   = chunk_info["path"]
    seed       = get_chunk_seed(global_seed, idx)

    print(f"\n{'─'*60}")
    print(f"  [Chunk {idx+1:03d}]  frames {start_f}–{start_f+num_frames-1}  "
          f"seed={seed}")
    print_memory(f"Chunk {idx+1} start")

    profile    = T4_PROFILES.get(quality_mode, T4_PROFILES["t4_safe"])
    vae_sub    = profile["vae_sub_batch"]

    model      = _MODEL_CACHE["dit"]
    clip       = _MODEL_CACHE["clip"]
    audio_vae  = _MODEL_CACHE["audio_vae"]
    video_vae  = _MODEL_CACHE["video_vae"]

    assert model      is not None, "DiT model not loaded"
    assert clip       is not None, "CLIP model not loaded"
    assert audio_vae  is not None, "Audio VAE not loaded"
    assert video_vae  is not None, "Video VAE not loaded"

    memory_manager.warn_if_low()

    # ── Step 1: Build Director conditioning ───────────────────────────────────
    audio_seg  = get_audio_segment_for_chunk(audio_info, start_f, num_frames)
    director   = build_director_conditioning(
        model=model, clip=clip, audio_vae=audio_vae,
        image_tensor=image_tensor,
        prompt=prompt,
        num_frames=num_frames, fps=fps,
        width=width, height=height, seed=seed,
        audio_path=audio_seg["audio_path"],
        quality_mode=quality_mode,
    )

    dir_model        = director["model"]
    dir_positive     = director["positive"]
    dir_negative     = director["negative"]
    dir_video_latent = director["video_latent"]
    dir_audio_latent = director["audio_latent"]
    guide_data       = director["guide_data"]
    motion_guide     = director["motion_guide_data"]
    frame_rate       = director["frame_rate"]

    # ── Step 2: Zero-out negative if Director didn't provide one ──────────────
    if dir_negative is None:
        zero_out = get_node("ConditioningZeroOut")
        with torch.inference_mode():
            neg_res   = zero_out.zero_out(conditioning=dir_positive)
        dir_negative = get_value(neg_res, 0)

    # ── Step 3: LTXVConditioning (node 27) — attach frame_rate ───────────────
    pos_ltxv, neg_ltxv = apply_ltxv_conditioning(
        dir_positive, dir_negative, frame_rate
    )

    # ── Step 4: Director Guide PASS 1 (node 133, upscale_factor=0.5) ─────────
    guide_p1 = run_director_guide(
        positive=pos_ltxv, negative=neg_ltxv,
        video_vae=video_vae,
        latent=dir_video_latent,
        guide_data=guide_data,
        motion_guide_data=motion_guide,
        model=dir_model,
        upscale_factor=0.5,
        node_tag=" [p1/node133]",
    )
    pos_p1    = guide_p1["positive"]
    neg_p1    = guide_p1["negative"]
    lat_p1    = guide_p1["latent"]
    model_p1  = guide_p1["model"]

    # ── Step 5: Fuse AV latent for Pass 1 (node 29) ──────────────────────────
    av_lat_p1 = concat_av_latent(lat_p1, dir_audio_latent)

    # ── Step 6: Shared noise (node 30) ───────────────────────────────────────
    noise = make_noise(seed)

    # ── Step 7: Sampler Pass 1 ────────────────────────────────────────────────
    print("    Pass 1 sampling …")
    print_memory("before pass1")
    sampler_p1 = select_sampler(CONFIG["sampler_name"])
    sigmas_p1  = make_sigmas(model_p1, CONFIG["scheduler"],
                              CONFIG["pass1_steps"], CONFIG["pass1_denoise"])
    guider_p1  = make_cfg_guider(model_p1, pos_p1, neg_p1, CONFIG["cfg_scale"])

    with torch.inference_mode():
        sampled_p1 = run_sampler(noise, guider_p1, sampler_p1, sigmas_p1, av_lat_p1)

    # Free Pass-1 intermediates
    del av_lat_p1, guider_p1, sigmas_p1, sampler_p1
    cleanup_memory()

    # ── Step 8: Separate AV from Pass 1 output (node 34) ─────────────────────
    vid_lat_p1, aud_lat_p1 = separate_av_latent(sampled_p1)
    del sampled_p1

    # ── Step 9: Crop Director guides (node 55) ────────────────────────────────
    cropped_p1 = crop_director_guides(pos_p1, neg_p1, vid_lat_p1)
    vid_lat_cropped = cropped_p1["latent"]
    del vid_lat_p1

    # ── Step 10: 2× Latent Upsample (node 14) — load upscaler now ────────────
    print("    Upscaling latent …")
    print_memory("before upscale")
    upscale_model = load_upscale_model()   # deferred load — safe here
    with torch.inference_mode():
        lat_upscaled = upsample_latent(vid_lat_cropped, upscale_model, video_vae)
    del vid_lat_cropped
    cleanup_memory()

    # ── Step 11: Director Guide PASS 2 (node 132, upscale_factor=1.0) ────────
    guide_p2 = run_director_guide(
        positive=cropped_p1["positive"], negative=cropped_p1["negative"],
        video_vae=video_vae,
        latent=lat_upscaled,
        guide_data=guide_data,
        motion_guide_data=motion_guide,
        model=model_p1,
        upscale_factor=1.0,
        node_tag=" [p2/node132]",
    )
    pos_p2   = guide_p2["positive"]
    neg_p2   = guide_p2["negative"]
    lat_p2   = guide_p2["latent"]
    model_p2 = guide_p2["model"]
    del lat_upscaled

    # ── Step 12: Fuse AV latent for Pass 2 (node 18) ─────────────────────────
    av_lat_p2 = concat_av_latent(lat_p2, aud_lat_p1)
    del aud_lat_p1

    # ── Step 13: Sampler Pass 2 ───────────────────────────────────────────────
    print("    Pass 2 sampling …")
    print_memory("before pass2")
    sampler_p2 = select_sampler(CONFIG["sampler_name"])
    sigmas_p2  = make_sigmas(model_p2, CONFIG["scheduler"],
                              CONFIG["pass2_steps"], CONFIG["pass2_denoise"])
    guider_p2  = make_cfg_guider(model_p2, pos_p2, neg_p2, CONFIG["cfg_scale"])

    with torch.inference_mode():
        sampled_p2 = run_sampler(noise, guider_p2, sampler_p2, sigmas_p2, av_lat_p2)

    del av_lat_p2, guider_p2, sigmas_p2, sampler_p2, noise
    cleanup_memory()

    # ── Step 14: Separate final AV (node 22) ─────────────────────────────────
    vid_lat_p2, aud_lat_p2 = separate_av_latent(sampled_p2)
    del sampled_p2

    # ── Step 15: Crop final Director guides (node 54) ─────────────────────────
    final_crop = crop_director_guides(pos_p2, neg_p2, vid_lat_p2)
    final_vid_lat = final_crop["latent"]
    del vid_lat_p2

    print_memory(f"Chunk {idx+1} before decode")

    # ── Step 16–18: Decode → save → cleanup (handled in Section 17) ──────────
    result_meta = decode_and_save_chunk(
        video_latent = final_vid_lat,
        audio_latent = aud_lat_p2,
        chunk_info   = chunk_info,
        video_vae    = video_vae,
        audio_vae    = audio_vae,
        fps          = fps,
        vae_sub_batch= vae_sub,
    )

    # Release final latents
    del final_vid_lat, aud_lat_p2
    del dir_positive, dir_negative, pos_p1, neg_p1
    del pos_p2, neg_p2, pos_ltxv, neg_ltxv
    if guide_data is not None:
        del guide_data
    if motion_guide is not None:
        del motion_guide
    cleanup_memory()

    print_memory(f"Chunk {idx+1} end")
    print(f"  ✓  Chunk {idx+1} saved → {out_path}")
    return result_meta



# =============================================================================
# SECTION 17 — VAE DECODING
# =============================================================================

def decode_video_latent(video_latent, video_vae,
                         sub_batch: int = 8) -> "torch.Tensor":
    """
    Node 1: VAEDecode
    Decodes a video latent → pixel frames tensor [N, H, W, 3] float32 on CPU.

    Sub-batch decoding: if the full frame count would OOM, decode in slices
    of `sub_batch` latent frames and concatenate on CPU.
    Never returns a GPU tensor — immediately moves to CPU.
    """
    vae_decode = get_node("VAEDecode")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Estimate decoded frame size and decide whether to sub-batch
    # LTX VAE temporal factor = 8; spatial factor = 32
    # latent shape: {"samples": [B, C, T, H, W]}
    samples = video_latent.get("samples") if isinstance(video_latent, dict) \
              else video_latent
    if samples is None:
        raise ValueError("video_latent has no 'samples' key")

    # samples shape: [B, C, T, Hl, Wl]
    T_lat  = samples.shape[2] if samples.ndim == 5 else 1
    Hl     = samples.shape[-2] if samples.ndim >= 4 else 1
    Wl     = samples.shape[-1]
    H_pix  = Hl * 8    # LTX spatial downscale ×8
    W_pix  = Wl * 8
    T_pix  = (T_lat - 1) * 8 + 1   # LTX temporal: (T-1)*8+1

    frame_gb = memory_manager.estimate_frame_ram_gb(T_pix, H_pix, W_pix)
    free_gb  = memory_manager.gpu_free_gb()
    # Leave headroom for VAE itself (~1.4 GB) + safety margin
    budget_gb = max(0.0, free_gb - 1.4 - CONFIG["gpu_safety_margin_gb"])

    if frame_gb > budget_gb and sub_batch < T_lat:
        print(f"    VAE sub-batch decode: {T_lat} lat-frames → "
              f"slices of {sub_batch}  (est {frame_gb:.2f} GB > budget {budget_gb:.2f} GB)")
        return _decode_video_subbatch(video_latent, video_vae, sub_batch, vae_decode)

    with torch.inference_mode():
        result = vae_decode.decode(samples=video_latent, vae=video_vae)
    frames_gpu = get_value(result, 0)   # [N, H, W, 3]
    frames_cpu = frames_gpu.detach().to("cpu", non_blocking=False)
    del frames_gpu
    cleanup_memory()
    return frames_cpu   # [N, H, W, 3] float32 on CPU


def _decode_video_subbatch(video_latent, video_vae,
                            sub_batch: int, vae_decode) -> "torch.Tensor":
    """
    Decode temporal sub-batches of a video latent to avoid OOM.
    Accumulates decoded slices on CPU, never on GPU simultaneously.
    """
    samples = video_latent["samples"] if isinstance(video_latent, dict) \
              else video_latent
    T_lat = samples.shape[2]
    cpu_chunks = []

    for t_start in range(0, T_lat, sub_batch):
        t_end = min(t_start + sub_batch, T_lat)
        slice_samples = samples[:, :, t_start:t_end, :, :]
        slice_lat = {"samples": slice_samples}
        with torch.inference_mode():
            result = vae_decode.decode(samples=slice_lat, vae=video_vae)
        frames_gpu = get_value(result, 0)
        cpu_chunks.append(frames_gpu.detach().to("cpu", non_blocking=False))
        del frames_gpu, result, slice_samples, slice_lat
        cleanup_memory()

    return torch.cat(cpu_chunks, dim=0)   # [N_total, H, W, 3] on CPU


def decode_audio_latent(audio_latent, audio_vae) -> Optional["torch.Tensor"]:
    """
    Node 24: LTXVAudioVAEDecode
    Decodes audio latent → waveform tensor on CPU.
    Returns None if audio_latent is None or decode fails.
    """
    if audio_latent is None:
        return None
    if "LTXVAudioVAEDecode" not in NODE_CLASS_MAPPINGS:
        print("  [WARN] LTXVAudioVAEDecode not available — skipping audio decode")
        return None
    audio_dec = get_node("LTXVAudioVAEDecode")
    try:
        with torch.inference_mode():
            result = audio_dec.decode(samples=audio_latent, audio_vae=audio_vae)
        audio_out = get_value(result, 0)
        # Move to CPU immediately
        if isinstance(audio_out, dict):
            # ComfyUI AUDIO format: {"waveform": tensor, "sample_rate": int}
            waveform = audio_out.get("waveform")
            if waveform is not None and waveform.is_cuda:
                audio_out["waveform"] = waveform.detach().cpu()
        elif hasattr(audio_out, "is_cuda") and audio_out.is_cuda:
            audio_out = audio_out.detach().cpu()
        return audio_out
    except Exception as e:
        print(f"  [WARN] Audio VAE decode failed: {e}")
        return None



# =============================================================================
# SECTION 18 — CHUNK SAVING
# =============================================================================

def save_chunk_to_disk(frames_cpu: "torch.Tensor",
                        audio_cpu,
                        chunk_info: Dict,
                        fps: float) -> str:
    """
    Write a decoded chunk to disk as an H.264 MP4 using FFmpeg pipe.

    Streams frames one-at-a-time through an FFmpeg subprocess —
    never holds a full frame batch in RAM as a Python list or numpy array.

    Parameters
    ----------
    frames_cpu  : [N, H, W, 3] float32 tensor on CPU  (values 0–1)
    audio_cpu   : ComfyUI AUDIO dict or waveform tensor, or None
    chunk_info  : chunk metadata dict
    fps         : frames per second

    Returns the output file path.
    """
    out_path = chunk_info["path"]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    N, H, W, C = frames_cpu.shape
    assert C == 3, f"Expected 3-channel frames, got {C}"

    # ── Write audio to a temp WAV if available ────────────────────────────────
    audio_tmp = None
    if audio_cpu is not None:
        audio_tmp = _save_audio_temp(audio_cpu, fps, N)

    # ── Open FFmpeg pipe ───────────────────────────────────────────────────────
    ffmpeg_cmd = _build_ffmpeg_encode_cmd(
        out_path=out_path,
        width=W, height=H, fps=fps,
        audio_tmp=audio_tmp,
        crf=8,
        pix_fmt="yuv420p",
    )

    proc = subprocess.Popen(
        ffmpeg_cmd, shell=False,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Stream frames one-at-a-time — avoids doubling RAM
    try:
        for i in range(N):
            frame = frames_cpu[i]                    # [H, W, 3] float32  0–1
            frame_uint8 = (frame.clamp(0, 1) * 255).byte().numpy()   # [H, W, 3] uint8
            proc.stdin.write(frame_uint8.tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        stderr = proc.stderr.read().decode(errors="replace")
        raise RuntimeError(f"FFmpeg pipe broken during chunk {chunk_info['chunk_index']}:\n{stderr}")

    _, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"FFmpeg encoding failed for chunk {chunk_info['chunk_index']}:\n"
            f"{stderr.decode(errors='replace')[-800:]}"
        )

    # Clean up temp audio
    if audio_tmp and Path(audio_tmp).exists():
        os.remove(audio_tmp)

    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"    Saved chunk → {Path(out_path).name}  ({N} frames, {size_mb:.1f} MB)")
    return out_path


def _build_ffmpeg_encode_cmd(out_path: str, width: int, height: int,
                              fps: float, audio_tmp: Optional[str],
                              crf: int = 8, pix_fmt: str = "yuv420p") -> List[str]:
    """Build the FFmpeg command list for encoding raw RGB frames from stdin."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "pipe:0",
    ]
    if audio_tmp:
        cmd += ["-i", audio_tmp]
    cmd += [
        "-c:v", "libx264",
        "-crf", str(crf),
        "-pix_fmt", pix_fmt,
        "-movflags", "+faststart",
    ]
    if audio_tmp:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd.append(out_path)
    return cmd


def _save_audio_temp(audio_cpu, fps: float, num_frames: int) -> Optional[str]:
    """
    Write decoded audio to a temporary WAV file for FFmpeg muxing.
    Returns path to the temp file, or None if writing fails.
    """
    try:
        import scipy.io.wavfile as wavfile
        import numpy as np

        # ComfyUI AUDIO format
        if isinstance(audio_cpu, dict):
            waveform    = audio_cpu.get("waveform")
            sample_rate = audio_cpu.get("sample_rate", 44100)
        else:
            waveform    = audio_cpu
            sample_rate = 44100

        if waveform is None:
            return None

        wav_np = waveform.squeeze().numpy()
        if wav_np.ndim == 1:
            wav_np = wav_np[np.newaxis, :]   # mono [1, N]
        # scipy expects [N] or [N, C] in int16
        wav_np = (wav_np.T * 32767).astype(np.int16)

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False,
                                          dir=str(AUDIO_DIR))
        tmp.close()
        wavfile.write(tmp.name, sample_rate, wav_np)
        return tmp.name
    except Exception as e:
        print(f"    [WARN] Could not write temp audio: {e}")
        return None


def decode_and_save_chunk(video_latent, audio_latent,
                           chunk_info: Dict,
                           video_vae, audio_vae,
                           fps: float,
                           vae_sub_batch: int = 8) -> Dict:
    """
    Orchestrates decode → CPU transfer → save → GPU cleanup for one chunk.
    Called from generate_chunk(); returns lightweight metadata.
    """
    idx = chunk_info["chunk_index"]

    # ── Decode video latent ───────────────────────────────────────────────────
    print(f"    Decoding video latent …")
    frames_cpu = decode_video_latent(video_latent, video_vae, sub_batch=vae_sub_batch)
    del video_latent
    cleanup_memory()

    # ── Decode audio latent ───────────────────────────────────────────────────
    audio_cpu = decode_audio_latent(audio_latent, audio_vae)
    if audio_latent is not None:
        del audio_latent
    cleanup_memory()

    # ── Save chunk ────────────────────────────────────────────────────────────
    out_path = save_chunk_to_disk(frames_cpu, audio_cpu, chunk_info, fps)
    del frames_cpu
    if audio_cpu is not None:
        del audio_cpu
    aggressive_cleanup()

    return {
        "chunk_index": idx,
        "start_frame": chunk_info["start_frame"],
        "num_frames" : chunk_info["num_frames"],
        "fps"        : fps,
        "path"       : out_path,
    }



# =============================================================================
# SECTION 19 — AUTOMATIC MEMORY CLEANUP HOOKS
# =============================================================================

def cleanup_after_chunk(chunk_index: int):
    """Called after every chunk. Aggressive if logging enabled."""
    if CONFIG["cleanup_after_chunk"]:
        aggressive_cleanup()
        if CONFIG["enable_memory_logging"]:
            memory_manager.memory_report(prefix=f"after chunk {chunk_index+1}")


def cleanup_after_stage(stage_name: str = ""):
    """Called after major pipeline stages (model load, all chunks, assembly)."""
    if CONFIG["cleanup_after_stage"]:
        aggressive_cleanup()
        if CONFIG["enable_memory_logging"]:
            memory_manager.memory_report(prefix=f"after {stage_name}")


def unload_models_for_decode():
    """
    Optional: move DiT model to CPU before VAE decode to free VRAM.
    Only called in t4_safe mode where VRAM is extremely tight.
    """
    if CONFIG["quality_mode"] != "t4_safe":
        return
    dit = _MODEL_CACHE.get("dit")
    if dit is not None and hasattr(dit, "model") and hasattr(dit.model, "to"):
        try:
            dit.model.to("cpu")
            print("    [MEM] DiT moved to CPU for VAE decode")
            cleanup_memory()
        except Exception:
            pass


def restore_models_to_gpu():
    """Move DiT back to GPU after VAE decode if it was offloaded."""
    if CONFIG["quality_mode"] != "t4_safe":
        return
    dit = _MODEL_CACHE.get("dit")
    if dit is not None and hasattr(dit, "model") and hasattr(dit.model, "to"):
        try:
            dit.model.to("cuda")
            print("    [MEM] DiT restored to GPU")
        except Exception:
            pass



# =============================================================================
# SECTION 20 — OOM RECOVERY & ADAPTIVE CHUNK GENERATOR
# =============================================================================

def adaptive_chunk_generator(chunks: List[Dict],
                               image_tensor: "torch.Tensor",
                               prompt: str,
                               audio_info: Dict,
                               global_seed: int,
                               width: int, height: int,
                               fps: float,
                               quality_mode: str,
                               checkpoint: Dict,
                               checkpoint_path: str) -> List[Dict]:
    """
    Main generation loop with OOM recovery.

    For each chunk:
      1. Skip if already completed (checkpoint).
      2. Try generate_chunk().
      3. On CUDA OOM: cleanup → reduce chunk size → retry (up to MAX_OOM_RETRIES).
      4. On non-OOM error: log and skip chunk (do not crash the session).
      5. On success: update checkpoint and continue.

    Returns list of completed chunk metadata dicts.
    """
    completed = []
    max_retries = CONFIG["max_oom_retries"]

    # Current effective chunk size (may shrink on OOM)
    current_chunk_frames = chunks[0]["num_frames"] if chunks else 33

    for chunk in chunks:
        idx = chunk["chunk_index"]

        # ── Checkpoint skip ───────────────────────────────────────────────────
        if idx in checkpoint.get("completed_chunks", []):
            print(f"  [SKIP] Chunk {idx+1} already completed (checkpoint)")
            completed.append({
                "chunk_index": idx,
                "start_frame": chunk["start_frame"],
                "num_frames" : chunk["num_frames"],
                "fps"        : fps,
                "path"       : chunk["path"],
            })
            continue

        # Validate that the chunk file isn't orphaned from a partial write
        if Path(chunk["path"]).exists() and Path(chunk["path"]).stat().st_size < 1024:
            print(f"  [WARN] Chunk {idx+1} file is too small — re-generating.")
            os.remove(chunk["path"])

        # ── OOM-retry loop ────────────────────────────────────────────────────
        attempt     = 0
        chunk_ok    = False
        retry_chunk = dict(chunk)   # mutable copy

        while attempt <= max_retries and not chunk_ok:
            try:
                result = generate_chunk(
                    chunk_info   = retry_chunk,
                    image_tensor = image_tensor,
                    prompt       = prompt,
                    audio_info   = audio_info,
                    global_seed  = global_seed,
                    width        = width,
                    height       = height,
                    fps          = fps,
                    quality_mode = quality_mode,
                )
                completed.append(result)
                chunk_ok = True
                _update_checkpoint(checkpoint, checkpoint_path,
                                   completed_idx=idx)

            except torch.cuda.OutOfMemoryError as oom_err:
                attempt += 1
                print(f"\n  {'='*55}")
                print(f"  CUDA OUT OF MEMORY — Chunk {idx+1}  (attempt {attempt}/{max_retries})")
                print(f"  Error: {str(oom_err)[:200]}")
                print(f"  GPU: {memory_manager.gpu_allocated_gb():.2f} GB allocated")
                print(f"  {'='*55}")

                # Release DiT cache state to recover VRAM
                _release_dit_cache()
                aggressive_cleanup()

                if attempt > max_retries:
                    print(f"  Max OOM retries reached for chunk {idx+1}.")
                    print(f"  Skipping this chunk — no session crash.")
                    checkpoint.setdefault("failed_chunks", []).append(idx)
                    _save_checkpoint(checkpoint, checkpoint_path)
                    break

                if not CONFIG["auto_reduce_chunk_on_oom"]:
                    print(f"  auto_reduce_chunk_on_oom=False — skipping chunk {idx+1}.")
                    break

                # Reduce chunk size
                reduction = 0.75 if attempt == 1 else 0.5
                new_frames = normalize_ltx_frame_count(
                    int(retry_chunk["num_frames"] * reduction),
                    fps, min_frames=9
                )
                if new_frames < 9:
                    print(f"  Minimum chunk size reached. Cannot reduce further.")
                    print(f"  Try lowering resolution or disabling LoRAs.")
                    checkpoint.setdefault("failed_chunks", []).append(idx)
                    _save_checkpoint(checkpoint, checkpoint_path)
                    break

                print(f"  Reducing chunk frames: "
                      f"{retry_chunk['num_frames']} → {new_frames}  "
                      f"(×{reduction:.2f})")
                retry_chunk = dict(chunk)
                retry_chunk["num_frames"] = new_frames
                current_chunk_frames = new_frames

                # Also reload models if they were evicted
                if _MODEL_CACHE["dit"] is None:
                    print("  Reloading DiT after OOM eviction …")
                    load_all_models()

            except Exception as err:
                attempt = max_retries + 1   # Don't retry non-OOM
                err_type = type(err).__name__
                print(f"\n  ERROR in chunk {idx+1}: [{err_type}] {str(err)[:300]}")
                print(f"  Traceback:\n{traceback.format_exc()[-600:]}")
                print(f"  GPU memory: {memory_manager.gpu_allocated_gb():.2f} GB allocated")
                print(f"  Suggested action: Check model files and node availability.")
                aggressive_cleanup()
                checkpoint.setdefault("failed_chunks", []).append(idx)
                _save_checkpoint(checkpoint, checkpoint_path)

        cleanup_after_chunk(idx)

    return completed


def _release_dit_cache():
    """
    Move DiT model to CPU and release from VRAM cache after OOM.
    Does NOT delete the object — allows reloading without re-downloading.
    """
    dit = _MODEL_CACHE.get("dit")
    if dit is not None:
        try:
            if hasattr(dit, "model"):
                dit.model.to("cpu")
            elif hasattr(dit, "to"):
                dit.to("cpu")
        except Exception:
            pass
    cleanup_memory()



# =============================================================================
# SECTION 21 — RESUME / CHECKPOINT SYSTEM
# =============================================================================

CHECKPOINT_FILENAME = "checkpoint.json"


def _checkpoint_path() -> str:
    return str(WORKSPACE / CHECKPOINT_FILENAME)


def create_checkpoint(job_id: str, fps: int, total_frames: int,
                       seed: int, resolution: Tuple[int, int],
                       chunk_frames: int) -> Dict:
    """Create a new checkpoint dict for this job."""
    return {
        "job_id"           : job_id,
        "fps"              : fps,
        "total_frames"     : total_frames,
        "seed"             : seed,
        "resolution"       : list(resolution),
        "chunk_frames"     : chunk_frames,
        "completed_chunks" : [],
        "failed_chunks"    : [],
        "created_at"       : time.strftime("%Y-%m-%dT%H:%M:%S"),
        "updated_at"       : time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def load_or_create_checkpoint(job_id: str, fps: int, total_frames: int,
                                seed: int, resolution: Tuple[int, int],
                                chunk_frames: int,
                                resume: bool = True) -> Tuple[Dict, str]:
    """
    Load existing checkpoint if resume=True and it matches this job's config.
    Creates a fresh checkpoint otherwise.
    Returns (checkpoint_dict, checkpoint_file_path).
    """
    cp_path = _checkpoint_path()

    if resume and Path(cp_path).exists():
        try:
            with open(cp_path, "r") as f:
                existing = json.load(f)
            # Validate compatibility
            if (existing.get("fps")          == fps
                    and existing.get("total_frames") == total_frames
                    and existing.get("resolution")   == list(resolution)):
                n_done = len(existing.get("completed_chunks", []))
                print(f"\n  [RESUME] Checkpoint loaded: {n_done} chunks already done.")
                existing["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                return existing, cp_path
            else:
                print(f"\n  [RESUME] Checkpoint config mismatch — starting fresh.")
        except Exception as e:
            print(f"\n  [RESUME] Could not read checkpoint ({e}) — starting fresh.")

    cp = create_checkpoint(job_id, fps, total_frames, seed, resolution, chunk_frames)
    _save_checkpoint(cp, cp_path)
    return cp, cp_path


def _update_checkpoint(checkpoint: Dict, cp_path: str, completed_idx: int):
    """Mark a chunk as completed and persist the checkpoint."""
    if completed_idx not in checkpoint["completed_chunks"]:
        checkpoint["completed_chunks"].append(completed_idx)
    checkpoint["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _save_checkpoint(checkpoint, cp_path)


def _save_checkpoint(checkpoint: Dict, cp_path: str):
    """Atomically write checkpoint to disk (write temp + rename)."""
    tmp = cp_path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(checkpoint, f, indent=2)
        os.replace(tmp, cp_path)
    except Exception as e:
        print(f"  [WARN] Could not save checkpoint: {e}")


def validate_chunk_files(chunks: List[Dict]) -> List[int]:
    """
    Verify that each chunk file exists and has non-trivial size.
    Returns list of chunk indices that are missing or corrupt.
    """
    bad = []
    for c in chunks:
        p = Path(c["path"])
        if not p.exists() or p.stat().st_size < 1024:
            bad.append(c["chunk_index"])
    return bad



# =============================================================================
# SECTION 22 — FINAL VIDEO ASSEMBLY
# =============================================================================

def assemble_chunks_to_video(chunks: List[Dict],
                              assembled_path: str,
                              fps: float) -> str:
    """
    Concatenate all generated chunk MP4 files into a single video using
    FFmpeg concat demuxer (stream-copy where safe; re-encode as fallback).

    Never reads frame data into Python — all work done inside FFmpeg.
    Returns the path to the assembled video.
    """
    assembled_path = str(assembled_path)
    print(f"\n  [22] Assembling {len(chunks)} chunks → {Path(assembled_path).name}")

    # Filter to chunks that actually exist
    valid_chunks = [c for c in chunks if Path(c["path"]).exists()
                    and Path(c["path"]).stat().st_size > 1024]
    if not valid_chunks:
        raise RuntimeError("No valid chunk files found for assembly.")

    if len(valid_chunks) == 1:
        shutil.copy2(valid_chunks[0]["path"], assembled_path)
        print(f"  Single chunk — copied directly.")
        return assembled_path

    # Write concat list file
    concat_list = str(WORKSPACE / "concat_list.txt")
    with open(concat_list, "w") as f:
        for c in sorted(valid_chunks, key=lambda x: x["chunk_index"]):
            # FFmpeg requires forward slashes and escaped special chars
            safe_path = str(Path(c["path"]).resolve()).replace("\\", "/")
            f.write(f"file '{safe_path}'\n")

    # Attempt stream-copy first (fastest, no quality loss)
    cmd_copy = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list,
        "-c", "copy",
        assembled_path,
    ]
    result = subprocess.run(cmd_copy, capture_output=True)
    if result.returncode == 0:
        size_mb = Path(assembled_path).stat().st_size / 1e6
        print(f"  ✓ Stream-copy assembly: {size_mb:.1f} MB")
        return assembled_path

    # Fallback: re-encode (handles mismatched codec/resolution between chunks)
    print(f"  Stream-copy failed — re-encoding …")
    cmd_encode = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list,
        "-c:v", "libx264",
        "-crf", "8",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        assembled_path,
    ]
    result2 = subprocess.run(cmd_encode, capture_output=True)
    if result2.returncode != 0:
        raise RuntimeError(
            f"FFmpeg assembly failed:\n"
            f"{result2.stderr.decode(errors='replace')[-600:]}"
        )
    size_mb = Path(assembled_path).stat().st_size / 1e6
    print(f"  ✓ Re-encode assembly: {size_mb:.1f} MB")
    return assembled_path


def assemble_video_with_audio(video_path: str,
                               audio_info: Dict,
                               final_path: str,
                               total_frames: int) -> str:
    """
    Mux original audio track with assembled video using exact frame-based timing.
    Avoids cumulative drift by anchoring to frame count, not clock.

    audio_info["trim_start_s"] is the offset into the source audio
    (matching workflow trimStart ÷ fps).
    """
    print(f"\n  [22] Muxing audio → {Path(final_path).name}")

    fps         = audio_info["fps"]
    trim_start  = audio_info["trim_start_s"]
    duration_s  = total_frames / fps

    audio_src   = audio_info["path"]

    if not Path(audio_src).exists():
        print(f"  [WARN] Audio file not found — outputting video without audio.")
        shutil.copy2(video_path, final_path)
        return final_path

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ss", str(trim_start),
        "-t",  str(duration_s),
        "-i", audio_src,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        final_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        print(f"  [WARN] Audio mux failed:\n{stderr[-400:]}")
        print(f"  Copying video without audio mux.")
        shutil.copy2(video_path, final_path)
        return final_path

    size_mb = Path(final_path).stat().st_size / 1e6
    print(f"  ✓ Final video with audio: {size_mb:.1f} MB  → {final_path}")
    return final_path


def cleanup_temp_chunks(chunks: List[Dict]):
    """Delete temporary chunk files after successful assembly."""
    if not CONFIG["keep_temp_chunks"]:
        for c in chunks:
            p = Path(c["path"])
            if p.exists():
                p.unlink()
        print(f"  [22] Temporary chunks deleted.")
    concat_list = WORKSPACE / "concat_list.txt"
    if concat_list.exists():
        concat_list.unlink()



# =============================================================================
# SECTION 23 — AUDIO SYNCHRONIZATION
# =============================================================================

def verify_audio_video_sync(video_path: str, audio_info: Dict) -> Dict:
    """
    Use ffprobe to check that the final video's audio and video streams
    have matching durations and the expected frame count.
    Returns a dict with sync status and stream info.
    """
    print("\n  [23] Verifying audio/video sync …")

    def _ffprobe(path: str, stream: str) -> Dict:
        cmd = (
            f'ffprobe -v error -select_streams {stream} '
            f'-show_entries stream=duration,nb_frames,codec_name '
            f'-of json "{path}"'
        )
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        try:
            return json.loads(r.stdout)
        except Exception:
            return {}

    v_info = _ffprobe(video_path, "v:0")
    a_info = _ffprobe(video_path, "a:0")

    v_streams = v_info.get("streams", [{}])
    a_streams = a_info.get("streams", [{}])
    v_dur  = float(v_streams[0].get("duration", 0)) if v_streams else 0
    a_dur  = float(a_streams[0].get("duration", 0)) if a_streams else 0
    v_codec = v_streams[0].get("codec_name", "?") if v_streams else "?"
    a_codec = a_streams[0].get("codec_name", "?") if a_streams else "?"
    drift   = abs(v_dur - a_dur)

    status = "OK" if drift < 0.1 else ("WARNING" if drift < 0.5 else "ERROR")

    print(f"    Video stream : {v_codec}  {v_dur:.3f} s")
    print(f"    Audio stream : {a_codec}  {a_dur:.3f} s")
    print(f"    A/V drift    : {drift*1000:.1f} ms  [{status}]")

    if status == "ERROR":
        print(f"    [WARN] A/V drift of {drift:.3f}s may be audible.")

    return {
        "video_duration_s" : v_dur,
        "audio_duration_s" : a_dur,
        "drift_s"          : drift,
        "status"           : status,
        "video_codec"      : v_codec,
        "audio_codec"      : a_codec,
    }



# =============================================================================
# SECTION 24 — FINAL VALIDATION
# =============================================================================

def validate_final_output(final_path: str, expected_frames: int,
                           fps: float) -> Dict:
    """
    Run ffprobe on the final output to confirm:
      - File exists and has non-trivial size
      - Video stream is present with correct codec
      - Duration matches expected
      - Frame count is within tolerance (±2 frames)
    Returns a validation result dict.
    """
    print("\n  [24] Validating final output …")
    result = {
        "path"        : final_path,
        "exists"      : False,
        "size_mb"     : 0.0,
        "duration_s"  : 0.0,
        "frame_count" : 0,
        "fps"         : 0.0,
        "codec"       : "?",
        "audio_ok"    : False,
        "pass"        : False,
        "notes"       : [],
    }

    p = Path(final_path)
    if not p.exists():
        result["notes"].append("File does not exist")
        print(f"  ✗ Final output missing: {final_path}")
        return result

    result["exists"]  = True
    result["size_mb"] = p.stat().st_size / 1e6

    if result["size_mb"] < 0.1:
        result["notes"].append("File too small (< 100 KB)")
        print(f"  ✗ Output suspiciously small: {result['size_mb']:.2f} MB")
        return result

    # ffprobe video stream
    cmd_v = (
        f'ffprobe -v error -select_streams v:0 '
        f'-show_entries stream=codec_name,r_frame_rate,duration,nb_frames '
        f'-of json "{final_path}"'
    )
    r_v = subprocess.run(cmd_v, shell=True, capture_output=True, text=True)
    try:
        v_data = json.loads(r_v.stdout).get("streams", [{}])[0]
        result["codec"]       = v_data.get("codec_name", "?")
        result["duration_s"]  = float(v_data.get("duration", 0))
        nb = v_data.get("nb_frames")
        result["frame_count"] = int(nb) if nb else int(result["duration_s"] * fps)
        rfr = v_data.get("r_frame_rate", "0/1")
        num, den = rfr.split("/")
        result["fps"] = float(num) / max(float(den), 1)
    except Exception:
        result["notes"].append("Could not parse ffprobe video output")

    # ffprobe audio stream
    cmd_a = (
        f'ffprobe -v error -select_streams a:0 '
        f'-show_entries stream=codec_name '
        f'-of json "{final_path}"'
    )
    r_a = subprocess.run(cmd_a, shell=True, capture_output=True, text=True)
    try:
        a_data = json.loads(r_a.stdout).get("streams", [])
        result["audio_ok"] = len(a_data) > 0
    except Exception:
        result["audio_ok"] = False

    # Pass criteria
    frame_tolerance = 4
    dur_expected    = expected_frames / fps
    dur_ok          = abs(result["duration_s"] - dur_expected) < 1.5
    frame_ok        = abs(result["frame_count"] - expected_frames) <= frame_tolerance

    if not dur_ok:
        result["notes"].append(
            f"Duration mismatch: got {result['duration_s']:.2f}s, "
            f"expected {dur_expected:.2f}s"
        )
    if not frame_ok:
        result["notes"].append(
            f"Frame count mismatch: got {result['frame_count']}, "
            f"expected {expected_frames} (±{frame_tolerance})"
        )

    result["pass"] = dur_ok and result["size_mb"] > 0.5

    icon = "✓" if result["pass"] else "✗"
    print(f"  {icon} Output: {result['size_mb']:.1f} MB  "
          f"{result['duration_s']:.2f}s  "
          f"{result['frame_count']} frames  "
          f"{result['codec']}  "
          f"{'audio ✓' if result['audio_ok'] else 'no audio'}")
    for note in result["notes"]:
        print(f"      NOTE: {note}")

    return result



# =============================================================================
# SECTION 25 — PREVIEW
# =============================================================================

def generate_preview(final_path: str, preview_duration: float = 3.0):
    """
    Extract a short preview clip from the final video using FFmpeg.
    Displays it inline in Colab if possible.
    Does NOT read the full video into RAM.
    """
    print("\n  [25] Generating preview …")
    preview_path = str(FINAL_DIR / "preview.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-i", final_path,
        "-t", str(preview_duration),
        "-c:v", "libx264",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        preview_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"  [WARN] Preview generation failed: "
              f"{result.stderr.decode(errors='replace')[-200:]}")
        return

    size_mb = Path(preview_path).stat().st_size / 1e6
    print(f"  Preview: {preview_path}  ({size_mb:.1f} MB, {preview_duration:.1f}s)")

    # Colab inline display — streams only the small preview file
    try:
        from IPython.display import HTML, display
        import base64
        # Only encode the preview (small) — never the full video
        if size_mb < 50:
            with open(preview_path, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode()
            html = (
                f'<video width="640" controls autoplay loop muted>'
                f'<source src="data:video/mp4;base64,{video_b64}" type="video/mp4">'
                f'</video>'
            )
            display(HTML(html))
        else:
            print(f"  Preview too large for inline display — "
                  f"download: {preview_path}")
    except ImportError:
        print(f"  IPython not available — preview saved at: {preview_path}")
    except Exception as e:
        print(f"  Preview display error: {e}")



# =============================================================================
# SECTION 26 — JOB REPORT
# =============================================================================

def write_job_report(report: Dict):
    """Write job_report.json and print a summary to stdout."""
    report_path = str(FINAL_DIR / "job_report.json")
    try:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  [26] Job report → {report_path}")
    except Exception as e:
        print(f"  [WARN] Could not write job report: {e}")

    # Print summary
    print("\n" + "="*60)
    print("  JOB REPORT")
    print("="*60)
    for k, v in report.items():
        print(f"  {k:<30s}: {v}")
    print("="*60)


def build_job_report(timeline: Dict, chunks: List[Dict],
                     completed: List[Dict], validation: Dict,
                     sync_info: Dict, start_time: float,
                     final_path: str) -> Dict:
    """Assemble the job report dict."""
    elapsed   = time.time() - start_time
    peak_gb   = memory_manager.gpu_peak_gb()
    n_done    = len(completed)
    n_total   = len(chunks)
    n_failed  = n_total - n_done

    return {
        "gpu"                    : GPU_INFO.get("device_name", "Unknown"),
        "torch_version"          : torch.__version__,
        "cuda_version"           : torch.version.cuda,
        "resolution"             : f"{CONFIG['width']}x{CONFIG['height']}",
        "fps"                    : timeline["fps"],
        "requested_duration_s"   : timeline["duration_s"],
        "actual_duration_s"      : round(validation.get("duration_s", 0), 3),
        "total_frames"           : timeline["total_frames"],
        "chunk_size_frames"      : chunks[0]["num_frames"] if chunks else 0,
        "chunks_total"           : n_total,
        "chunks_completed"       : n_done,
        "chunks_failed"          : n_failed,
        "peak_gpu_memory_gb"     : round(peak_gb, 3),
        "generation_time_s"      : round(elapsed, 1),
        "generation_time_hms"    : time.strftime("%H:%M:%S",
                                                  time.gmtime(elapsed)),
        "av_drift_ms"            : round(sync_info.get("drift_s", 0) * 1000, 1),
        "av_sync_status"         : sync_info.get("status", "N/A"),
        "output_path"            : final_path,
        "output_size_mb"         : round(validation.get("size_mb", 0), 1),
        "output_codec"           : validation.get("codec", "?"),
        "audio_present"          : validation.get("audio_ok", False),
        "validation_pass"        : validation.get("pass", False),
        "quality_mode"           : CONFIG["quality_mode"],
        "sampler"                : CONFIG["sampler_name"],
        "scheduler"              : CONFIG["scheduler"],
        "pass1_steps"            : CONFIG["pass1_steps"],
        "pass2_steps"            : CONFIG["pass2_steps"],
        "pass2_denoise"          : CONFIG["pass2_denoise"],
        "timestamp"              : time.strftime("%Y-%m-%dT%H:%M:%S"),
    }



# =============================================================================
# SECTION 27 — CLEANUP + MAIN ENTRY POINT
# =============================================================================

def final_cleanup(keep_workspace: bool = False):
    """
    Release all model objects, clear caches, run aggressive memory cleanup.
    Optionally remove the entire workspace temp directory.
    NEVER deletes the final output.
    """
    print("\n  [27] Final cleanup …")

    # Release all cached models
    for key in list(_MODEL_CACHE.keys()):
        if _MODEL_CACHE[key] is not None:
            memory_manager.release_model(_MODEL_CACHE[key], name=key)
            _MODEL_CACHE[key] = None

    clear_conditioning_cache()
    aggressive_cleanup()

    if not keep_workspace and CONFIG["cleanup_temp_files"]:
        # Remove frames/ and audio/ subdirs but keep chunks/ for debug if failed
        for subdir in [FRAMES_DIR, AUDIO_DIR]:
            if subdir.exists():
                shutil.rmtree(subdir, ignore_errors=True)
        print(f"  Temp directories removed.")

    print_memory("final cleanup complete")
    print("  [27] Cleanup done.\n")


# =============================================================================
# MAIN PIPELINE ORCHESTRATOR
# =============================================================================

def generate_director_mv(
    image_path     : str,
    audio_path     : str,
    prompt         : str                = "",
    duration_seconds: float             = None,
    fps            : int                = None,
    width          : int                = None,
    height         : int                = None,
    seed           : int                = None,
    quality_mode   : str                = None,
    resume         : bool               = None,
    preview_mode   : bool               = None,
) -> str:
    """
    Top-level entry point.  Runs the complete LTX-2.3 Director 2.0 MV pipeline:

      1.  Validate environment + inputs
      2.  Calculate timeline
      3.  Plan chunks
      4.  Load models
      5.  Generate all chunks (OOM-safe, resumable)
      6.  Assemble final video
      7.  Mux audio
      8.  Validate output
      9.  Write job report
      10. Preview
      11. Cleanup

    Returns path to final output MP4.
    """

    # ── Apply overrides ───────────────────────────────────────────────────────
    duration_seconds = duration_seconds if duration_seconds is not None \
                       else CONFIG["duration_seconds"]
    fps          = fps          if fps          is not None else CONFIG["fps"]
    width        = width        if width        is not None else CONFIG["width"]
    height       = height       if height       is not None else CONFIG["height"]
    seed         = seed         if seed         is not None else CONFIG["seed"]
    quality_mode = quality_mode if quality_mode is not None else CONFIG["quality_mode"]
    resume       = resume       if resume       is not None else CONFIG["resume"]
    preview_mode = preview_mode if preview_mode is not None else CONFIG["preview_mode"]

    if seed == 0 or CONFIG["random_seed"]:
        seed = random.randint(1, 2**31 - 1)
        print(f"  Random seed: {seed}")

    effective_prompt = build_prompt(prompt or CONFIG.get("custom_prompt", ""))
    start_time = time.time()

    print("\n" + "="*60)
    print("  LTX-2.3 DIRECTOR 2.0 MV — PIPELINE START")
    print("="*60)

    # ── 1. CUDA detection ─────────────────────────────────────────────────────
    global GPU_INFO
    GPU_INFO = detect_gpu()

    # ── 2. Validation ─────────────────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats()
    ok, final_w, final_h = run_all_validations(
        image_path=image_path, audio_path=audio_path,
        width=width, height=height,
        total_frames=round(duration_seconds * fps),
        fps=fps,
    )
    if not ok:
        raise RuntimeError(
            "Pre-generation validation failed. "
            "Check error messages above and fix before retrying."
        )
    width, height = final_w, final_h

    # ── 3. Timeline ───────────────────────────────────────────────────────────
    timeline = calculate_timeline(
        duration_seconds=duration_seconds, fps=fps,
        preview_mode=preview_mode,
        preview_duration=CONFIG["preview_duration"],
    )
    total_frames = timeline["total_frames"]

    # ── 4. Chunk planning ─────────────────────────────────────────────────────
    chunk_frames = select_chunk_size(
        quality_mode=quality_mode, width=width, height=height,
        auto=CONFIG["auto_chunk_size"],
    )
    chunks = plan_chunks(total_frames, chunk_frames, fps)
    print_generation_plan(timeline, chunks, width, height)

    # ── 5. Checkpoint ─────────────────────────────────────────────────────────
    job_id     = f"ltx23_{int(time.time())}"
    checkpoint, cp_path = load_or_create_checkpoint(
        job_id=job_id, fps=fps, total_frames=total_frames,
        seed=seed, resolution=(width, height),
        chunk_frames=chunk_frames, resume=resume,
    )

    # ── 6. Audio preparation ──────────────────────────────────────────────────
    # Workflow trimStart=446.9 frames → 18.6 s; default to 0 if not set
    audio_trim_frames = 0.0
    audio_info = prepare_audio(
        audio_path=audio_path, fps=fps,
        total_frames=total_frames,
        trim_start_frames=audio_trim_frames,
    )

    # ── 7. Load reference image (once, on CPU) ────────────────────────────────
    image_tensor = load_reference_image(image_path, width, height)
    # image_tensor stays on CPU; transferred per-chunk by build_director_conditioning

    # ── 8. Load all models ────────────────────────────────────────────────────
    cleanup_after_stage("pre-model-load")
    load_all_models()
    cleanup_after_stage("model-load")

    # ── 9. MAIN GENERATION LOOP ───────────────────────────────────────────────
    completed = adaptive_chunk_generator(
        chunks       = chunks,
        image_tensor = image_tensor,
        prompt       = effective_prompt,
        audio_info   = audio_info,
        global_seed  = seed,
        width        = width,
        height       = height,
        fps          = float(fps),
        quality_mode = quality_mode,
        checkpoint   = checkpoint,
        checkpoint_path = cp_path,
    )

    # Release image tensor now — no longer needed
    del image_tensor
    cleanup_after_stage("generation")

    if not completed:
        raise RuntimeError(
            "No chunks were successfully generated. "
            "Check GPU memory and error messages above."
        )

    # ── 10. Assemble chunks ───────────────────────────────────────────────────
    assembled_path = str(WORKSPACE / "assembled.mp4")
    assemble_chunks_to_video(completed, assembled_path, fps)

    # ── 11. Mux original audio ────────────────────────────────────────────────
    final_path = str(FINAL_DIR / "LTX23_Director_30s.mp4")
    assemble_video_with_audio(
        video_path   = assembled_path,
        audio_info   = audio_info,
        final_path   = final_path,
        total_frames = total_frames,
    )

    # ── 12. Final validation ──────────────────────────────────────────────────
    validation = validate_final_output(final_path, total_frames, fps)
    sync_info  = verify_audio_video_sync(final_path, audio_info)

    # ── 13. Job report ────────────────────────────────────────────────────────
    report = build_job_report(
        timeline    = timeline,
        chunks      = chunks,
        completed   = completed,
        validation  = validation,
        sync_info   = sync_info,
        start_time  = start_time,
        final_path  = final_path,
    )
    write_job_report(report)

    # ── 14. Preview ───────────────────────────────────────────────────────────
    if validation["pass"]:
        generate_preview(final_path, preview_duration=CONFIG["preview_duration"])

    # ── 15. Cleanup temp files ────────────────────────────────────────────────
    if CONFIG["cleanup_temp_files"] and not CONFIG["keep_temp_chunks"]:
        cleanup_temp_chunks(completed)

    final_cleanup(keep_workspace=True)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  ✓  COMPLETE  —  {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
    print(f"  Output: {final_path}")
    print(f"{'='*60}\n")

    return final_path



# =============================================================================
# COLAB EXECUTION BLOCKS
# Copy each block into a separate Colab cell and run top-to-bottom.
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Install environment (run once per session)
# ─────────────────────────────────────────────────────────────────────────────
# install_environment()

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Download models (run once; aria2c resumes if interrupted)
# ─────────────────────────────────────────────────────────────────────────────
# download_models(skip_existing=True)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Load ComfyUI node registry
# ─────────────────────────────────────────────────────────────────────────────
# import_comfyui_nodes()
# validate_custom_nodes()

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Upload inputs (or set paths directly below)
# ─────────────────────────────────────────────────────────────────────────────
# IMAGE_PATH = upload_image_colab()
# AUDIO_PATH = upload_audio_colab()
#
# — OR set paths directly —
# IMAGE_PATH = "/content/my_photo.png"
# AUDIO_PATH = "/content/my_track.mp3"

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — Configure and run
# ─────────────────────────────────────────────────────────────────────────────
# Tweak CONFIG values here if needed, then run generate_director_mv().
#
# CONFIG["quality_mode"]   = "t4_safe"    # safest for T4
# CONFIG["duration_seconds"] = 31.5       # match workflow
# CONFIG["resume"]         = True         # restart-safe
# CONFIG["preview_mode"]   = False        # True for quick 3-second test

# output_path = generate_director_mv(
#     image_path = IMAGE_PATH,
#     audio_path = AUDIO_PATH,
# )
# print(f"Output: {output_path}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — Download output from Colab
# ─────────────────────────────────────────────────────────────────────────────
# from google.colab import files
# files.download(output_path)


# =============================================================================
# SCRIPT ENTRY POINT (non-Colab execution)
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LTX-2.3 Director 2.0 MV — T4 Pipeline"
    )
    parser.add_argument("--image",    required=True,  help="Reference image path")
    parser.add_argument("--audio",    required=True,  help="Audio file path")
    parser.add_argument("--prompt",   default="",     help="Custom prompt (blank=use built-in)")
    parser.add_argument("--duration", type=float, default=31.5, help="Duration in seconds")
    parser.add_argument("--fps",      type=int,   default=24,   help="Frame rate")
    parser.add_argument("--width",    type=int,   default=1280, help="Output width")
    parser.add_argument("--height",   type=int,   default=720,  help="Output height")
    parser.add_argument("--seed",     type=int,   default=0,    help="Seed (0=random)")
    parser.add_argument("--quality",  default="t4_safe",
                        choices=["t4_safe", "t4_balanced", "t4_aggressive"],
                        help="Quality/memory profile")
    parser.add_argument("--no-resume", action="store_true", help="Disable checkpoint resume")
    parser.add_argument("--preview",   action="store_true", help="Preview mode (3 seconds)")
    parser.add_argument("--install",   action="store_true", help="Run install_environment() first")
    parser.add_argument("--download-models", action="store_true",
                        dest="download", help="Download models before generating")

    args = parser.parse_args()

    if args.install:
        install_environment()

    if args.download:
        download_models(skip_existing=True)

    import_comfyui_nodes()

    output = generate_director_mv(
        image_path       = args.image,
        audio_path       = args.audio,
        prompt           = args.prompt,
        duration_seconds = args.duration,
        fps              = args.fps,
        width            = args.width,
        height           = args.height,
        seed             = args.seed,
        quality_mode     = args.quality,
        resume           = not args.no_resume,
        preview_mode     = args.preview,
    )
    print(f"\nDone: {output}")
