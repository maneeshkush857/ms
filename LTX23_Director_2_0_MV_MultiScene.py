# -*- coding: utf-8 -*-
# =============================================================================
# LTX23_Director_2_0_MV_MultiScene.py  —  v4.0  COMPLETE REWRITE
#
# LTX-2.3 Director 2.0  Multi-Scene Infinite Flow Engine
# Google Colab T4  (14.56 GB VRAM)  —  Crash-Resilient  —  Auto-Resume
#
# ARCHITECTURE  (based on experiment_ltx23.py — proven working pipeline):
#   • Uses ManualSigmas instead of BasicScheduler (avoids PromptServer deps)
#   • Calls NODE_CLASS_MAPPINGS directly  —  no EXECUTE_NORMALIZED on sampler
#   • PromptServer mock provides ALL attributes KJNodes + VideoHelperSuite need
#   • comfy.model_management.unload_all_models() between sampling and decode
#   • Per-scene JSON checkpoint  —  auto-resumes after ANY crash
#   • 8-shot SCENE_JSON storyboard with camera/emotion/dialogue per shot
#
# USAGE (Google Colab):
#   CELL 1:  install_environment()
#   CELL 2:  download_all_models()
#   CELL 3:  upload reference image + audio  (optional)
#   CELL 4:  output = generate_multiscene_mv()   ← re-run after crash to resume
#   CELL 5:  files.download(output)
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1  —  CONFIGURATION  (@param Colab form widgets)
# ─────────────────────────────────────────────────────────────────────────────

# @markdown ### 📁 Input Files
IMAGE_PATH = "/content/ComfyUI/input/reference.png"  # @param {type:"string"}
AUDIO_PATH = "/content/ComfyUI/input/audio.mp3"      # @param {type:"string"}

# @markdown ### 🎬 Project
PROJECT_NAME     = "LTX23_MultiScene"          # @param {type:"string"}
SCENE_DURATION_S = 4.0                         # @param {type:"number"}
FPS              = 24                          # @param [8,12,16,24,25] {type:"raw"}
OUTPUT_WIDTH     = 832                         # @param {type:"integer"}
OUTPUT_HEIGHT    = 480                         # @param {type:"integer"}
OUTPUT_FILENAME  = "LTX23_MultiScene_Final.mp4" # @param {type:"string"}

# @markdown ### 🎲 Seed
SEED        = 42     # @param {type:"integer"}
RANDOM_SEED = False  # @param {type:"boolean"}

# @markdown ### 🖥️ Quality
# t4_safe = 49-frame chunks, 832×480  ← use this on T4
# t4_balanced = 73-frame chunks, 832×480  (more VRAM risk)
QUALITY_MODE = "t4_safe"  # @param ["t4_safe","t4_balanced"]

# @markdown ### 💾 Resume
RESUME = True  # @param {type:"boolean"}

# @markdown ### 🔁 Retries
MAX_SCENE_RETRIES = 3  # @param {type:"integer"}

# @markdown ### 📂 Paths
COMFYUI_DIR   = "/content/ComfyUI"          # @param {type:"string"}
WORKSPACE_DIR = "/content/ltx23_workspace"  # @param {type:"string"}
OUTPUT_DIR    = "/content/ltx23_output"     # @param {type:"string"}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2  —  SCENE JSON  (edit this to define your multi-scene video)
# ─────────────────────────────────────────────────────────────────────────────

SCENE_JSON = {
    "scene_id": "music_video_01",
    "project_name": PROJECT_NAME,
    "video_style": (
        "Photorealistic cinematic music video, ultra-high fidelity, "
        "blockbuster concert performance"
    ),
    "environment": {
        "location":      "Sold-out stadium concert stage with massive LED screen backdrop",
        "mood":          "Explosive energy, charismatic performance",
        "lighting":      "Dynamic concert lighting, neon rims, volumetric haze",
        "color_palette": "Electric blues, hot pinks, deep blacks, gold highlights",
    },
    "main_characters": [{
        "name": "Singer",
        "detailed_appearance": {
            "face":       "Expressive, charismatic, intense performance energy",
            "hair":       "Styled naturally, matches reference image exactly",
            "clothing":   "Stage performance outfit, matches reference image exactly",
            "build":      "Athletic, confident stage presence",
            "skin_tone":  "Matches reference image exactly",
            "accessories":"Wireless microphone, stage lighting",
        },
    }],
    "story_action": {
        "shots": [
            {"time":"0-4s",   "camera":"Wide establishing shot",
             "camera_movement":"dolly_forward", "motion_intensity":0.5,
             "action":"Singer enters from stage right into spotlight. Crowd erupts.",
             "emotion":"energetic_entrance", "prompt_override":""},
            {"time":"4-8s",   "camera":"Medium tracking shot, low-angle hero",
             "camera_movement":"low_angle_hero", "motion_intensity":0.7,
             "action":"Singer moves to front of stage, commanding the room.",
             "emotion":"powerful_dominance", "prompt_override":""},
            {"time":"8-12s",  "camera":"Extreme close-up on face",
             "camera_movement":"static_intense", "motion_intensity":0.3,
             "action":"Close-up of singer's face, lips perfectly matching the beat.",
             "emotion":"intense_passionate", "prompt_override":""},
            {"time":"12-16s", "camera":"Fast pull-back to wide then push-in",
             "camera_movement":"dolly_forward", "motion_intensity":0.8,
             "action":"Camera pull-back reveals full stage. LED walls explode with visuals.",
             "emotion":"triumphant_climax", "prompt_override":""},
            {"time":"16-20s", "camera":"360 orbit starting left",
             "camera_movement":"orbit_left", "motion_intensity":0.6,
             "action":"Camera orbits around singer performing center stage.",
             "emotion":"building_intensity", "prompt_override":""},
            {"time":"20-24s", "camera":"Wide aerial tilt down",
             "camera_movement":"tilt_down", "motion_intensity":0.4,
             "action":"Tilt-down from overhead. Singer small against massive crowd.",
             "emotion":"awe_inspiring_scale", "prompt_override":""},
            {"time":"24-28s", "camera":"Close tracking from side",
             "camera_movement":"dolly_right", "motion_intensity":0.5,
             "action":"Side-angle tracking follows singer moving across stage.",
             "emotion":"confident_swagger", "prompt_override":""},
            {"time":"28-32s", "camera":"Fast push-in for final close-up",
             "camera_movement":"zoom_in_fast", "motion_intensity":0.9,
             "action":"Final aggressive push-in to singer's face. Maximum emotion.",
             "emotion":"maximum_intensity_finale", "prompt_override":""},
        ],
    },
    "dialogue_with_timing": [],
    "audio": {
        "background_music": "High-energy pop/hip-hop with hard beats",
        "environment_sfx":  "Stadium crowd roar, echo",
        "voice_processing": "Live concert mix",
    },
}

BASE_PROMPT = (
    "Photorealistic cinematic music video. Singer performing live on stadium stage. "
    "Ultra-high facial fidelity, preserve exact identity from reference image. "
    "Blockbuster concert production quality. Dynamic concert lighting."
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3  —  IMPORTS & CUDA
# ─────────────────────────────────────────────────────────────────────────────

import os, sys, gc, json, time, shutil, hashlib, subprocess, traceback, math
import asyncio, threading, concurrent.futures, warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from functools import lru_cache
import random as _random

# Must be set BEFORE torch import
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
    import ctypes
    _LIBC = ctypes.CDLL("libc.so.6")
except Exception:
    _LIBC = None

def _malloc_trim():
    if _LIBC:
        try: _LIBC.malloc_trim(0)
        except Exception: pass

def _gpu_free_gb():
    if not torch.cuda.is_available(): return 0.0
    free, _ = torch.cuda.mem_get_info(0)
    return free / (1024**3)

def _gpu_total_gb():
    if not torch.cuda.is_available(): return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)

def _ram_avail_gb():
    if _HAS_PSUTIL: return psutil.virtual_memory().available / (1024**3)
    return 8.0

def cleanup_memory(verbose=False):
    for _ in range(3): gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    _malloc_trim()
    if verbose:
        print(f"  [mem] GPU free: {_gpu_free_gb():.2f} GB | RAM avail: {_ram_avail_gb():.2f} GB")

def get_value_at_index(obj, index):
    try: return obj[index]
    except KeyError: return obj["result"][index]

if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU detected. In Colab: Runtime → Change runtime type → T4 GPU")

print(f"PyTorch : {torch.__version__}")
print(f"GPU     : {torch.cuda.get_device_name(0)}")
print(f"VRAM    : {_gpu_total_gb():.1f} GB  ({_gpu_free_gb():.1f} GB free)")

# Resolve config
IMAGE_PATH = (IMAGE_PATH or "").strip() or None
AUDIO_PATH = (AUDIO_PATH or "").strip() or None
if RANDOM_SEED:
    SEED = _random.randint(0, 2**31 - 1)
    print(f"  Random seed: {SEED}")

QUALITY_PROFILES = {
    "t4_safe":     {"chunk_frames": 49, "width": 832, "height": 480, "longer_edge": 848, "img_compression": 33},
    "t4_balanced": {"chunk_frames": 73, "width": 832, "height": 480, "longer_edge": 848, "img_compression": 18},
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4  —  MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

MODELS = {
    "dit":            "ltx-2-3-22b-dev-Q4_K_M.gguf",
    "text_encoder_1": "gemma_3_12B_it_fp4_mixed.safetensors",
    "text_encoder_2": "ltx-2.3_text_projection_bf16.safetensors",
    "audio_vae":      "LTX23_audio_vae_bf16.safetensors",
    "video_vae":      "LTX23_video_vae_bf16.safetensors",
    "upscaler":       "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
}

DOWNLOAD_URLS = {
    "dit":            "https://huggingface.co/vantagewithai/LTX-2.3-GGUF/resolve/main/dev/ltx-2-3-22b-dev-Q4_K_M.gguf",
    "text_encoder_1": "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
    "text_encoder_2": "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors",
    "audio_vae":      "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_audio_vae_bf16.safetensors",
    "video_vae":      "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_video_vae_bf16.safetensors",
    "upscaler":       "https://huggingface.co/vidfom/aimusic/resolve/main/ComfyUI/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
}

MODEL_DIRS = {
    "dit":            f"{COMFYUI_DIR}/models/unet",
    "text_encoder_1": f"{COMFYUI_DIR}/models/text_encoders",
    "text_encoder_2": f"{COMFYUI_DIR}/models/text_encoders",
    "audio_vae":      f"{COMFYUI_DIR}/models/vae",
    "video_vae":      f"{COMFYUI_DIR}/models/vae",
    "upscaler":       f"{COMFYUI_DIR}/models/latent_upscale_models",
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5  —  ENVIRONMENT SETUP
# ─────────────────────────────────────────────────────────────────────────────

def _run(cmd, label):
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"  ✓ {label}")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ {label}: {e.stderr.strip()[:200]}")

def install_environment():
    print("="*60)
    print("[1/5] Python packages...")
    _run("pip install -q torch torchvision torchaudio", "torch")
    _run("pip install -q torchsde einops diffusers accelerate", "diffusers")
    _run("pip install -q av spandrel albumentations onnx opencv-python onnxruntime", "vision")
    _run("pip install -q psutil nest_asyncio moviepy", "utils")
    print("[2/5] aria2 + ffmpeg...")
    _run("apt-get -y install -qq aria2 ffmpeg", "apt packages")
    print("[3/5] ComfyUI...")
    if not os.path.exists(COMFYUI_DIR):
        _run(f"git clone -q https://github.com/comfyanonymous/ComfyUI {COMFYUI_DIR}", "ComfyUI")
    else:
        print("  ComfyUI already present.")
    _run(f"pip install -q -r {COMFYUI_DIR}/requirements.txt", "ComfyUI requirements")
    print("[4/5] Custom nodes...")
    nd = f"{COMFYUI_DIR}/custom_nodes"
    Path(nd).mkdir(exist_ok=True)
    NODES = [
        ("https://github.com/kijai/ComfyUI-KJNodes",               "ComfyUI-KJNodes"),
        ("https://github.com/city96/ComfyUI-GGUF",                 "ComfyUI-GGUF"),
        ("https://github.com/Lightricks/ComfyUI-LTXVideo",         "ComfyUI-LTXVideo"),
    ]
    for url, name in NODES:
        dest = os.path.join(nd, name)
        if not os.path.exists(dest): _run(f"git clone -q {url} {dest}", f"  {name}")
        else: print(f"  ✓ {name} (present)")
        req = os.path.join(dest, "requirements.txt")
        if os.path.exists(req): _run(f"pip install -q -r {req}", f"  req {name}")
    print("[5/5] Directories...")
    for d in [WORKSPACE_DIR, OUTPUT_DIR, f"{WORKSPACE_DIR}/chunks",
              f"{WORKSPACE_DIR}/scenes", f"{COMFYUI_DIR}/input"]:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✓ Environment ready.")

def model_download(url, dest_dir, filename=None):
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    if not filename: filename = url.split("/")[-1].split("?")[0]
    fp = os.path.join(dest_dir, filename)
    if os.path.exists(fp) and os.path.getsize(fp) > 0:
        print(f"  ✓ {filename} (cached)"); return filename
    print(f"  ↓ {filename}...", end=" ", flush=True)
    cmd = ["aria2c","--console-log-level=error","-c","-x","16","-s","16","-k","1M",
           "--summary-interval=0","--quiet","-d",dest_dir,"-o",filename,url]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("done"); return filename
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {e.stderr.strip()[:200]}"); return None

def download_all_models():
    print("\n  Downloading models...")
    for key in MODELS:
        model_download(DOWNLOAD_URLS[key], MODEL_DIRS[key], MODELS[key])

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6  —  COMFYUI SETUP WITH CORRECT PROMPTSERVER MOCK
#
# ROOT CAUSE of previous crashes:
#   KJNodes checks:  PromptServer.instance.app.router.frozen
#   VideoHelperSuite checks: PromptServer.instance.last_node_id
#   SamplerCustomAdvanced calls: PromptServer.instance.send_sync(...)
#
# We must mock ALL these attributes on a single shared instance so every
# custom node that checks them at import-time AND at runtime finds them.
# ─────────────────────────────────────────────────────────────────────────────

_NODES_LOADED = False

class _MockRouter:
    frozen = True
    def add_route(self, *a, **kw): pass

class _MockApp:
    router = _MockRouter()
    def add_routes(self, *a, **kw): pass

class _MockPromptServer:
    """
    Minimal PromptServer stand-in that satisfies every attribute access
    made by KJNodes, VideoHelperSuite, rgthree, and ComfyUI core nodes
    when running headless (no aiohttp web server).
    """
    instance = None

    def __init__(self):
        self.app            = _MockApp()
        self.loop           = asyncio.new_event_loop()
        self.messages       = asyncio.Queue()
        self.client_id      = None
        self.last_node_id   = None          # VideoHelperSuite latent_preview
        self.last_prompt_id = None
        self.queue          = None
        self.number         = 0
        self.node_paths     = {}
        self.node_replace_manager = None    # nodes_replacements.py

    # ComfyUI calls this during sampling for live previews — we just ignore it
    def send_sync(self, event, data, sid=None):
        pass

    def trigger_on_prompt(self, *a, **kw):
        pass

def setup_comfyui():
    """Add ComfyUI to sys.path and install the headless PromptServer mock."""
    if COMFYUI_DIR not in sys.path:
        sys.path.insert(0, COMFYUI_DIR)

    # Install mock BEFORE importing any ComfyUI module that checks it
    try:
        import server as _srv
        # Only mock if the real server has no real app (headless context)
        if not hasattr(getattr(_srv.PromptServer, 'instance', None), 'app'):
            mock = _MockPromptServer()
            _srv.PromptServer.instance = mock
            _MockPromptServer.instance = mock
    except Exception:
        pass

    print(f"  ComfyUI path: {COMFYUI_DIR}")

def import_custom_nodes():
    global _NODES_LOADED
    if _NODES_LOADED: return

    # nest_asyncio lets asyncio.run() work inside an already-running event loop
    import nest_asyncio
    nest_asyncio.apply()

    # Re-apply mock after server module import (some ComfyUI versions reset it)
    try:
        import server as _srv
        if not hasattr(getattr(_srv.PromptServer, 'instance', None), 'app'):
            mock = _MockPromptServer()
            _srv.PromptServer.instance = mock
    except Exception:
        pass

    # Patch kornia if needed (ComfyUI-LTXVideo compatibility)
    try:
        import kornia.geometry.transform.pyramid as _kp
        if not hasattr(_kp, 'pad'):
            import torch.nn.functional as F
            _kp.pad = F.pad
    except Exception:
        pass

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

def N(name):
    """Get an instantiated ComfyUI node by name."""
    from nodes import NODE_CLASS_MAPPINGS
    if name not in NODE_CLASS_MAPPINGS:
        raise KeyError(f"ComfyUI node '{name}' not found. Is the custom node installed?")
    return NODE_CLASS_MAPPINGS[name]()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7  —  LTX FRAME MATH
# ─────────────────────────────────────────────────────────────────────────────

def ltx_frames(n, fps=24):
    """Round n up to the nearest valid LTX frame count: 8k+1 (9,17,25,33...)"""
    if n < 9: n = 9
    if (n - 1) % 8 == 0: return n
    k = math.ceil((n - 1) / 8)
    adj = k * 8 + 1
    print(f"  LTX frame adjustment: {n} → {adj} ({adj/fps:.2f}s)")
    return adj

def plan_chunks(total_frames, chunk_size, fps):
    chunks, start, idx = [], 0, 0
    while start < total_frames:
        raw = min(chunk_size, total_frames - start)
        size = ltx_frames(raw, fps)
        if start + size > total_frames:
            size = total_frames - start
            if size < 9:
                if chunks: chunks[-1]["num_frames"] += size
                break
        chunks.append({"idx": idx, "start": start, "frames": size})
        start += size; idx += 1
    return chunks

def tensor_wh(img):
    if isinstance(img, (tuple, list)): img = get_value_at_index(img, 0)
    if img.ndim == 4: return int(img.shape[2]), int(img.shape[1])
    if img.ndim == 3: return int(img.shape[1]), int(img.shape[0])
    raise ValueError(f"Bad tensor shape: {img.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8  —  AUDIO VAE LOADER (version-resilient)
# ─────────────────────────────────────────────────────────────────────────────

def load_audio_vae():
    from nodes import NODE_CLASS_MAPPINGS
    nm = MODELS["audio_vae"]
    if "VAELoaderKJ" in NODE_CLASS_MAPPINGS:
        l = NODE_CLASS_MAPPINGS["VAELoaderKJ"]()
        return l.load_vae(vae_name=nm, device="main_device", weight_dtype="fp16")
    l = NODE_CLASS_MAPPINGS["VAELoader"]()
    return l.load_vae(vae_name=nm)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9  —  CORE GENERATION  (based on experiment_ltx23.py mainLTX)
#
# KEY DIFFERENCES from broken EXECUTE_NORMALIZED approach:
#   1. SamplerCustomAdvanced called as: sca.EXECUTE_NORMALIZED(...)
#      was failing because VideoHelperSuite overrides latent_preview.py
#      and reads server.last_node_id at runtime.
#      FIX: our _MockPromptServer.send_sync() is a no-op, and last_node_id=None.
#   2. Uses ManualSigmas instead of BasicScheduler (no server dependency).
#   3. Loads/deletes VAE within the same chunk to keep VRAM budget tight.
#   4. Calls comfy.model_management.unload_all_models() before VAE decode.
# ─────────────────────────────────────────────────────────────────────────────

def generate_one_chunk(
    image_path,    # str or None
    prompt,        # str
    seed,          # int
    width, height,
    fps,
    num_frames,    # must be 8k+1
    longer_edge,
    img_compression,
):
    """
    Generate one temporal chunk. Returns path to saved chunk MP4, or None.

    Pipeline (mirrors experiment_ltx23.py exactly):
      ResizeImageMaskNode → ResizeImagesByLongerEdge → LTXVPreprocess
      DualCLIPLoader → CLIPTextEncode → ConditioningZeroOut → LTXVConditioning
      VAELoader (video) → LTXVImgToVideoInplace → del video_vae
      VAELoaderKJ (audio) → LTXVEmptyLatentAudio → LTXVConcatAVLatent
      UnetLoaderGGUF → CFGGuider
      SamplerCustomAdvanced (ManualSigmas, euler, 8 steps, denoise=1.0)
      del unet
      comfy.model_management.unload_all_models()   ← CRITICAL
      LTXVSeparateAVLatent → LTXVCropGuides → CFGGuider
      VAELoader → LatentUpscaleModelLoader → LTXVLatentUpsampler
      LTXVImgToVideoInplace → del video_vae
      LTXVConcatAVLatent
      SamplerCustomAdvanced (ManualSigmas, gradient_estimation, 4 steps, denoise=0.42)
      del unet
      comfy.model_management.unload_all_models()   ← CRITICAL
      LTXVSeparateAVLatent → VAELoader → VAEDecode → del video_vae
      LTXVAudioVAEDecode → del audio_vae
      CreateVideo → save
    """
    print(f"  [chunk] {num_frames} frames, {width}×{height}, seed={seed}")
    cleanup_memory()

    with torch.inference_mode():
        # ── Image preprocessing ───────────────────────────────────────────────
        resize_node  = N("ResizeImageMaskNode")
        resize_edge  = N("ResizeImagesByLongerEdge")
        preproc_node = N("LTXVPreprocess")

        if image_path and os.path.exists(image_path):
            img_load   = N("LoadImage").load_image(image=os.path.basename(image_path))
            img_tensor = get_value_at_index(img_load, 0)
            img_strength = 1.0; img_bypass = False
        else:
            img_tensor   = torch.full((1, height, width, 3), 0.5)
            img_strength = 0.0; img_bypass = True

        resized = resize_node.EXECUTE_NORMALIZED(
            input=img_tensor, scale_method="lanczos",
            resize_type={"resize_type":"scale dimensions","width":width,"height":height,"crop":"center"})
        rlong   = resize_edge.EXECUTE_NORMALIZED(
            longer_edge=longer_edge, images=get_value_at_index(resized, 0))
        preproc = preproc_node.EXECUTE_NORMALIZED(
            img_compression=img_compression, image=get_value_at_index(rlong, 0))

        rw, rh   = tensor_wh(get_value_at_index(resized, 0))
        latent_w = max(1, rw // 2)
        latent_h = max(1, rh // 2)
        del resized, rlong

        # ── Empty video latent ────────────────────────────────────────────────
        empty_vid = N("EmptyLTXVLatentVideo").EXECUTE_NORMALIZED(
            width=latent_w, height=latent_h, length=num_frames, batch_size=1)

        # ── Text conditioning ─────────────────────────────────────────────────
        print("  Loading text encoder...")
        dclip = N("DualCLIPLoader")
        try:
            clip_r = dclip.load_clip(
                clip_name1=MODELS["text_encoder_1"],
                clip_name2=MODELS["text_encoder_2"],
                type="ltxv", device="default")
        except Exception as e:
            print(f"  CLIP fp4 failed ({e}), trying fp8...")
            clip_r = dclip.load_clip(
                clip_name1="gemma_3_12B_it_fp8_scaled.safetensors",
                clip_name2="ltx-2.3-22b-dev_embeddings_connectors.safetensors",
                type="ltxv", device="default")

        clip_obj  = get_value_at_index(clip_r, 0)
        cte       = N("CLIPTextEncode")
        pos_enc   = cte.encode(text=prompt, clip=clip_obj)
        czo       = N("ConditioningZeroOut")
        neg_enc   = czo.zero_out(conditioning=get_value_at_index(pos_enc, 0))
        del clip_r, clip_obj; cleanup_memory()

        ltxcond   = N("LTXVConditioning")
        cond      = ltxcond.EXECUTE_NORMALIZED(
            frame_rate=fps,
            positive=get_value_at_index(pos_enc, 0),
            negative=get_value_at_index(neg_enc, 0))
        cond_pos  = get_value_at_index(cond, 0)
        cond_neg  = get_value_at_index(cond, 1)

        # ── Video VAE → image conditioning ───────────────────────────────────
        vae_load = N("VAELoader")
        vae1     = vae_load.load_vae(vae_name=MODELS["video_vae"])
        i2v      = N("LTXVImgToVideoInplace")
        img_cond = i2v.EXECUTE_NORMALIZED(
            strength=img_strength, bypass=img_bypass,
            vae=get_value_at_index(vae1, 0),
            image=get_value_at_index(preproc, 0),
            latent=get_value_at_index(empty_vid, 0))
        del vae1; cleanup_memory()

        # ── Audio VAE → empty audio latent ───────────────────────────────────
        audio_vae = load_audio_vae()
        aud_lat   = N("LTXVEmptyLatentAudio").EXECUTE_NORMALIZED(
            frames_number=num_frames, frame_rate=fps, batch_size=1,
            audio_vae=get_value_at_index(audio_vae, 0))

        # ── Concat AV (pass 1 input) ──────────────────────────────────────────
        catav     = N("LTXVConcatAVLatent")
        if not img_bypass:
            av1 = catav.EXECUTE_NORMALIZED(
                video_latent=get_value_at_index(img_cond, 0),
                audio_latent=get_value_at_index(aud_lat, 0))
        else:
            av1 = catav.EXECUTE_NORMALIZED(
                video_latent=get_value_at_index(empty_vid, 0),
                audio_latent=get_value_at_index(aud_lat, 0))

        # ── Load UNet (no LoRAs — T4 safe) ────────────────────────────────────
        print("  Loading DiT (UnetLoaderGGUF)...")
        cleanup_memory()
        unet_gg   = N("UnetLoaderGGUF")
        unet      = get_value_at_index(unet_gg.load_unet(unet_name=MODELS["dit"]), 0)
        print(f"  DiT loaded. VRAM free: {_gpu_free_gb():.2f} GB")

        # ── Pass 1 sampling ───────────────────────────────────────────────────
        ksel1 = N("KSamplerSelect").EXECUTE_NORMALIZED(sampler_name="euler")
        # ManualSigmas avoids BasicScheduler's PromptServer dependency
        sig1  = N("ManualSigmas").EXECUTE_NORMALIZED(
            sigmas="1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0")
        rn1   = N("RandomNoise").EXECUTE_NORMALIZED(noise_seed=seed)
        cfg1  = N("CFGGuider").EXECUTE_NORMALIZED(
            cfg=1, model=unet, positive=cond_pos, negative=cond_neg)
        sca   = N("SamplerCustomAdvanced")
        print("  Pass 1 sampling (8 steps, denoise=1.0)...")
        out1  = sca.EXECUTE_NORMALIZED(
            noise=get_value_at_index(rn1, 0),
            guider=get_value_at_index(cfg1, 0),
            sampler=get_value_at_index(ksel1, 0),
            sigmas=get_value_at_index(sig1, 0),
            latent_image=get_value_at_index(av1, 0))
        del cfg1; gc.collect()

        # ── Separate AV + crop guides ─────────────────────────────────────────
        sep   = N("LTXVSeparateAVLatent")
        s1    = sep.EXECUTE_NORMALIZED(av_latent=get_value_at_index(out1, 0))
        crop  = N("LTXVCropGuides")
        cr    = crop.EXECUTE_NORMALIZED(
            positive=cond_pos, negative=cond_neg,
            latent=get_value_at_index(s1, 0))
        cfg2  = N("CFGGuider").EXECUTE_NORMALIZED(
            cfg=1, model=unet,
            positive=get_value_at_index(cr, 0),
            negative=get_value_at_index(cr, 1))

        # ── Latent upscale ────────────────────────────────────────────────────
        vae2  = vae_load.load_vae(vae_name=MODELS["video_vae"])
        uml   = N("LatentUpscaleModelLoader")
        um    = uml.EXECUTE_NORMALIZED(model_name=MODELS["upscaler"])
        lup   = N("LTXVLatentUpsampler")
        upsampled = lup.upsample_latent(
            samples=get_value_at_index(cr, 2),
            upscale_model=get_value_at_index(um, 0),
            vae=get_value_at_index(vae2, 0))
        del um; cleanup_memory()

        # ── Re-condition on upscaled latent ───────────────────────────────────
        iv2   = i2v.EXECUTE_NORMALIZED(
            strength=img_strength, bypass=img_bypass,
            vae=get_value_at_index(vae2, 0),
            image=get_value_at_index(preproc, 0),
            latent=get_value_at_index(upsampled, 0))
        del vae2; cleanup_memory()

        if not img_bypass:
            av2 = catav.EXECUTE_NORMALIZED(
                video_latent=get_value_at_index(iv2, 0),
                audio_latent=get_value_at_index(s1, 1))
        else:
            av2 = catav.EXECUTE_NORMALIZED(
                video_latent=get_value_at_index(upsampled, 0),
                audio_latent=get_value_at_index(s1, 1))

        # ── Pass 2 sampling ───────────────────────────────────────────────────
        ksel2 = N("KSamplerSelect").EXECUTE_NORMALIZED(sampler_name="gradient_estimation")
        sig2  = N("ManualSigmas").EXECUTE_NORMALIZED(
            sigmas="0.909375, 0.725, 0.421875, 0.0")
        rn2   = N("RandomNoise").EXECUTE_NORMALIZED(noise_seed=0)
        print("  Pass 2 sampling (4 steps, denoise=0.42)...")
        out2  = sca.EXECUTE_NORMALIZED(
            noise=get_value_at_index(rn2, 0),
            guider=get_value_at_index(cfg2, 0),
            sampler=get_value_at_index(ksel2, 0),
            sigmas=get_value_at_index(sig2, 0),
            latent_image=get_value_at_index(av2, 0))
        del cfg2, unet; gc.collect()

        # ── CRITICAL: unload ALL models from ComfyUI internal cache ──────────
        # Without this, 14+ GB stays allocated after sampling and VAE decode OOMs.
        print(f"  VRAM before unload: {_gpu_free_gb():.2f} GB free")
        try:
            import comfy.model_management as mm
            mm.unload_all_models()
        except Exception as e:
            print(f"  comfy.model_management unavailable: {e}")
        cleanup_memory()
        print(f"  VRAM after unload: {_gpu_free_gb():.2f} GB free")

        # ── Decode video ──────────────────────────────────────────────────────
        s2    = sep.EXECUTE_NORMALIZED(av_latent=get_value_at_index(out2, 1))
        vae3  = vae_load.load_vae(vae_name=MODELS["video_vae"])
        vd    = N("VAEDecode")
        print("  VAE decoding video...")
        vid_dec = vd.decode(
            samples=get_value_at_index(s2, 0),
            vae=get_value_at_index(vae3, 0))
        del vae3; cleanup_memory()

        # ── Decode audio ──────────────────────────────────────────────────────
        aud_dec = N("LTXVAudioVAEDecode").EXECUTE_NORMALIZED(
            samples=get_value_at_index(s2, 1),
            audio_vae=get_value_at_index(audio_vae, 0))
        del audio_vae; cleanup_memory()

        # ── Save video ────────────────────────────────────────────────────────
        print("  Creating video...")
        cv    = N("CreateVideo")
        vobj  = cv.EXECUTE_NORMALIZED(
            fps=fps,
            images=get_value_at_index(vid_dec, 0),
            audio=get_value_at_index(aud_dec, 0))

        video  = get_value_at_index(vobj, 0)
        import folder_paths
        from comfy_api.latest import Types
        w_out, h_out = video.get_dimensions()
        folder, fname, ctr, _, _ = folder_paths.get_save_image_path(
            "ltx23_chunk", folder_paths.get_output_directory(), w_out, h_out)
        ext  = Types.VideoContainer.get_extension("auto")
        path = os.path.join(folder, f"{fname}_{ctr:05d}_.{ext}")
        video.save_to(path, format=Types.VideoContainer("auto"), codec="auto", metadata=None)
        del vobj, video; cleanup_memory()
        print(f"  ✓ Chunk saved: {path}")
        return path

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10  —  SCENE CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────

def _cp_path(name): return os.path.join(WORKSPACE_DIR, f"{name}_checkpoint.json")

def load_checkpoint(name):
    p = _cp_path(name)
    if not os.path.exists(p): return None
    try:
        with open(p) as f: return json.load(f)
    except Exception: return None

def save_checkpoint(cp, name):
    cp["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp = _cp_path(name) + ".tmp"
    with open(tmp, "w") as f: json.dump(cp, f, indent=2)
    os.replace(tmp, _cp_path(name))

def init_checkpoint(name, num_scenes):
    cp = {
        "project":         name,
        "num_scenes":      num_scenes,
        "completed":       [],    # list of scene indices
        "clip_paths":      {},    # {str(i): path}
        "anchor_paths":    {},    # {str(i): anchor_path}
        "created_at":      time.strftime("%Y-%m-%dT%H:%M:%S"),
        "updated_at":      time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    save_checkpoint(cp, name); return cp

def get_checkpoint(name, num_scenes):
    if RESUME:
        cp = load_checkpoint(name)
        if cp and cp.get("num_scenes") == num_scenes:
            done = len(cp.get("completed", []))
            print(f"  ↷ Resuming: {done}/{num_scenes} scenes done.")
            return cp
        elif cp:
            print("  ⚠ Checkpoint config mismatch — starting fresh.")
    return init_checkpoint(name, num_scenes)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11  —  SCENE PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_shot_prompt(shot, scene_json, idx):
    if shot.get("prompt_override", "").strip():
        return shot["prompt_override"].strip()
    style   = scene_json.get("video_style", "cinematic")
    env     = scene_json.get("environment", {})
    chars   = scene_json.get("main_characters", [])
    char_desc = " | ".join(
        f"{c['name']}: {c['detailed_appearance'].get('face','')} "
        f"wearing {c['detailed_appearance'].get('clothing','')}"
        for c in chars)
    env_desc  = (f"SETTING: {env.get('location','stage')}. "
                 f"MOOD: {env.get('mood','energetic')}. "
                 f"LIGHTING: {env.get('lighting','concert lights')}.")
    return (
        f"CHARACTER: {char_desc}. "
        f"SHOT {idx+1}: {shot.get('action','')}. "
        f"CAMERA: {shot.get('camera','')}. "
        f"EMOTION: {shot.get('emotion','neutral')}. "
        f"{env_desc} "
        f"STYLE: {style}. "
        f"Photorealistic, ultra-high fidelity, maintain exact character appearance."
    )

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12  —  ANCHOR EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_anchor(video_path, scene_idx, overlap=8):
    """Extract best anchor frame from end of clip for next-scene continuity."""
    if not video_path or not os.path.exists(video_path): return None
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0: cap.release(); return None
    target = max(0, total - max(2, overlap))
    best_frame, best_score = None, -1.0
    for fi in range(max(0, target-3), min(total, target+4)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok: continue
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bright = float(cv2.mean(gray)[0])
        sharp  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        score  = bright * 0.3 + sharp * 0.1
        if score > best_score: best_score = score; best_frame = frame
    cap.release()
    if best_frame is None: return None
    out = os.path.join(COMFYUI_DIR, "input", f"anchor_{scene_idx:04d}.png")
    cv2.imwrite(out, best_frame)
    print(f"  ✓ Anchor extracted: {out}")
    return out

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13  —  SCENE ASSEMBLER  (chunks → scene MP4)
# ─────────────────────────────────────────────────────────────────────────────

def concat_chunks(chunk_paths, output_path, fps):
    if not chunk_paths: return False
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    lst = output_path + "_list.txt"
    with open(lst, "w") as f:
        for p in sorted(chunk_paths):
            safe_p = p.replace("'", "'\\''")
            f.write(f"file '{safe_p}'\n")
    r = subprocess.run(
        ["ffmpeg","-y","-f","concat","-safe","0","-i",lst,"-c","copy",output_path],
        capture_output=True)
    if os.path.exists(lst): os.remove(lst)
    if r.returncode == 0:
        print(f"  ✓ Scene assembled: {output_path}"); return True
    # fallback re-encode
    r2 = subprocess.run(
        ["ffmpeg","-y","-f","concat","-safe","0","-i",lst+"_",
         "-vcodec","libx264","-pix_fmt","yuv420p","-crf","18","-preset","fast",output_path],
        capture_output=True)
    return r2.returncode == 0

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14  —  FINAL ASSEMBLY  (scenes → movie + audio)
# ─────────────────────────────────────────────────────────────────────────────

def assemble_final(scene_clips, output_path, fps, audio_path=None, overlap_frames=8):
    if not scene_clips: return None
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    no_audio = output_path.replace(".mp4","_noaudio.mp4")

    # Trim overlap from all-but-last clips then concat
    trimmed = []
    for i, clip in enumerate(scene_clips):
        if not os.path.exists(clip): continue
        if i < len(scene_clips) - 1 and overlap_frames > 0:
            try:
                pr = subprocess.run(
                    ["ffprobe","-v","quiet","-print_format","json","-show_format",clip],
                    capture_output=True, text=True)
                dur = float(json.loads(pr.stdout).get("format",{}).get("duration",0))
                trim_dur = dur - overlap_frames / fps
                if trim_dur > 0:
                    tp = clip.replace(".mp4",f"_t{i}.mp4")
                    subprocess.run(["ffmpeg","-y","-i",clip,"-t",str(trim_dur),"-c","copy",tp],
                                   capture_output=True)
                    if os.path.exists(tp): trimmed.append(tp); continue
            except Exception: pass
        trimmed.append(clip)

    if not concat_chunks(trimmed, no_audio, fps):
        return None

    # Clean up temp trim files
    for p in trimmed:
        if "_t" in p and os.path.exists(p):
            try: os.remove(p)
            except Exception: pass

    if audio_path and os.path.exists(audio_path):
        cmd = ["ffmpeg","-y","-i",no_audio,"-i",audio_path,
               "-map","0:v:0","-map","1:a:0","-c:v","copy","-c:a","aac","-b:a","192k",
               "-shortest",output_path]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode == 0:
            if os.path.exists(no_audio): os.remove(no_audio)
            return output_path
    # No audio or mux failed → use the no-audio version
    if output_path != no_audio: shutil.move(no_audio, output_path)
    return output_path

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15  —  SRT SUBTITLE EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_srt(dialogue_list, srt_path):
    def _t(s):
        ms = int((s - int(s))*1000)
        h,m,sc = int(s//3600), int((s%3600)//60), int(s%60)
        return f"{h:02}:{m:02}:{sc:02},{ms:03}"
    try:
        with open(srt_path,"w",encoding="utf-8") as f:
            for i,e in enumerate(dialogue_list,1):
                t0 = float(e.get("time",0)); t1 = t0 + float(e.get("duration",4))
                f.write(f"{i}\n{_t(t0)} --> {_t(t1)}\n[{e.get('character','?')}] {e.get('dialogue','')}\n\n")
        print(f"  ✓ SRT: {srt_path}")
    except Exception as e: print(f"  ⚠ SRT failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 16  —  MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_multiscene_mv(
    scene_json   = None,
    image_path   = None,
    audio_path   = None,
    quality_mode = None,
    seed         = None,
    fps          = None,
):
    """
    Full multi-scene generation pipeline.

    For each shot in SCENE_JSON.story_action.shots:
      1. Build shot-specific prompt
      2. Set input image (reference or previous-scene anchor)
      3. Generate video in chunks via generate_one_chunk()
      4. Concat chunks → scene MP4
      5. Extract anchor frame for next scene
      6. Save to checkpoint (auto-resume on crash)
    Then: stitch all scenes → final movie → mux audio → export SRT.

    Re-run this function after ANY crash — completed scenes are skipped.
    """
    sj   = scene_json   or SCENE_JSON
    img  = image_path   or IMAGE_PATH
    aud  = audio_path   or AUDIO_PATH
    qm   = quality_mode or QUALITY_MODE
    s    = seed         if seed is not None else SEED
    fps  = fps          or FPS

    prof     = QUALITY_PROFILES.get(qm, QUALITY_PROFILES["t4_safe"])
    w, h     = prof["width"], prof["height"]
    chunk_sz = ltx_frames(prof["chunk_frames"], fps)

    print("\n" + "="*60)
    print("LTX-2.3 DIRECTOR 2.0  ·  MULTI-SCENE MV ENGINE  v4.0")
    print("="*60)
    print(f"  GPU       : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM      : {_gpu_total_gb():.1f} GB  ({_gpu_free_gb():.1f} GB free)")
    print(f"  Quality   : {qm}  →  {w}×{h} @ {fps}fps")
    print(f"  Seed      : {s}")
    print(f"  Resume    : {RESUME}")
    print("="*60)

    shots = sj["story_action"]["shots"]
    n     = len(shots)
    print(f"  {n} shots in storyboard")

    # ComfyUI init
    setup_comfyui()
    import_custom_nodes()

    # Checkpoint
    cp   = get_checkpoint(PROJECT_NAME, n)
    Path(WORKSPACE_DIR).mkdir(exist_ok=True)
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    Path(f"{WORKSPACE_DIR}/chunks").mkdir(exist_ok=True)
    Path(f"{WORKSPACE_DIR}/scenes").mkdir(exist_ok=True)

    scene_clips     = []
    current_anchor  = img   # updated after each scene

    # Restore from checkpoint
    start_idx = 0
    for i in range(n):
        cp_clip   = cp["clip_paths"].get(str(i))
        cp_anchor = cp["anchor_paths"].get(str(i))
        if i in cp["completed"] and cp_clip and os.path.exists(cp_clip):
            scene_clips.append(cp_clip)
            if cp_anchor and os.path.exists(cp_anchor):
                current_anchor = cp_anchor
            start_idx = i + 1
        else:
            break

    if start_idx > 0:
        print(f"  ✓ Restored {start_idx} completed scenes from checkpoint.")

    generation_start = time.time()

    for i in range(start_idx, n):
        shot   = shots[i]
        prompt = build_shot_prompt(shot, sj, i)
        print(f"\n{'='*60}")
        print(f"  SCENE {i+1:02d}/{n:02d}  |  {shot.get('emotion','')}")
        print(f"  {prompt[:100]}...")
        print("="*60)

        # Duration from shot time field (e.g. "4-8s")
        try:
            t_parts  = shot["time"].replace("s","").split("-")
            shot_dur = float(t_parts[1]) - float(t_parts[0])
        except Exception:
            shot_dur = SCENE_DURATION_S
        total_frames = ltx_frames(round(shot_dur * fps), fps)
        chunks       = plan_chunks(total_frames, chunk_sz, fps)
        print(f"  {total_frames} frames ({shot_dur:.1f}s), {len(chunks)} chunk(s)")

        success    = False
        scene_clip = None

        for attempt in range(1, MAX_SCENE_RETRIES + 1):
            chunk_seed = (s + i * 9973 + attempt * 197) & 0x7FFFFFFF
            print(f"\n  Attempt {attempt}/{MAX_SCENE_RETRIES}  seed={chunk_seed}")

            chunk_paths = []
            try:
                for ci, ck in enumerate(chunks):
                    print(f"  [chunk {ci+1}/{len(chunks)}] {ck['frames']} frames")
                    cpath = generate_one_chunk(
                        image_path    = current_anchor,
                        prompt        = prompt,
                        seed          = chunk_seed + ci * 1000,
                        width         = w,
                        height        = h,
                        fps           = fps,
                        num_frames    = ck["frames"],
                        longer_edge   = prof["longer_edge"],
                        img_compression = prof["img_compression"],
                    )
                    if cpath: chunk_paths.append(cpath)

                if not chunk_paths:
                    print("  ⚠ No chunks generated."); continue

                # Concat chunks → scene MP4
                scene_out = os.path.join(WORKSPACE_DIR, "scenes", f"scene_{i:04d}.mp4")
                if len(chunk_paths) == 1:
                    shutil.copy(chunk_paths[0], scene_out)
                else:
                    concat_chunks(chunk_paths, scene_out, fps)

                if os.path.exists(scene_out):
                    scene_clips.append(scene_out)
                    anchor = extract_anchor(scene_out, i)
                    if anchor: current_anchor = anchor
                    cp["completed"].append(i)
                    cp["clip_paths"][str(i)]   = scene_out
                    cp["anchor_paths"][str(i)] = anchor or ""
                    save_checkpoint(cp, PROJECT_NAME)
                    success = True
                    print(f"  ✓ Scene {i+1} complete! ({len(scene_clips)}/{n})")
                    # Clean up chunk files
                    for cp_ in chunk_paths:
                        if cp_ != scene_out:
                            try: os.remove(cp_)
                            except Exception: pass
                    break

            except torch.cuda.OutOfMemoryError as oom:
                print(f"  OOM on scene {i+1} attempt {attempt}: {oom}")
                cleanup_memory(verbose=True)
                try:
                    import comfy.model_management as mm
                    mm.unload_all_models()
                except Exception: pass
                cleanup_memory()
            except Exception as e:
                print(f"  ERROR scene {i+1} attempt {attempt}: {type(e).__name__}: {str(e)[:200]}")
                traceback.print_exc()
                cleanup_memory()

        if not success:
            print(f"  ⚠ Scene {i+1} failed after {MAX_SCENE_RETRIES} attempts — skipping.")

    # ── Final assembly ─────────────────────────────────────────────────────────
    if not scene_clips:
        print("\n✗ No scenes generated."); return None

    print(f"\n{'='*60}")
    print(f"  Assembling {len(scene_clips)}/{n} scenes...")

    final_out = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    result    = assemble_final(scene_clips, final_out, fps, aud, overlap_frames=8)

    # SRT
    dlg = sj.get("dialogue_with_timing", [])
    if dlg:
        export_srt(dlg, final_out.replace(".mp4",".srt"))

    elapsed = time.time() - generation_start
    print(f"\n{'='*60}")
    if result:
        sz = os.path.getsize(result)/(1024*1024) if os.path.exists(result) else 0
        print(f"  ✓ GENERATION COMPLETE")
        print(f"  Output  : {result}  ({sz:.1f} MB)")
        print(f"  Scenes  : {len(scene_clips)}/{n}")
        print(f"  Time    : {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    else:
        print("  ✗ Final assembly failed — individual scenes in:", WORKSPACE_DIR+"/scenes/")
    print("="*60)

    cleanup_memory()
    return result

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 17  —  COLAB DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def display_video(path, max_mb=50):
    from IPython.display import display, HTML
    from base64 import b64encode
    if not path or not os.path.exists(path): return
    mb = os.path.getsize(path)/(1024*1024)
    if mb > max_mb:
        print(f"  Video {mb:.0f} MB — download with: files.download('{path}')")
        return
    data = b64encode(open(path,"rb").read()).decode()
    display(HTML(f'<video width=800 controls autoplay loop muted>'
                 f'<source src="data:video/mp4;base64,{data}" type="video/mp4"></video>'))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 18  —  DIRECT EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nLTX-2.3 Director 2.0 Multi-Scene MV Engine  v4.0\n")

    # Step 1: Install packages and custom nodes (run once per runtime)
    install_environment()

    # Step 2: Download all model weights (run once per runtime)
    download_all_models()

    # Step 3: Generate
    # Edit SCENE_JSON above to customise your shots.
    # Upload reference image to IMAGE_PATH before running.
    output = generate_multiscene_mv(
        image_path   = IMAGE_PATH,
        audio_path   = AUDIO_PATH,
        quality_mode = QUALITY_MODE,
        seed         = SEED,
        fps          = FPS,
    )

    if output:
        print(f"\n✓ Done: {output}")
        try:
            from google.colab import files
            files.download(output)
        except Exception:
            pass
    else:
        print("\n✗ Generation failed — see errors above.")

# ─────────────────────────────────────────────────────────────────────────────
# COLAB QUICK-START (paste each into a separate cell):
#
# CELL 1  install_environment()
# CELL 2  download_all_models()
# CELL 3  # upload reference image + audio (optional)
#         from google.colab import files, shutil, os
#         up = files.upload()
#         for fn in up: shutil.move(fn, f"{COMFYUI_DIR}/input/{fn}")
# CELL 4  output = generate_multiscene_mv()
#         if output: display_video(output)
# ─────────────────────────────────────────────────────────────────────────────
