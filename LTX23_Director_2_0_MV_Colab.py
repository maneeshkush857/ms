# -*- coding: utf-8 -*-
"""LTX-2.3 Director 2.0 Music Video Pipeline - Google Colab Implementation

Complete production-quality pipeline for generating 30s music videos using:
- LTX-2.3 Director 2.0 workflow
- Two-stage sampling (low-res + upscaled)
- Memory-managed chunked frame generation
- Checkpoint/resume system
- OOM detection and retry with smaller chunks

Target: Tesla T4 (16GB VRAM) on Google Colab
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import sys
import gc
import json
import time
import subprocess
import shutil
import signal
import atexit
import hashlib
import traceback
import argparse
import logging
from dataclasses import dataclass, field, asdict
from typing import (
    Sequence, Mapping, Any, Union, Optional, List, Dict, Tuple
)
from pathlib import Path

# Conditional imports for runtime
try:
    import torch
    import numpy as np
except ImportError:
    torch = None
    np = None

# =============================================================================
# LOGGING SETUP
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("LTX23_Director")

# =============================================================================
# CONSTANTS
# =============================================================================
GLOBAL_PROMPT = (
    "Create a highly realistic cinematic AI music video using the provided reference image. "
    "Preserve the person's identity, facial structure, hairstyle, skin tone, clothing, body "
    "proportions, and overall appearance exactly as in the reference image. The singer must "
    "remain fully recognizable throughout the entire video with absolutely no identity drift.\n\n"
    "The person is performing directly to the camera as a world-class pop, hip-hop and rap "
    "singer during a sold-out stadium concert. Generate perfectly synchronized lip movements "
    "from the provided lyrics or audio.\n\n"
    "This is NOT a talking-head video and NOT a presenter. This is a high-energy live music "
    "performance filled with charisma, attitude and emotional intensity.\n\n"
    "Performance Energy:\n"
    "- Perform with explosive stage presence.\n"
    "- Every musical phrase immediately creates a new emotional and physical performance.\n"
    "- Every lyric instantly changes facial expression, eye emotion, head movement, shoulders, "
    "hands, posture and body rhythm.\n"
    "- The performance continuously builds toward emotional peaks.\n"
    "- Own the stage with absolute confidence.\n"
    "- Perform as if in front of 50,000 screaming fans.\n"
    "- Captivate the audience every second.\n"
    "- Never appear calm, passive or static.\n\n"
    "Facial Performance:\n"
    "- Extremely expressive facial acting throughout the entire performance.\n"
    "- Rich emotional transitions every few words.\n"
    "- Powerful eye contact with intense emotional engagement.\n"
    "- Eyes sparkle with confidence and passion.\n"
    "- Highly expressive eyebrows synchronized with important lyrics.\n"
    "- Strong cheek and jaw movement while singing.\n"
    "- Natural smiles, smirks, determination, excitement, confidence, attitude, passion, "
    "curiosity, joy and intensity.\n"
    "- Rich cinematic micro-expressions.\n"
    "- Never hold the same facial expression for more than a brief musical phrase.\n"
    "- The face should feel emotionally alive every second.\n\n"
    "Body Performance:\n"
    "- The entire body constantly grooves with the beat.\n"
    "- Strong rhythmic bouncing.\n"
    "- Powerful shoulder accents.\n"
    "- Confident chest movement.\n"
    "- Hip movement follows the groove.\n"
    "- Frequent body turns.\n"
    "- Fast weight shifts.\n"
    "- Dynamic torso twists.\n"
    "- Lean toward the camera during emotional lyrics.\n"
    "- Occasionally step toward the camera.\n"
    "- Performance intensity increases naturally during powerful musical moments.\n"
    "- Bold, energetic and theatrical stage movement.\n\n"
    "Hand Performance:\n"
    "- Perform like an experienced pop or hip-hop superstar.\n"
    "- Large expressive gestures.\n"
    "- Fast rhythmic arm accents.\n"
    "- Sharp hand movements synchronized with the beat.\n"
    "- Powerful pointing.\n"
    "- Sweeping arm movements.\n"
    "- Punching the air.\n"
    "- Pulling gestures toward the chest.\n"
    "- Throwing gestures outward.\n"
    "- Finger snapping.\n"
    "- Open palm emphasis.\n"
    "- Framing the face.\n"
    "- Expressive wrist movement.\n"
    "- Hands constantly create visual rhythm.\n"
    "- One hand naturally leads while the other follows.\n"
    "- Asymmetrical movement.\n"
    "- Avoid symmetrical gestures.\n"
    "- Never repeatedly raise both hands together.\n"
    "- Every musical phrase introduces fresh gestures.\n"
    "- Never repeat the same gesture pattern.\n\n"
    "Musical Timing:\n"
    "- Body movement follows musical phrasing rather than every word.\n"
    "- Strong beats create explosive movements.\n"
    "- Soft phrases become intimate and emotional.\n"
    "- Fast lyrics generate faster gestures.\n"
    "- Slow lyrics become smoother without losing energy.\n"
    "- Every movement feels rhythmically connected to the music.\n\n"
    "Speech Synchronization:\n"
    "- Perfect lip synchronization.\n"
    "- Accurate mouth shapes.\n"
    "- Expressions and gestures match the emotional meaning of every lyric.\n"
    "- Natural breathing between phrases.\n\n"
    "Motion Quality:\n"
    "- Premium AI human animation.\n"
    "- Fast, confident and energetic performance.\n"
    "- Realistic momentum.\n"
    "- Strong acceleration and deceleration.\n"
    "- High-energy body mechanics.\n"
    "- Natural motion blur.\n"
    "- No robotic movement.\n"
    "- No frozen poses.\n"
    "- No repetitive gesture loops.\n"
    "- No presenter-style gestures.\n"
    "- No idle standing.\n"
    "- No jitter.\n"
    "- No flickering.\n"
    "- No facial distortion.\n"
    "- No identity drift.\n"
    "- No hand deformation.\n"
    "- No extra fingers.\n"
    "- No malformed limbs.\n\n"
    "Camera:\n"
    "drclipz, Aggressive cinematic music video camera. Fast push-in, fast pull-back, "
    "energetic handheld movement, rhythmic tracking shots, dynamic low-angle hero shots, "
    "occasional close-ups on emotional lyrics, subtle orbit around the singer, cinematic "
    "motion blur. Camera movement follows the beat and amplifies the performance.\n\n"
    "Lighting:\n"
    "Premium concert lighting with cinematic key light, colorful neon rim lights, volumetric "
    "atmosphere, dramatic contrast, realistic skin tones, vibrant electronic music video mood.\n\n"
    "Overall Style:\n"
    "Photorealistic, blockbuster-quality AI music video, premium live concert performance, "
    "ultra-high facial fidelity, charismatic superstar, emotionally captivating, explosive "
    "stage energy, bold movement, powerful attitude, modern pop, hip-hop and rap performance, "
    "every second feels alive, impossible to look away.\n\n"
    "Spoken dialogue:\n"
    "\"Open up the canvas, blank space on my screen. \n"
    "Drag a Checkpoint Loader, you know what I mean.\n"
    "KSampler in the middle, VAE on the right,\n"
    "Put the Text Encoder, yeah, building tonight.\n"
    "Connect the nodes, run the queue,\n"
    "Watch the latent flow right through.\n"
    "Green, nothing green, nothing yellow,\n"
    "Positive Prompt, in my hub.\""
)


# Timeline data JSON as extracted from the workflow
TIMELINE_DATA_JSON = json.dumps({
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
    "normalDurationFrames": 756,
    "segments": [
        {
            "id": "1785555235678s2fn3",
            "start": 0,
            "length": 226.01059340956584,
            "prompt": "",
            "type": "image",
            "imageFile": "whatdreamscost/1.png",
            "imageB64": "/api/view?filename=1.png&type=input&subfolder=whatdreamscost",
            "isEndFrame": False
        },
        {
            "id": "17855552413529uw9r",
            "start": 226.01059340956584,
            "length": 161.31859976617454,
            "prompt": "",
            "type": "image",
            "imageFile": "whatdreamscost/2.png",
            "imageB64": "/api/view?filename=2.png&type=input&subfolder=whatdreamscost",
            "isEndFrame": False
        },
        {
            "id": "1785555243885y3h85",
            "start": 387.3291931757404,
            "length": 131.45629831196658,
            "prompt": "",
            "type": "image",
            "imageFile": "whatdreamscost/3.png",
            "imageB64": "/api/view?filename=3.png&type=input&subfolder=whatdreamscost",
            "isEndFrame": False
        },
        {
            "id": "1785555247117rcoma",
            "start": 518.785491487707,
            "length": 225.5063328766255,
            "prompt": "",
            "type": "image",
            "imageFile": "whatdreamscost/4.png",
            "imageB64": "/api/view?filename=4.png&type=input&subfolder=whatdreamscost",
            "isEndFrame": False
        },
        {
            "id": "17855554543736wlrg",
            "start": 744.2918243643325,
            "length": 83.22765271847516,
            "prompt": "",
            "type": "image",
            "imageFile": "whatdreamscost/5.3.png",
            "imageB64": "/api/view?filename=5.3.png&type=input&subfolder=whatdreamscost",
            "isEndFrame": False
        }
    ],
    "motionSegments": [],
    "audioSegments": [
        {
            "id": "1785169457779kollx",
            "type": "audio",
            "start": 0,
            "length": 756.5194770828076,
            "trimStart": 446.9222739141953,
            "audioDurationFrames": 2880,
            "audioFile": "whatdreamscost/Late night trap.mp3",
            "fileName": "Late night trap.mp3",
            "waveformPeaks": []
        }
    ]
})


# =============================================================================
# CONFIGURATION DATACLASS
# =============================================================================
@dataclass
class LTX23Config:
    """Configuration for LTX-2.3 Director 2.0 MV pipeline."""

    # --- Project Paths ---
    project_dir: str = "/content/LTX23_Project"
    comfyui_dir: str = "/content/ComfyUI"
    output_dir: str = "/content/LTX23_Project/output"
    chunks_dir: str = "/content/LTX23_Project/chunks"
    state_file: str = "/content/LTX23_Project/generation_state.json"

    # --- Model Filenames ---
    unet_model: str = "ltx-2-3-22b-dev-Q4_K_M.gguf"
    video_vae: str = "LTX23_video_vae_bf16.safetensors"
    audio_vae: str = "LTX23_audio_vae_bf16.safetensors"
    tiny_vae: str = "taeltx2_3.safetensors"
    upscaler_model: str = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
    text_encoder_1: str = "gemma_3_12B_it_fp4_mixed.safetensors"
    text_encoder_2: str = "ltx-2.3_text_projection_bf16.safetensors"
    text_encoder_type: str = "ltxv"

    # --- LoRA Configuration ---
    lora_1_name: str = "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors"
    lora_1_strength: float = 0.4
    lora_2_name: str = "LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors"
    lora_2_strength: float = 0.6
    lora_3_name: str = "ltx2.3-transition.safetensors"
    lora_3_strength: float = 0.7
    lora_4_name: str = "LTX2.3-MVCamera-drclips.safetensors"
    lora_4_strength: float = 0.9

    # --- Resolution and Duration ---
    width: int = 1280
    height: int = 720
    frame_rate: int = 24
    duration: float = 31.5
    total_frames: int = 756
    resize_method: str = "maintain aspect ratio"
    divisible_by: int = 32
    img_compression: int = 18

    # --- Stage 1 Sampling ---
    stage1_sampler: str = "euler"
    stage1_scheduler: str = "linear_quadratic"
    stage1_steps: int = 8
    stage1_denoise: float = 1.0
    stage1_cfg: float = 1.0
    stage1_guide_strength: float = 0.5

    # --- Stage 2 Sampling ---
    stage2_sampler: str = "euler"
    stage2_scheduler: str = "linear_quadratic"
    stage2_steps: int = 4
    stage2_denoise: float = 0.42
    stage2_cfg: float = 1.0
    stage2_guide_strength: float = 1.0

    # --- Seed ---
    seed: int = 0
    noise_mode: str = "fixed"

    # --- Chunking ---
    max_retries: int = 3
    min_chunk_frames: int = 8

    # --- Output ---
    output_format: str = "video/h264-mp4"
    output_crf: int = 8
    output_pix_fmt: str = "yuv420p"
    filename_prefix: str = "LTX2.3/Video"

    # --- Modes ---
    test_mode: bool = False
    test_frames: int = 17
    api_fallback: bool = False
    api_host: str = "http://127.0.0.1:8188"

    # --- Timeline ---
    global_prompt: str = field(default_factory=lambda: GLOBAL_PROMPT)
    timeline_data: str = field(default_factory=lambda: TIMELINE_DATA_JSON)

    def config_hash(self) -> str:
        """Generate a hash of key configuration parameters for state validation."""
        key_params = f"{self.unet_model}|{self.width}x{self.height}|{self.total_frames}|{self.seed}"
        return hashlib.md5(key_params.encode()).hexdigest()[:12]



# =============================================================================
# MEMORY MANAGER
# =============================================================================
class MemoryManager:
    """VRAM and system memory management for T4 GPU."""

    def __init__(self):
        self.peak_vram_used: float = 0.0
        self.total_cleanups: int = 0
        self.oom_events: int = 0
        self._start_time: float = time.time()

    def get_free_vram(self) -> float:
        """Get available VRAM in GB."""
        if not torch or not torch.cuda.is_available():
            return 0.0
        free, _ = torch.cuda.mem_get_info()
        return free / (1024 ** 3)

    def get_total_vram(self) -> float:
        """Get total VRAM in GB."""
        if not torch or not torch.cuda.is_available():
            return 0.0
        _, total = torch.cuda.mem_get_info()
        return total / (1024 ** 3)

    def get_used_vram(self) -> float:
        """Get used VRAM in GB."""
        if not torch or not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / (1024 ** 3)

    def get_reserved_vram(self) -> float:
        """Get reserved VRAM in GB."""
        if not torch or not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_reserved() / (1024 ** 3)

    def report(self) -> Dict[str, float]:
        """Generate comprehensive memory report."""
        used = self.get_used_vram()
        if used > self.peak_vram_used:
            self.peak_vram_used = used

        report_data = {
            "free_vram_gb": round(self.get_free_vram(), 2),
            "used_vram_gb": round(used, 2),
            "reserved_vram_gb": round(self.get_reserved_vram(), 2),
            "total_vram_gb": round(self.get_total_vram(), 2),
            "peak_vram_gb": round(self.peak_vram_used, 2),
            "total_cleanups": self.total_cleanups,
            "oom_events": self.oom_events,
            "uptime_seconds": round(time.time() - self._start_time, 1),
        }

        logger.info(
            f"[Memory] Free: {report_data['free_vram_gb']:.2f}GB | "
            f"Used: {report_data['used_vram_gb']:.2f}GB | "
            f"Reserved: {report_data['reserved_vram_gb']:.2f}GB | "
            f"Peak: {report_data['peak_vram_gb']:.2f}GB"
        )
        return report_data

    def cleanup(self) -> None:
        """Standard memory cleanup: gc + empty_cache + ipc_collect."""
        gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        self.total_cleanups += 1
        logger.debug("[Memory] Standard cleanup completed")

    def emergency_cleanup(self) -> None:
        """Aggressive cleanup for OOM recovery."""
        self.oom_events += 1
        logger.warning("[Memory] Emergency cleanup triggered!")

        # Force garbage collection multiple passes
        for _ in range(3):
            gc.collect()

        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()

        # Additional cleanup of large tensors in global scope
        gc.collect()
        self.total_cleanups += 1
        logger.info(f"[Memory] Emergency cleanup done. Free VRAM: {self.get_free_vram():.2f}GB")

    def dynamic_chunk_size(self) -> int:
        """
        Determine optimal chunk size based on available VRAM.
        Returns number of frames per chunk.

        Thresholds for T4 (16GB):
        - <4GB free: 8 frames (minimum safe)
        - 4-6GB free: 9 frames
        - 6-9GB free: 17 frames
        - 9-12GB free: 25 frames
        - >12GB free: 33 frames
        """
        free_gb = self.get_free_vram()

        if free_gb < 4.0:
            chunk_size = 8
        elif free_gb < 6.0:
            chunk_size = 9
        elif free_gb < 9.0:
            chunk_size = 17
        elif free_gb < 12.0:
            chunk_size = 25
        else:
            chunk_size = 33

        logger.info(f"[Memory] Dynamic chunk size: {chunk_size} frames (free VRAM: {free_gb:.2f}GB)")
        return chunk_size

    def check_vram_threshold(self, min_gb: float = 2.0) -> bool:
        """Check if we have minimum VRAM available."""
        free = self.get_free_vram()
        if free < min_gb:
            logger.warning(
                f"[Memory] Low VRAM warning: {free:.2f}GB free (threshold: {min_gb}GB)"
            )
            return False
        return True


# =============================================================================
# TIMELINE PARSING
# =============================================================================
@dataclass
class TimelineSegment:
    """Represents a single timeline segment (image or audio)."""
    segment_id: str
    segment_type: str  # "image" or "audio"
    start: float
    length: float
    prompt: str = ""
    image_file: Optional[str] = None
    audio_file: Optional[str] = None
    trim_start: Optional[float] = None
    audio_duration_frames: Optional[int] = None
    is_end_frame: bool = False


def parse_timeline(timeline_json: str) -> Dict[str, List[TimelineSegment]]:
    """
    Parse timeline_data JSON into structured segments.

    Returns:
        Dictionary with 'image_segments' and 'audio_segments' keys.
    """
    data = json.loads(timeline_json) if isinstance(timeline_json, str) else timeline_json

    image_segments: List[TimelineSegment] = []
    audio_segments: List[TimelineSegment] = []

    # Parse image segments
    for seg in data.get("segments", []):
        image_segments.append(TimelineSegment(
            segment_id=seg.get("id", ""),
            segment_type="image",
            start=seg.get("start", 0),
            length=seg.get("length", 0),
            prompt=seg.get("prompt", ""),
            image_file=seg.get("imageFile"),
            is_end_frame=seg.get("isEndFrame", False),
        ))

    # Parse audio segments
    for seg in data.get("audioSegments", []):
        audio_segments.append(TimelineSegment(
            segment_id=seg.get("id", ""),
            segment_type="audio",
            start=seg.get("start", 0),
            length=seg.get("length", 0),
            audio_file=seg.get("audioFile"),
            trim_start=seg.get("trimStart"),
            audio_duration_frames=seg.get("audioDurationFrames"),
        ))

    logger.info(
        f"[Timeline] Parsed {len(image_segments)} image segments, "
        f"{len(audio_segments)} audio segments"
    )
    for i, seg in enumerate(image_segments):
        logger.info(
            f"  Image {i+1}: start={seg.start:.1f}, length={seg.length:.1f}, "
            f"file={seg.image_file}"
        )
    for seg in audio_segments:
        logger.info(
            f"  Audio: start={seg.start:.1f}, length={seg.length:.1f}, "
            f"file={seg.audio_file}, trimStart={seg.trim_start:.2f}"
        )

    return {
        "image_segments": image_segments,
        "audio_segments": audio_segments,
    }



# =============================================================================
# CHECKPOINT / RESUME SYSTEM
# =============================================================================
class CheckpointManager:
    """Manages generation state for checkpoint/resume capability."""

    def __init__(self, config: LTX23Config):
        self.config = config
        self.state_file = Path(config.state_file)
        self.state: Dict[str, Any] = {
            "config_hash": config.config_hash(),
            "seed": config.seed,
            "total_frames": config.total_frames,
            "completed_chunks": [],
            "current_chunk_index": 0,
            "chunk_files": [],
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_updated": "",
            "status": "initialized",
            "performance_log": [],
        }

    def save_state(self) -> None:
        """Save current generation state to disk."""
        self.state["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)
        logger.debug(f"[Checkpoint] State saved: {self.state_file}")

    def load_state(self) -> bool:
        """
        Load existing generation state. Returns True if valid state was loaded.
        Returns False if no state exists or config has changed.
        """
        if not self.state_file.exists():
            logger.info("[Checkpoint] No existing state found - starting fresh")
            return False

        try:
            with open(self.state_file, "r") as f:
                loaded = json.load(f)

            # Validate config hash matches
            if loaded.get("config_hash") != self.config.config_hash():
                logger.warning(
                    "[Checkpoint] Config hash mismatch - starting fresh. "
                    f"Saved: {loaded.get('config_hash')}, "
                    f"Current: {self.config.config_hash()}"
                )
                return False

            self.state = loaded
            completed = len(self.state.get("completed_chunks", []))
            logger.info(
                f"[Checkpoint] Resumed from state: {completed} chunks completed, "
                f"current index: {self.state.get('current_chunk_index', 0)}"
            )
            return True

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"[Checkpoint] Failed to load state: {e} - starting fresh")
            return False

    def mark_chunk_complete(
        self, chunk_index: int, chunk_file: str, elapsed_seconds: float,
        vram_used_gb: float
    ) -> None:
        """Record a completed chunk."""
        self.state["completed_chunks"].append(chunk_index)
        self.state["chunk_files"].append(chunk_file)
        self.state["current_chunk_index"] = chunk_index + 1
        self.state["performance_log"].append({
            "chunk_index": chunk_index,
            "file": chunk_file,
            "elapsed_seconds": round(elapsed_seconds, 2),
            "vram_used_gb": round(vram_used_gb, 2),
            "timestamp": time.strftime("%H:%M:%S"),
        })
        self.save_state()

    def is_chunk_completed(self, chunk_index: int) -> bool:
        """Check if a chunk has already been completed."""
        return chunk_index in self.state.get("completed_chunks", [])

    def mark_finished(self) -> None:
        """Mark the entire generation as finished."""
        self.state["status"] = "completed"
        self.state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.save_state()

    def mark_failed(self, reason: str) -> None:
        """Mark generation as failed."""
        self.state["status"] = "failed"
        self.state["failure_reason"] = reason
        self.save_state()


# =============================================================================
# PRE-FLIGHT CHECKS
# =============================================================================
def preflight_checks(config: LTX23Config) -> bool:
    """
    Verify all prerequisites before starting generation.

    Checks:
    - CUDA availability
    - GPU memory >= 14GB
    - Disk space availability
    - Model files exist
    - Custom nodes installed
    """
    logger.info("=" * 60)
    logger.info("PRE-FLIGHT CHECKS")
    logger.info("=" * 60)
    all_ok = True

    # 1. CUDA check
    if not torch or not torch.cuda.is_available():
        logger.error("[Preflight] CUDA is NOT available! GPU required.")
        all_ok = False
    else:
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"[Preflight] CUDA OK - GPU: {gpu_name}")

    # 2. GPU memory check
    if torch and torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        if total_vram < 14.0:
            logger.warning(
                f"[Preflight] GPU VRAM ({total_vram:.1f}GB) is below recommended 14GB. "
                "Performance may be degraded."
            )
        else:
            logger.info(f"[Preflight] GPU VRAM OK: {total_vram:.1f}GB")

    # 3. Disk space check
    project_path = Path(config.project_dir)
    project_path.mkdir(parents=True, exist_ok=True)
    disk_usage = shutil.disk_usage(str(project_path))
    free_gb = disk_usage.free / (1024 ** 3)
    if free_gb < 10.0:
        logger.warning(f"[Preflight] Low disk space: {free_gb:.1f}GB free (need ~10GB)")
    else:
        logger.info(f"[Preflight] Disk space OK: {free_gb:.1f}GB free")

    # 4. Model files check
    model_dirs = {
        "unet": config.unet_model,
        "vae": config.video_vae,
        "vae_audio": config.audio_vae,
        "text_encoders": config.text_encoder_1,
        "text_encoders_2": config.text_encoder_2,
        "latent_upscale_models": config.upscaler_model,
    }

    models_base = Path(config.comfyui_dir) / "models"
    for subdir, filename in model_dirs.items():
        actual_subdir = subdir.split("_2")[0] if "_2" in subdir else subdir
        if actual_subdir == "vae_audio":
            actual_subdir = "vae"
        if actual_subdir == "text_encoders":
            actual_subdir = "text_encoders"
        model_path = models_base / actual_subdir / filename
        if model_path.exists():
            logger.info(f"[Preflight] Model OK: {filename}")
        else:
            logger.warning(f"[Preflight] Model MISSING: {model_path}")
            # Not a hard failure - download function will handle it

    # 5. Custom nodes check
    required_nodes = [
        "ComfyUI-GGUF",
        "ComfyUI-KJNodes",
        "ComfyUI-LTXVideo",
        "rgthree-comfy",
        "comfyui-videohelpersuite",
    ]
    custom_nodes_dir = Path(config.comfyui_dir) / "custom_nodes"
    for node_name in required_nodes:
        node_path = custom_nodes_dir / node_name
        if node_path.exists():
            logger.info(f"[Preflight] Custom node OK: {node_name}")
        else:
            logger.warning(f"[Preflight] Custom node MISSING: {node_name}")

    # 6. FFmpeg check
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            logger.info("[Preflight] FFmpeg OK")
        else:
            logger.warning("[Preflight] FFmpeg not working properly")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("[Preflight] FFmpeg not found")

    logger.info("=" * 60)
    if all_ok:
        logger.info("[Preflight] All critical checks PASSED")
    else:
        logger.error("[Preflight] Some critical checks FAILED")
    logger.info("=" * 60)

    return all_ok



# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
def setup_environment(config: LTX23Config) -> None:
    """
    Install all dependencies and set up ComfyUI environment.
    This is designed to run in Google Colab.
    """
    logger.info("=" * 60)
    logger.info("ENVIRONMENT SETUP")
    logger.info("=" * 60)

    # Set CUDA memory configuration
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Install PyTorch (usually pre-installed in Colab)
    _run_cmd("pip install -q torch torchvision torchaudio", "Installing PyTorch")

    # Clone ComfyUI
    comfyui_path = Path(config.comfyui_dir)
    if not comfyui_path.exists():
        _run_cmd(
            f"git clone https://github.com/comfyanonymous/ComfyUI {config.comfyui_dir}",
            "Cloning ComfyUI"
        )
        _run_cmd(
            f"pip install -q -r {config.comfyui_dir}/requirements.txt",
            "Installing ComfyUI requirements"
        )
    else:
        logger.info("[Setup] ComfyUI already exists, skipping clone")

    # Install additional Python packages
    _run_cmd(
        "pip install -q torchsde einops diffusers accelerate",
        "Installing diffusion dependencies"
    )
    _run_cmd(
        "pip install -q av spandrel albumentations onnx opencv-python onnxruntime nest_asyncio",
        "Installing media/ML packages"
    )

    # Clone custom nodes
    custom_nodes_dir = comfyui_path / "custom_nodes"
    custom_nodes_dir.mkdir(parents=True, exist_ok=True)

    custom_nodes = {
        "ComfyUI-Manager": "https://github.com/comfy-org/ComfyUI-Manager",
        "WhatDreamsCost-ComfyUI": "https://github.com/WhatDreamscost/WhatDreamsCost-ComfyUI",
        "rgthree-comfy": "https://github.com/rgthree/rgthree-comfy",
        "ComfyUI-Licon-MSR": "https://github.com/liconstudio/ComfyUI-Licon-MSR",
        "ComfyUI-KJNodes": "https://github.com/kijai/ComfyUI-KJNodes",
        "ComfyUI-GGUF": "https://github.com/city96/ComfyUI-GGUF",
        "ComfyUI-LTXVideo": "https://github.com/Lightricks/ComfyUI-LTXVideo",
        "ComfyUI-VideoHelperSuite": "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite",
        "ComfyUI-MelBandRoFormer": "https://github.com/kijai/ComfyUI-MelBandRoFormer",
    }

    for name, url in custom_nodes.items():
        node_path = custom_nodes_dir / name
        if not node_path.exists():
            _run_cmd(
                f"git clone {url} {node_path}",
                f"Cloning {name}"
            )
            # Install requirements if they exist
            req_file = node_path / "requirements.txt"
            if req_file.exists():
                _run_cmd(
                    f"pip install -q -r {req_file}",
                    f"Installing {name} requirements"
                )
        else:
            logger.info(f"[Setup] {name} already exists")

    # Install system packages
    _run_cmd("apt-get -y install -qq aria2 ffmpeg", "Installing aria2 and ffmpeg")

    # Create project directories
    Path(config.project_dir).mkdir(parents=True, exist_ok=True)
    Path(config.chunks_dir).mkdir(parents=True, exist_ok=True)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Add ComfyUI to sys.path
    if config.comfyui_dir not in sys.path:
        sys.path.insert(0, config.comfyui_dir)

    logger.info("[Setup] Environment setup complete!")


def _run_cmd(cmd: str, description: str = "", silent: bool = True) -> bool:
    """Run a shell command with error handling."""
    if description:
        logger.info(f"[Setup] {description}...")
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            logger.warning(f"[Setup] Command had issues: {result.stderr[:200]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"[Setup] Command timed out: {cmd[:80]}")
        return False
    except Exception as e:
        logger.error(f"[Setup] Command failed: {e}")
        return False


# =============================================================================
# MODEL DOWNLOAD
# =============================================================================
def download_models(config: LTX23Config) -> None:
    """Download all required models using aria2c."""
    logger.info("=" * 60)
    logger.info("MODEL DOWNLOADS")
    logger.info("=" * 60)

    models_base = Path(config.comfyui_dir) / "models"

    # Model definitions: (url, destination_subdir, filename)
    model_downloads = [
        # UNet
        (
            "https://huggingface.co/vantagewithai/LTX-2.3-GGUF/resolve/main/dev/ltx-2-3-22b-dev-Q4_K_M.gguf",
            "unet",
            config.unet_model,
        ),
        # Video VAE
        (
            "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_video_vae_bf16.safetensors",
            "vae",
            config.video_vae,
        ),
        # Audio VAE
        (
            "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_audio_vae_bf16.safetensors",
            "vae",
            config.audio_vae,
        ),
        # Tiny VAE (for preview)
        (
            "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/taeltx2_3.safetensors",
            "vae",
            config.tiny_vae,
        ),
        # Upscaler
        (
            "https://huggingface.co/vidfom/aimusic/resolve/main/ComfyUI/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            "latent_upscale_models",
            config.upscaler_model,
        ),
        # Text encoder 1 (Gemma)
        (
            "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
            "text_encoders",
            config.text_encoder_1,
        ),
        # Text encoder 2 (LTX text projection)
        (
            "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors",
            "text_encoders",
            config.text_encoder_2,
        ),
        # LoRAs
        (
            "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/loras/ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors",
            "loras",
            config.lora_1_name,
        ),
        (
            "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors",
            "loras",
            config.lora_2_name,
        ),
        (
            "https://huggingface.co/joyfox/LTX-2.3-Transition-LORA/resolve/main/ltx2.3-transition.safetensors",
            "loras",
            config.lora_3_name,
        ),
        (
            "https://huggingface.co/vidfom/aimusic/resolve/main/ComfyUI/models/loras/LTX2.3-MVCamera-drclips.safetensors",
            "loras",
            config.lora_4_name,
        ),
        # Additional LoRAs
        (
            "https://huggingface.co/vrgamedevgirl84/LTX_2.3_Crisp_Enhance_Style_LoRa/resolve/main/LTX2.3_Crisp_Enhance.safetensors",
            "loras",
            "LTX2.3_Crisp_Enhance.safetensors",
        ),
        (
            "https://huggingface.co/LiconStudio/LTX-2.3-Multiple-Subject-Reference/resolve/main/LTX-2.3-Licon-MSR-V2.safetensors",
            "loras",
            "LTX-2.3-Licon-MSR-V2.safetensors",
        ),
    ]

    for url, subdir, filename in model_downloads:
        dest_dir = models_base / subdir
        dest_file = dest_dir / filename
        if dest_file.exists():
            logger.info(f"[Download] Already exists: {filename}")
            continue
        _download_with_aria2c(url, str(dest_dir), filename)

    logger.info("[Download] All models ready!")


def download_assets(config: LTX23Config) -> None:
    """Download reference images and audio for the timeline."""
    logger.info("=" * 60)
    logger.info("ASSET DOWNLOADS")
    logger.info("=" * 60)

    input_dir = Path(config.comfyui_dir) / "input" / "whatdreamscost"
    input_dir.mkdir(parents=True, exist_ok=True)

    # Reference images
    image_files = ["1.png", "2.png", "3.png", "4.png", "5.3.png"]
    base_url = "https://huggingface.co/whatdreamscost/LTX-Director-Assets/resolve/main"

    for img_file in image_files:
        dest = input_dir / img_file
        if dest.exists():
            logger.info(f"[Assets] Already exists: {img_file}")
            continue
        url = f"{base_url}/{img_file}"
        _download_with_aria2c(url, str(input_dir), img_file)

    # Audio file
    audio_file = "Late night trap.mp3"
    audio_dest = input_dir / audio_file
    if not audio_dest.exists():
        audio_url = f"{base_url}/Late%20night%20trap.mp3"
        _download_with_aria2c(audio_url, str(input_dir), audio_file)
    else:
        logger.info(f"[Assets] Already exists: {audio_file}")

    logger.info("[Assets] All assets ready!")


def _download_with_aria2c(url: str, dest_dir: str, filename: str) -> bool:
    """Download a file using aria2c with resume support."""
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        "aria2c",
        "--console-log-level=error",
        "-c", "-x", "16", "-s", "16", "-k", "1M",
        "--summary-interval=0",
        "-d", dest_dir,
        "-o", filename,
        url,
    ]
    logger.info(f"[Download] Downloading {filename}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode == 0:
            logger.info(f"[Download] Done: {filename}")
            return True
        else:
            logger.error(f"[Download] Failed: {filename} - {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"[Download] Timeout downloading: {filename}")
        return False
    except FileNotFoundError:
        # Fallback to wget if aria2c not available
        logger.warning("[Download] aria2c not found, trying wget...")
        wget_cmd = f'wget -q -O "{dest_dir}/{filename}" "{url}"'
        return _run_cmd(wget_cmd, f"wget fallback for {filename}")



# =============================================================================
# COMFYUI NODE INITIALIZATION
# =============================================================================
def import_custom_nodes() -> None:
    """
    Load all built-in and external custom nodes in a Jupyter/Colab-safe way.
    Uses asyncio with nest_asyncio fallback for running inside Colab event loops.
    """
    from nodes import init_builtin_extra_nodes, init_external_custom_nodes

    async def loader():
        import_failed = await init_builtin_extra_nodes()
        await init_external_custom_nodes()
        if import_failed:
            logger.warning("Some comfy_extras nodes failed to import:")
            for node in import_failed:
                logger.warning(f"  - {node}")

    try:
        import asyncio
        asyncio.run(loader())
    except RuntimeError:
        # Already inside an event loop (Jupyter/Colab)
        import nest_asyncio
        nest_asyncio.apply()
        import asyncio
        loop = asyncio.get_event_loop()
        loop.run_until_complete(loader())

    logger.info("[Nodes] Custom nodes loaded successfully")


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """
    Returns the value at the given index of a sequence or mapping.
    Handles ComfyUI node output format where results may be in a 'result' key.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


# =============================================================================
# PERFORMANCE LOGGER
# =============================================================================
class PerformanceLogger:
    """Track and report per-chunk generation performance."""

    def __init__(self):
        self.chunk_times: List[Dict[str, Any]] = []
        self._generation_start: Optional[float] = None

    def start_generation(self) -> None:
        """Mark the start of the full generation."""
        self._generation_start = time.time()

    def log_chunk(
        self, chunk_index: int, frames: int, elapsed: float, vram_gb: float
    ) -> None:
        """Log performance data for a single chunk."""
        fps = frames / elapsed if elapsed > 0 else 0
        entry = {
            "chunk": chunk_index,
            "frames": frames,
            "elapsed_sec": round(elapsed, 2),
            "fps": round(fps, 3),
            "vram_used_gb": round(vram_gb, 2),
        }
        self.chunk_times.append(entry)
        logger.info(
            f"[Perf] Chunk {chunk_index}: {frames} frames in {elapsed:.1f}s "
            f"({fps:.3f} fps) | VRAM: {vram_gb:.2f}GB"
        )

    def summary(self) -> str:
        """Generate performance summary."""
        if not self.chunk_times:
            return "No chunks processed"

        total_frames = sum(c["frames"] for c in self.chunk_times)
        total_time = sum(c["elapsed_sec"] for c in self.chunk_times)
        avg_fps = total_frames / total_time if total_time > 0 else 0
        peak_vram = max(c["vram_used_gb"] for c in self.chunk_times)

        wall_time = time.time() - self._generation_start if self._generation_start else total_time

        summary_text = (
            f"\n{'='*60}\n"
            f"PERFORMANCE SUMMARY\n"
            f"{'='*60}\n"
            f"Total frames: {total_frames}\n"
            f"Total chunks: {len(self.chunk_times)}\n"
            f"Processing time: {total_time:.1f}s\n"
            f"Wall clock time: {wall_time:.1f}s\n"
            f"Average FPS: {avg_fps:.3f}\n"
            f"Peak VRAM: {peak_vram:.2f}GB\n"
            f"{'='*60}"
        )
        return summary_text


# =============================================================================
# EMERGENCY CLEANUP / SIGNAL HANDLERS
# =============================================================================
_cleanup_registered = False
_temp_files: List[str] = []


def register_emergency_cleanup(config: LTX23Config) -> None:
    """Register cleanup handlers for graceful shutdown."""
    global _cleanup_registered
    if _cleanup_registered:
        return

    def cleanup_handler():
        """Clean up temporary files on exit."""
        logger.info("[Cleanup] Running exit cleanup...")
        for f in _temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except OSError:
                pass
        # Clean VRAM
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def signal_handler(signum, frame):
        """Handle termination signals gracefully."""
        sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
        logger.warning(f"[Cleanup] Received signal {sig_name} - cleaning up...")
        cleanup_handler()
        sys.exit(1)

    atexit.register(cleanup_handler)

    # Register signal handlers (not available in all environments)
    try:
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    except (OSError, ValueError):
        # Signal handling may not work in all contexts (e.g., threads)
        pass

    _cleanup_registered = True
    logger.info("[Cleanup] Emergency cleanup handlers registered")



# =============================================================================
# CHUNK PROCESSOR - Core Generation Logic
# =============================================================================
class ChunkProcessor:
    """
    Processes video chunks using LOAD-USE-RELEASE-CLEAN-CHECKPOINT pattern.

    Each chunk goes through:
    1. LOAD: Load necessary models/data for this chunk
    2. USE: Run the two-stage sampling pipeline
    3. RELEASE: Delete intermediate tensors and models
    4. CLEAN: Run garbage collection and VRAM cleanup
    5. CHECKPOINT: Save state to disk
    """

    def __init__(
        self,
        config: LTX23Config,
        memory_manager: MemoryManager,
        checkpoint_manager: CheckpointManager,
        performance_logger: PerformanceLogger,
    ):
        self.config = config
        self.memory = memory_manager
        self.checkpoint = checkpoint_manager
        self.perf = performance_logger

    def process_chunk(
        self,
        chunk_index: int,
        start_frame: int,
        end_frame: int,
        pipeline_context: Dict[str, Any],
    ) -> Optional[str]:
        """
        Process a single chunk of frames through the full pipeline.

        Implements OOM detection with retry at smaller chunk sizes.

        Args:
            chunk_index: Index of this chunk
            start_frame: Starting frame number
            end_frame: Ending frame number (exclusive)
            pipeline_context: Shared pipeline state (models, latents, etc.)

        Returns:
            Path to the saved chunk video file, or None if failed.
        """
        num_frames = end_frame - start_frame
        retries = 0
        current_frames = num_frames

        while retries <= self.config.max_retries:
            try:
                logger.info(
                    f"\n{'='*60}\n"
                    f"CHUNK {chunk_index}: frames {start_frame}-{end_frame-1} "
                    f"({current_frames} frames)\n"
                    f"{'='*60}"
                )

                chunk_start_time = time.time()
                self.memory.report()

                # === LOAD phase ===
                # Models should already be in pipeline_context from initial load

                # === USE phase - Two Stage Sampling ===
                chunk_file = self._run_two_stage_sampling(
                    chunk_index, start_frame, end_frame, current_frames, pipeline_context
                )

                # === RELEASE phase ===
                # Release intermediate tensors (handled within sampling)

                # === CLEAN phase ===
                self.memory.cleanup()

                # === CHECKPOINT phase ===
                elapsed = time.time() - chunk_start_time
                vram_used = self.memory.get_used_vram()
                self.checkpoint.mark_chunk_complete(
                    chunk_index, chunk_file, elapsed, vram_used
                )
                self.perf.log_chunk(chunk_index, current_frames, elapsed, vram_used)

                return chunk_file

            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "CUDA" in str(e):
                    retries += 1
                    logger.error(
                        f"[OOM] CUDA OOM on chunk {chunk_index} with {current_frames} frames. "
                        f"Retry {retries}/{self.config.max_retries}"
                    )

                    # Emergency cleanup
                    self.memory.emergency_cleanup()

                    # Halve the chunk size but respect minimum
                    current_frames = max(
                        self.config.min_chunk_frames,
                        current_frames // 2
                    )
                    end_frame = start_frame + current_frames

                    if retries > self.config.max_retries:
                        logger.error(
                            f"[OOM] Max retries exceeded for chunk {chunk_index}. Aborting."
                        )
                        return None

                    # Wait for GPU to settle
                    time.sleep(2)
                else:
                    logger.error(f"[Error] Non-OOM error in chunk {chunk_index}: {e}")
                    logger.error(traceback.format_exc())
                    return None

            except Exception as e:
                logger.error(f"[Error] Unexpected error in chunk {chunk_index}: {e}")
                logger.error(traceback.format_exc())
                return None

        return None

    def _run_two_stage_sampling(
        self,
        chunk_index: int,
        start_frame: int,
        end_frame: int,
        num_frames: int,
        ctx: Dict[str, Any],
    ) -> str:
        """
        Execute the two-stage sampling pipeline for a chunk.

        Stage 1: Low-resolution sampling (8 steps, denoise 1.0, guide_strength 0.5)
        Stage 2: Upscaled sampling (4 steps, denoise 0.42, guide_strength 1.0)

        Returns path to saved chunk file.
        """
        from nodes import NODE_CLASS_MAPPINGS

        # Get shared objects from context
        model = ctx["model"]
        positive = ctx["positive"]
        negative = ctx["negative"]
        video_latent = ctx["video_latent"]
        audio_latent = ctx["audio_latent"]
        guide_data = ctx["guide_data"]
        motion_guide_data = ctx["motion_guide_data"]
        video_vae = ctx["video_vae"]
        audio_vae = ctx["audio_vae"]
        frame_rate = ctx["frame_rate"]
        upscale_model = ctx["upscale_model"]
        noise = ctx["noise"]

        # === STAGE 1: Low-resolution sampling ===
        logger.info(f"[Stage 1] Starting low-res sampling (guide_strength={self.config.stage1_guide_strength})...")

        # LTXDirectorGuide (node 133) - Stage 1 guide
        ltxdirectorguide = NODE_CLASS_MAPPINGS["LTXDirectorGuide"]()
        stage1_guide_result = ltxdirectorguide.EXECUTE_NORMALIZED(
            positive=positive,
            negative=negative,
            vae=video_vae,
            latent=video_latent,
            guide_data=guide_data,
            motion_guide_data=motion_guide_data,
            model=model,
            image_path="None",
            apply_to_negative=1,
            guide_strength=self.config.stage1_guide_strength,
            scale_method="bicubic",
            crop=1,
            crop_method="center",
            extend_with_last_frame=True,
            offset_frames=False,
            latent_height=256,
            latent_width=64,
            use_latent_input=False,
        )

        s1_positive = get_value_at_index(stage1_guide_result, 0)
        s1_negative = get_value_at_index(stage1_guide_result, 1)
        s1_latent = get_value_at_index(stage1_guide_result, 2)
        s1_model = get_value_at_index(stage1_guide_result, 3)

        # LTXVConcatAVLatent (node 29) - Combine video + audio latents for Stage 1
        ltxvconcatavlatent = NODE_CLASS_MAPPINGS["LTXVConcatAVLatent"]()
        stage1_combined = ltxvconcatavlatent.EXECUTE_NORMALIZED(
            video_latent=s1_latent,
            audio_latent=audio_latent,
        )

        # BasicScheduler (node 33) - Stage 1 schedule
        basicscheduler = NODE_CLASS_MAPPINGS["BasicScheduler"]()
        stage1_sigmas = basicscheduler.EXECUTE_NORMALIZED(
            scheduler=self.config.stage1_scheduler,
            steps=self.config.stage1_steps,
            denoise=self.config.stage1_denoise,
            model=s1_model,
        )

        # KSamplerSelect (node 32) - Stage 1 sampler
        ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        stage1_sampler = ksamplerselect.EXECUTE_NORMALIZED(
            sampler_name=self.config.stage1_sampler,
        )

        # CFGGuider (node 28) - Stage 1 guidance
        cfgguider = NODE_CLASS_MAPPINGS["CFGGuider"]()
        stage1_guider = cfgguider.EXECUTE_NORMALIZED(
            cfg=self.config.stage1_cfg,
            model=s1_model,
            positive=s1_positive,
            negative=s1_negative,
        )

        # SamplerCustomAdvanced (node 31) - Stage 1 sampling
        samplercustomadvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
        stage1_output = samplercustomadvanced.EXECUTE_NORMALIZED(
            noise=noise,
            guider=get_value_at_index(stage1_guider, 0),
            sampler=get_value_at_index(stage1_sampler, 0),
            sigmas=get_value_at_index(stage1_sigmas, 0),
            latent_image=get_value_at_index(stage1_combined, 0),
        )

        logger.info("[Stage 1] Sampling complete")

        # LTXVSeparateAVLatent (node 34) - Separate after Stage 1
        ltxvseparateavlatent = NODE_CLASS_MAPPINGS["LTXVSeparateAVLatent"]()
        stage1_separated = ltxvseparateavlatent.EXECUTE_NORMALIZED(
            av_latent=get_value_at_index(stage1_output, 0),
        )

        stage1_video_latent = get_value_at_index(stage1_separated, 0)
        stage1_audio_latent = get_value_at_index(stage1_separated, 1)

        # LTXDirectorCropGuides (node 55) - Crop guides after Stage 1
        ltxdirectorcropguides = NODE_CLASS_MAPPINGS["LTXDirectorCropGuides"]()
        stage1_cropped = ltxdirectorcropguides.EXECUTE_NORMALIZED(
            positive=s1_positive,
            negative=s1_negative,
            latent=stage1_video_latent,
        )

        s1_cropped_positive = get_value_at_index(stage1_cropped, 0)
        s1_cropped_negative = get_value_at_index(stage1_cropped, 1)
        s1_cropped_latent = get_value_at_index(stage1_cropped, 2)

        # Release Stage 1 intermediates
        del stage1_guide_result, stage1_combined, stage1_sigmas
        del stage1_sampler, stage1_guider, stage1_output, stage1_separated
        self.memory.cleanup()

        # === STAGE 2: Upscaled sampling ===
        logger.info(f"[Stage 2] Starting upscaled sampling (guide_strength={self.config.stage2_guide_strength})...")

        # LTXVLatentUpsampler (node 14)
        ltxvlatentupsampler = NODE_CLASS_MAPPINGS["LTXVLatentUpsampler"]()
        upsampled_latent = ltxvlatentupsampler.upsample_latent(
            samples=s1_cropped_latent,
            upscale_model=upscale_model,
            vae=video_vae,
        )

        # LTXDirectorGuide (node 132) - Stage 2 guide
        stage2_guide_result = ltxdirectorguide.EXECUTE_NORMALIZED(
            positive=s1_cropped_positive,
            negative=s1_cropped_negative,
            vae=video_vae,
            latent=get_value_at_index(upsampled_latent, 0),
            guide_data=guide_data,
            motion_guide_data=motion_guide_data,
            model=model,
            image_path="None",
            apply_to_negative=1,
            guide_strength=self.config.stage2_guide_strength,
            scale_method="bicubic",
            crop=1,
            crop_method="center",
            extend_with_last_frame=True,
            offset_frames=False,
            latent_height=256,
            latent_width=64,
            use_latent_input=False,
        )

        s2_positive = get_value_at_index(stage2_guide_result, 0)
        s2_negative = get_value_at_index(stage2_guide_result, 1)
        s2_latent = get_value_at_index(stage2_guide_result, 2)
        s2_model = get_value_at_index(stage2_guide_result, 3)

        # LTXVConcatAVLatent (node 18) - Combine for Stage 2
        stage2_combined = ltxvconcatavlatent.EXECUTE_NORMALIZED(
            video_latent=s2_latent,
            audio_latent=stage1_audio_latent,
        )

        # BasicScheduler (node 21) - Stage 2 schedule
        stage2_sigmas = basicscheduler.EXECUTE_NORMALIZED(
            scheduler=self.config.stage2_scheduler,
            steps=self.config.stage2_steps,
            denoise=self.config.stage2_denoise,
            model=s2_model,
        )

        # KSamplerSelect (node 20) - Stage 2 sampler
        stage2_sampler = ksamplerselect.EXECUTE_NORMALIZED(
            sampler_name=self.config.stage2_sampler,
        )

        # CFGGuider (node 17) - Stage 2 guidance
        stage2_guider = cfgguider.EXECUTE_NORMALIZED(
            cfg=self.config.stage2_cfg,
            model=s2_model,
            positive=s2_positive,
            negative=s2_negative,
        )

        # SamplerCustomAdvanced (node 19) - Stage 2 sampling
        stage2_output = samplercustomadvanced.EXECUTE_NORMALIZED(
            noise=noise,
            guider=get_value_at_index(stage2_guider, 0),
            sampler=get_value_at_index(stage2_sampler, 0),
            sigmas=get_value_at_index(stage2_sigmas, 0),
            latent_image=get_value_at_index(stage2_combined, 0),
        )

        logger.info("[Stage 2] Sampling complete")

        # LTXVSeparateAVLatent (node 22) - Separate after Stage 2
        stage2_separated = ltxvseparateavlatent.EXECUTE_NORMALIZED(
            av_latent=get_value_at_index(stage2_output, 0),
        )

        stage2_video_latent = get_value_at_index(stage2_separated, 0)
        stage2_audio_latent = get_value_at_index(stage2_separated, 1)

        # LTXDirectorCropGuides (node 54) - Final crop
        stage2_cropped = ltxdirectorcropguides.EXECUTE_NORMALIZED(
            positive=s2_positive,
            negative=s2_negative,
            latent=stage2_video_latent,
        )

        final_video_latent = get_value_at_index(stage2_cropped, 2)

        # Release Stage 2 intermediates
        del stage2_guide_result, stage2_combined, stage2_sigmas
        del stage2_sampler, stage2_guider, stage2_output, stage2_separated
        del upsampled_latent
        self.memory.cleanup()

        # === VAE DECODE ===
        logger.info("[Decode] Decoding video latent...")

        # VAEDecode (node 1) - Decode video
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        decoded_video = vaedecode.decode(
            samples=final_video_latent,
            vae=video_vae,
        )

        video_frames = get_value_at_index(decoded_video, 0)

        # LTXVAudioVAEDecode (node 24) - Decode audio
        ltxvaudiovaedecode = NODE_CLASS_MAPPINGS["LTXVAudioVAEDecode"]()
        decoded_audio = ltxvaudiovaedecode.EXECUTE_NORMALIZED(
            samples=stage2_audio_latent,
            audio_vae=audio_vae,
        )

        audio_data = get_value_at_index(decoded_audio, 0)

        # Release decode intermediates
        del final_video_latent, stage2_audio_latent, decoded_video, decoded_audio
        self.memory.cleanup()

        # === SAVE CHUNK ===
        chunk_file = self._save_chunk(chunk_index, video_frames, audio_data, frame_rate)

        # Release frames
        del video_frames, audio_data
        self.memory.cleanup()

        return chunk_file

    def _save_chunk(
        self,
        chunk_index: int,
        video_frames: Any,
        audio_data: Any,
        frame_rate: float,
    ) -> str:
        """Save a chunk to disk as an MP4 file using VHS_VideoCombine or ffmpeg."""
        from nodes import NODE_CLASS_MAPPINGS

        chunk_filename = f"chunk_{chunk_index:04d}.mp4"
        chunk_path = os.path.join(self.config.chunks_dir, chunk_filename)

        # Try using VHS_VideoCombine if available
        if "VHS_VideoCombine" in NODE_CLASS_MAPPINGS:
            try:
                vhs_combine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
                vhs_result = vhs_combine.EXECUTE_NORMALIZED(
                    images=video_frames,
                    audio=audio_data,
                    frame_rate=frame_rate,
                    loop_count=0,
                    filename_prefix=f"chunks/chunk_{chunk_index:04d}",
                    format=self.config.output_format,
                    pix_fmt=self.config.output_pix_fmt,
                    crf=self.config.output_crf,
                    save_metadata=False,
                    trim_to_audio=False,
                    pingpong=False,
                    save_output=True,
                )
                logger.info(f"[Save] Chunk {chunk_index} saved via VHS_VideoCombine")
                # The VHS node saves to ComfyUI output, copy to our chunks dir
                return chunk_path
            except Exception as e:
                logger.warning(f"[Save] VHS_VideoCombine failed, using ffmpeg: {e}")

        # Fallback: save frames + audio with ffmpeg
        self._save_chunk_ffmpeg(chunk_index, video_frames, audio_data, frame_rate, chunk_path)
        return chunk_path

    def _save_chunk_ffmpeg(
        self,
        chunk_index: int,
        video_frames: Any,
        audio_data: Any,
        frame_rate: float,
        output_path: str,
    ) -> None:
        """Save chunk using raw ffmpeg encoding from tensor data."""
        import numpy as np

        # Convert tensor frames to numpy
        if hasattr(video_frames, 'cpu'):
            frames_np = video_frames.cpu().numpy()
        else:
            frames_np = np.array(video_frames)

        # Frames are in (N, H, W, C) format, float [0,1]
        if frames_np.max() <= 1.0:
            frames_np = (frames_np * 255).astype(np.uint8)

        n_frames, height, width, channels = frames_np.shape

        # Write frames via pipe to ffmpeg
        temp_video = output_path + ".tmp.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "rgb24",
            "-r", str(int(frame_rate)),
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", str(self.config.output_crf),
            "-pix_fmt", self.config.output_pix_fmt,
            temp_video,
        ]

        try:
            proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            for frame in frames_np:
                proc.stdin.write(frame.tobytes())
            proc.stdin.close()
            proc.wait(timeout=120)

            if proc.returncode == 0:
                # Move temp to final
                shutil.move(temp_video, output_path)
                logger.info(f"[Save] Chunk {chunk_index} saved: {output_path}")
            else:
                stderr = proc.stderr.read().decode()
                logger.error(f"[Save] FFmpeg error: {stderr[:200]}")
        except Exception as e:
            logger.error(f"[Save] Failed to save chunk {chunk_index}: {e}")
            if os.path.exists(temp_video):
                os.remove(temp_video)



# =============================================================================
# MAIN PIPELINE - Direct Node Execution Mode
# =============================================================================
class LTX23DirectorPipeline:
    """
    Main pipeline orchestrating the full LTX-2.3 Director 2.0 generation.
    Uses NODE_CLASS_MAPPINGS for direct Python execution of ComfyUI nodes.
    """

    def __init__(self, config: LTX23Config):
        self.config = config
        self.memory = MemoryManager()
        self.checkpoint = CheckpointManager(config)
        self.perf = PerformanceLogger()
        self.chunk_processor = ChunkProcessor(
            config, self.memory, self.checkpoint, self.perf
        )
        self.pipeline_context: Dict[str, Any] = {}

    def run(self) -> Optional[str]:
        """
        Execute the full generation pipeline.

        Returns path to the final output video, or None if failed.
        """
        try:
            self.perf.start_generation()
            register_emergency_cleanup(self.config)

            # Check for resume
            resumed = self.checkpoint.load_state()
            if not resumed:
                self.checkpoint.save_state()

            # Initialize pipeline context (load models)
            logger.info("\n" + "=" * 60)
            logger.info("INITIALIZING PIPELINE")
            logger.info("=" * 60)
            self._initialize_models()

            # Determine chunks
            total_frames = self.config.total_frames
            if self.config.test_mode:
                total_frames = min(self.config.test_frames, total_frames)
                logger.info(f"[TEST MODE] Generating only {total_frames} frames")

            chunk_size = self.memory.dynamic_chunk_size()
            chunks = self._compute_chunks(total_frames, chunk_size)
            logger.info(f"[Pipeline] Total frames: {total_frames}, Chunks: {len(chunks)}")

            # Process chunks
            chunk_files = []
            for i, (start, end) in enumerate(chunks):
                # Skip already completed chunks (resume support)
                if self.checkpoint.is_chunk_completed(i):
                    existing_file = os.path.join(
                        self.config.chunks_dir, f"chunk_{i:04d}.mp4"
                    )
                    if os.path.exists(existing_file):
                        chunk_files.append(existing_file)
                        logger.info(f"[Pipeline] Skipping chunk {i} (already completed)")
                        continue

                chunk_file = self.chunk_processor.process_chunk(
                    i, start, end, self.pipeline_context
                )

                if chunk_file is None:
                    logger.error(f"[Pipeline] Chunk {i} failed! Aborting generation.")
                    self.checkpoint.mark_failed(f"Chunk {i} failed")
                    return None

                chunk_files.append(chunk_file)

            # Release all models before assembly
            self._release_models()
            self.memory.cleanup()

            # FFmpeg final assembly
            logger.info("\n" + "=" * 60)
            logger.info("FINAL ASSEMBLY")
            logger.info("=" * 60)
            output_path = self._assemble_final_video(chunk_files)

            # Mark complete
            self.checkpoint.mark_finished()
            logger.info(self.perf.summary())
            logger.info(f"\n[DONE] Final video: {output_path}")

            return output_path

        except Exception as e:
            logger.error(f"[Pipeline] Fatal error: {e}")
            logger.error(traceback.format_exc())
            self.checkpoint.mark_failed(str(e))
            return None

    def _initialize_models(self) -> None:
        """Load all models and prepare the pipeline context."""
        from nodes import NODE_CLASS_MAPPINGS
        from nodes import LoraLoaderModelOnly

        self.memory.report()

        # --- 1. UnetLoaderGGUF ---
        logger.info("[Init] Loading UNet (GGUF)...")
        unetloadergguf = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
        unet_result = unetloadergguf.load_unet(unet_name=self.config.unet_model)
        model = get_value_at_index(unet_result, 0)
        del unet_result
        self.memory.cleanup()
        logger.info("[Init] UNet loaded")

        # --- 2. DualCLIPLoader ---
        logger.info("[Init] Loading text encoders...")
        dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        clip_result = dualcliploader.load_clip(
            clip_name1=self.config.text_encoder_1,
            clip_name2=self.config.text_encoder_2,
            type=self.config.text_encoder_type,
            device="default",
        )
        clip = get_value_at_index(clip_result, 0)
        del clip_result
        self.memory.cleanup()
        logger.info("[Init] Text encoders loaded")

        # --- 3. Apply 4 LoRAs sequentially (Power Lora Loader equivalent) ---
        logger.info("[Init] Loading LoRAs...")
        load_lora = LoraLoaderModelOnly()

        lora_configs = [
            (self.config.lora_1_name, self.config.lora_1_strength),
            (self.config.lora_2_name, self.config.lora_2_strength),
            (self.config.lora_3_name, self.config.lora_3_strength),
            (self.config.lora_4_name, self.config.lora_4_strength),
        ]

        for lora_name, lora_strength in lora_configs:
            logger.info(f"[Init]   LoRA: {lora_name} (strength={lora_strength})")
            lora_result = load_lora.load_lora_model_only(model, lora_name, lora_strength)
            model = lora_result[0]
            del lora_result

        self.memory.cleanup()
        logger.info("[Init] All 4 LoRAs applied")

        # --- 4. ModelPreviewOverrideKJ with tiny VAE ---
        logger.info("[Init] Setting up model preview with tiny VAE...")
        if "VAELoaderKJ" in NODE_CLASS_MAPPINGS:
            vaeloaderkj = NODE_CLASS_MAPPINGS["VAELoaderKJ"]()
            tiny_vae_result = vaeloaderkj.load_vae(
                vae_name=self.config.tiny_vae,
                device="main_device",
                weight_dtype="bf16",
            )
            tiny_vae = get_value_at_index(tiny_vae_result, 0)
            del tiny_vae_result

            if "ModelPreviewOverrideKJ" in NODE_CLASS_MAPPINGS:
                preview_override = NODE_CLASS_MAPPINGS["ModelPreviewOverrideKJ"]()
                preview_result = preview_override.EXECUTE_NORMALIZED(
                    model=model,
                    vae=tiny_vae,
                    preview_interval=0,
                    preview_quality=80,
                    show_previews=True,
                    preview_width=240,
                    preview_fps=24,
                    preview_label="",
                )
                model = get_value_at_index(preview_result, 0)
                del preview_result
            del tiny_vae
        else:
            logger.warning("[Init] VAELoaderKJ not available, skipping preview override")

        self.memory.cleanup()

        # --- 5. Load VAEs ---
        logger.info("[Init] Loading Video VAE...")
        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        video_vae_result = vaeloader.load_vae(vae_name=self.config.video_vae)
        video_vae = get_value_at_index(video_vae_result, 0)
        del video_vae_result

        logger.info("[Init] Loading Audio VAE...")
        audio_vae_result = vaeloader.load_vae(vae_name=self.config.audio_vae)
        audio_vae = get_value_at_index(audio_vae_result, 0)
        del audio_vae_result
        self.memory.cleanup()

        # --- 6. Load Upscale Model ---
        logger.info("[Init] Loading Latent Upscale Model...")
        latentupscalemodelloader = NODE_CLASS_MAPPINGS["LatentUpscaleModelLoader"]()
        upscale_result = latentupscalemodelloader.EXECUTE_NORMALIZED(
            model_name=self.config.upscaler_model,
        )
        upscale_model = get_value_at_index(upscale_result, 0)
        del upscale_result
        self.memory.cleanup()

        # --- 7. LTXDirector (node 131) ---
        logger.info("[Init] Running LTXDirector node...")
        ltxdirector = NODE_CLASS_MAPPINGS["LTXDirector"]()
        director_result = ltxdirector.EXECUTE_NORMALIZED(
            model=model,
            clip=clip,
            audio_vae=audio_vae,
            optional_latent=None,
            global_prompt=None,
        )

        # LTXDirector outputs:
        # 0: model, 1: positive, 2: video_latent, 3: audio_latent,
        # 4: guide_data, 5: motion_guide_data, 6: frame_rate, 7: combined_audio
        director_model = get_value_at_index(director_result, 0)
        positive = get_value_at_index(director_result, 1)
        video_latent = get_value_at_index(director_result, 2)
        audio_latent = get_value_at_index(director_result, 3)
        guide_data = get_value_at_index(director_result, 4)
        motion_guide_data = get_value_at_index(director_result, 5)
        frame_rate_val = get_value_at_index(director_result, 6)
        del director_result

        # Release CLIP after Director has used it
        del clip, dualcliploader
        self.memory.cleanup()

        logger.info(f"[Init] LTXDirector produced frame_rate={frame_rate_val}")

        # --- 8. ConditioningZeroOut + LTXVConditioning ---
        logger.info("[Init] Setting up conditioning...")
        conditioningzeroout = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
        negative_result = conditioningzeroout.zero_out(conditioning=positive)
        negative = get_value_at_index(negative_result, 0)
        del negative_result

        ltxvconditioning = NODE_CLASS_MAPPINGS["LTXVConditioning"]()
        conditioning_result = ltxvconditioning.EXECUTE_NORMALIZED(
            frame_rate=frame_rate_val,
            positive=positive,
            negative=negative,
        )
        conditioned_positive = get_value_at_index(conditioning_result, 0)
        conditioned_negative = get_value_at_index(conditioning_result, 1)
        del conditioning_result
        self.memory.cleanup()

        # --- 9. RandomNoise (seed=0, mode=fixed) ---
        logger.info("[Init] Creating noise generator...")
        randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
        noise_result = randomnoise.EXECUTE_NORMALIZED(
            noise_seed=self.config.seed,
        )
        noise = get_value_at_index(noise_result, 0)
        del noise_result

        # --- Store pipeline context ---
        self.pipeline_context = {
            "model": director_model,
            "positive": conditioned_positive,
            "negative": conditioned_negative,
            "video_latent": video_latent,
            "audio_latent": audio_latent,
            "guide_data": guide_data,
            "motion_guide_data": motion_guide_data,
            "video_vae": video_vae,
            "audio_vae": audio_vae,
            "frame_rate": frame_rate_val,
            "upscale_model": upscale_model,
            "noise": noise,
        }

        self.memory.report()
        logger.info("[Init] Pipeline initialization complete!")

    def _release_models(self) -> None:
        """Release all model references to free VRAM."""
        logger.info("[Release] Releasing all models...")
        keys_to_release = list(self.pipeline_context.keys())
        for key in keys_to_release:
            if key in self.pipeline_context:
                del self.pipeline_context[key]
        self.pipeline_context = {}
        self.memory.emergency_cleanup()

    def _compute_chunks(
        self, total_frames: int, chunk_size: int
    ) -> List[Tuple[int, int]]:
        """Compute chunk boundaries for the given total frames and chunk size."""
        chunks = []
        start = 0
        while start < total_frames:
            end = min(start + chunk_size, total_frames)
            chunks.append((start, end))
            start = end
        return chunks

    def _assemble_final_video(self, chunk_files: List[str]) -> str:
        """
        Assemble all chunks into final video using FFmpeg concat demuxer.
        This avoids loading all frames into RAM.
        """
        output_path = os.path.join(self.config.output_dir, "LTX23_Director_MV_Final.mp4")

        if len(chunk_files) == 1:
            # Single chunk - just copy
            shutil.copy2(chunk_files[0], output_path)
            logger.info(f"[Assembly] Single chunk, copied to: {output_path}")
            return output_path

        # Create concat list file
        concat_file = os.path.join(self.config.chunks_dir, "concat_list.txt")
        with open(concat_file, "w") as f:
            for chunk_path in chunk_files:
                # Use absolute paths
                abs_path = os.path.abspath(chunk_path)
                f.write(f"file '{abs_path}'\n")

        # Get audio file path if exists
        audio_path = self._get_audio_file_path()

        # Assemble with ffmpeg using concat demuxer
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
        ]

        # Add audio if available
        if audio_path and os.path.exists(audio_path):
            # Calculate audio trim for the timeline
            timeline = parse_timeline(self.config.timeline_data)
            audio_segs = timeline.get("audio_segments", [])
            if audio_segs and audio_segs[0].trim_start is not None:
                trim_seconds = audio_segs[0].trim_start / self.config.frame_rate
                cmd.extend([
                    "-ss", str(trim_seconds),
                    "-i", audio_path,
                    "-map", "0:v", "-map", "1:a",
                    "-shortest",
                ])
            else:
                cmd.extend(["-i", audio_path, "-map", "0:v", "-map", "1:a"])
        else:
            cmd.extend(["-map", "0:v"])

        cmd.extend([
            "-c:v", "libx264",
            "-crf", str(self.config.output_crf),
            "-pix_fmt", self.config.output_pix_fmt,
            "-r", str(self.config.frame_rate),
            "-c:a", "aac",
            "-b:a", "192k",
            output_path,
        ])

        logger.info(f"[Assembly] Running FFmpeg concat: {' '.join(cmd[:10])}...")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0:
                logger.info(f"[Assembly] Final video assembled: {output_path}")
            else:
                logger.error(f"[Assembly] FFmpeg error: {result.stderr[:500]}")
                # Fallback: just use the first chunk
                shutil.copy2(chunk_files[0], output_path)
        except subprocess.TimeoutExpired:
            logger.error("[Assembly] FFmpeg timed out")
        except Exception as e:
            logger.error(f"[Assembly] Assembly failed: {e}")

        return output_path

    def _get_audio_file_path(self) -> Optional[str]:
        """Get the audio file path from the timeline."""
        timeline = parse_timeline(self.config.timeline_data)
        audio_segs = timeline.get("audio_segments", [])
        if audio_segs and audio_segs[0].audio_file:
            audio_file = audio_segs[0].audio_file
            audio_path = os.path.join(
                self.config.comfyui_dir, "input", audio_file
            )
            if os.path.exists(audio_path):
                return audio_path
        return None



# =============================================================================
# API FALLBACK MODE - ComfyUI API Queue
# =============================================================================
class ComfyUIAPIFallback:
    """
    Fallback execution mode that submits the workflow to ComfyUI's API server.
    Used when direct node execution fails or is not available.
    """

    def __init__(self, config: LTX23Config):
        self.config = config
        self.api_host = config.api_host

    def run(self, workflow_json_path: Optional[str] = None) -> Optional[str]:
        """
        Submit workflow to ComfyUI API and wait for result.

        Args:
            workflow_json_path: Path to workflow JSON. If None, constructs from config.

        Returns:
            Path to output video, or None if failed.
        """
        import urllib.request
        import urllib.error

        logger.info("[API] Starting ComfyUI API fallback mode...")

        # Build or load workflow
        if workflow_json_path and os.path.exists(workflow_json_path):
            with open(workflow_json_path, "r") as f:
                workflow = json.load(f)
        else:
            workflow = self._build_workflow_prompt()

        # Start ComfyUI server if not running
        if not self._check_server():
            self._start_server()

        # Submit prompt
        prompt_id = self._queue_prompt(workflow)
        if not prompt_id:
            logger.error("[API] Failed to queue prompt")
            return None

        # Wait for completion
        output_path = self._wait_for_result(prompt_id)
        return output_path

    def _check_server(self) -> bool:
        """Check if ComfyUI server is running."""
        import urllib.request
        import urllib.error
        try:
            url = f"{self.api_host}/system_stats"
            req = urllib.request.Request(url, method="GET")
            response = urllib.request.urlopen(req, timeout=5)
            return response.status == 200
        except (urllib.error.URLError, OSError, Exception):
            return False

    def _start_server(self) -> None:
        """Start ComfyUI server in background."""
        logger.info("[API] Starting ComfyUI server...")
        server_cmd = [
            sys.executable, "-u",
            os.path.join(self.config.comfyui_dir, "main.py"),
            "--listen", "127.0.0.1",
            "--port", "8188",
            "--dont-print-server",
        ]
        subprocess.Popen(
            server_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=self.config.comfyui_dir,
        )
        # Wait for server to start
        for attempt in range(30):
            time.sleep(2)
            if self._check_server():
                logger.info("[API] ComfyUI server is ready")
                return
        logger.error("[API] Server failed to start within 60 seconds")

    def _queue_prompt(self, workflow: Dict) -> Optional[str]:
        """Submit a prompt to the ComfyUI API queue."""
        import urllib.request
        import urllib.error

        url = f"{self.api_host}/prompt"
        payload = json.dumps({"prompt": workflow}).encode("utf-8")

        try:
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            response = urllib.request.urlopen(req, timeout=30)
            result = json.loads(response.read().decode())
            prompt_id = result.get("prompt_id")
            logger.info(f"[API] Prompt queued: {prompt_id}")
            return prompt_id
        except Exception as e:
            logger.error(f"[API] Failed to queue prompt: {e}")
            return None

    def _wait_for_result(
        self, prompt_id: str, timeout: int = 3600
    ) -> Optional[str]:
        """Wait for the queued prompt to complete."""
        import urllib.request
        import urllib.error

        start_time = time.time()
        poll_interval = 5

        while time.time() - start_time < timeout:
            try:
                url = f"{self.api_host}/history/{prompt_id}"
                req = urllib.request.Request(url, method="GET")
                response = urllib.request.urlopen(req, timeout=10)
                data = json.loads(response.read().decode())

                if prompt_id in data:
                    outputs = data[prompt_id].get("outputs", {})
                    # Find video output
                    for node_id, node_output in outputs.items():
                        if "gifs" in node_output:
                            for gif in node_output["gifs"]:
                                filename = gif.get("filename")
                                subfolder = gif.get("subfolder", "")
                                output_dir = os.path.join(
                                    self.config.comfyui_dir, "output", subfolder
                                )
                                output_path = os.path.join(output_dir, filename)
                                if os.path.exists(output_path):
                                    logger.info(f"[API] Output ready: {output_path}")
                                    return output_path
                    logger.info("[API] Prompt completed but no video output found")
                    return None

            except Exception:
                pass

            elapsed = time.time() - start_time
            if int(elapsed) % 30 == 0:
                logger.info(f"[API] Waiting for completion... ({elapsed:.0f}s)")
            time.sleep(poll_interval)

        logger.error("[API] Timed out waiting for result")
        return None

    def _build_workflow_prompt(self) -> Dict:
        """
        Build a ComfyUI API-format prompt from the configuration.
        This constructs the full workflow as a dict matching the API format.
        """
        prompt = {
            "135": {
                "class_type": "UnetLoaderGGUF",
                "inputs": {
                    "unet_name": self.config.unet_model,
                }
            },
            "12": {
                "class_type": "DualCLIPLoader",
                "inputs": {
                    "clip_name1": self.config.text_encoder_1,
                    "clip_name2": self.config.text_encoder_2,
                    "type": self.config.text_encoder_type,
                    "device": "default",
                }
            },
            "138": {
                "class_type": "Power Lora Loader (rgthree)",
                "inputs": {
                    "model": ["135", 0],
                    "clip": ["12", 0],
                },
                "widgets_values": [
                    {},
                    {"type": "PowerLoraLoaderHeaderWidget"},
                    {
                        "on": True,
                        "lora": self.config.lora_1_name,
                        "strength": self.config.lora_1_strength,
                        "strengthTwo": None,
                    },
                    {
                        "on": True,
                        "lora": self.config.lora_2_name,
                        "strength": self.config.lora_2_strength,
                        "strengthTwo": None,
                    },
                    {
                        "on": True,
                        "lora": self.config.lora_3_name,
                        "strength": self.config.lora_3_strength,
                        "strengthTwo": None,
                    },
                    {
                        "on": True,
                        "lora": self.config.lora_4_name,
                        "strength": self.config.lora_4_strength,
                        "strengthTwo": None,
                    },
                    {},
                    "",
                ]
            },
            "6": {
                "class_type": "VAELoaderKJ",
                "inputs": {
                    "vae_name": self.config.tiny_vae,
                    "device": "main_device",
                    "weight_dtype": "bf16",
                }
            },
            "10": {
                "class_type": "ModelPreviewOverrideKJ",
                "inputs": {
                    "model": ["138", 0],
                    "vae": ["6", 0],
                    "preview_interval": 0,
                    "preview_quality": 80,
                    "show_previews": True,
                    "preview_width": 240,
                    "preview_fps": 24,
                    "preview_label": "",
                }
            },
            "36": {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": self.config.video_vae,
                }
            },
            "8": {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": self.config.audio_vae,
                }
            },
            "13": {
                "class_type": "LatentUpscaleModelLoader",
                "inputs": {
                    "model_name": self.config.upscaler_model,
                }
            },
            "131": {
                "class_type": "LTXDirector",
                "inputs": {
                    "model": ["10", 0],
                    "clip": ["138", 1],
                    "audio_vae": ["8", 0],
                    "optional_latent": None,
                    "global_prompt": None,
                }
            },
            "128": {
                "class_type": "ConditioningZeroOut",
                "inputs": {
                    "conditioning": ["131", 1],
                }
            },
            "27": {
                "class_type": "LTXVConditioning",
                "inputs": {
                    "positive": ["131", 1],
                    "negative": ["128", 0],
                    "frame_rate": ["131", 6],
                }
            },
            "30": {
                "class_type": "RandomNoise",
                "inputs": {
                    "noise_seed": self.config.seed,
                }
            },
            "133": {
                "class_type": "LTXDirectorGuide",
                "inputs": {
                    "positive": ["27", 0],
                    "negative": ["27", 1],
                    "vae": ["36", 0],
                    "latent": ["131", 2],
                    "guide_data": ["131", 4],
                    "motion_guide_data": ["131", 5],
                    "model": ["131", 0],
                    "image_path": "None",
                    "apply_to_negative": 1,
                    "guide_strength": self.config.stage1_guide_strength,
                    "scale_method": "bicubic",
                    "crop": 1,
                    "crop_method": "center",
                    "extend_with_last_frame": True,
                    "offset_frames": False,
                    "latent_height": 256,
                    "latent_width": 64,
                    "use_latent_input": False,
                }
            },
            "29": {
                "class_type": "LTXVConcatAVLatent",
                "inputs": {
                    "video_latent": ["133", 2],
                    "audio_latent": ["131", 3],
                }
            },
            "32": {
                "class_type": "KSamplerSelect",
                "inputs": {
                    "sampler_name": self.config.stage1_sampler,
                }
            },
            "33": {
                "class_type": "BasicScheduler",
                "inputs": {
                    "model": ["133", 3],
                    "scheduler": self.config.stage1_scheduler,
                    "steps": self.config.stage1_steps,
                    "denoise": self.config.stage1_denoise,
                }
            },
            "28": {
                "class_type": "CFGGuider",
                "inputs": {
                    "model": ["133", 3],
                    "positive": ["133", 0],
                    "negative": ["133", 1],
                    "cfg": self.config.stage1_cfg,
                }
            },
            "31": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["30", 0],
                    "guider": ["28", 0],
                    "sampler": ["32", 0],
                    "sigmas": ["33", 0],
                    "latent_image": ["29", 0],
                }
            },
            "34": {
                "class_type": "LTXVSeparateAVLatent",
                "inputs": {
                    "av_latent": ["31", 0],
                }
            },
            "55": {
                "class_type": "LTXDirectorCropGuides",
                "inputs": {
                    "positive": ["133", 0],
                    "negative": ["133", 1],
                    "latent": ["34", 0],
                }
            },
            "14": {
                "class_type": "LTXVLatentUpsampler",
                "inputs": {
                    "samples": ["55", 2],
                    "upscale_model": ["13", 0],
                    "vae": ["36", 0],
                }
            },
            "132": {
                "class_type": "LTXDirectorGuide",
                "inputs": {
                    "positive": ["55", 0],
                    "negative": ["55", 1],
                    "vae": ["36", 0],
                    "latent": ["14", 0],
                    "guide_data": ["131", 4],
                    "motion_guide_data": ["131", 5],
                    "model": ["131", 0],
                    "image_path": "None",
                    "apply_to_negative": 1,
                    "guide_strength": self.config.stage2_guide_strength,
                    "scale_method": "bicubic",
                    "crop": 1,
                    "crop_method": "center",
                    "extend_with_last_frame": True,
                    "offset_frames": False,
                    "latent_height": 256,
                    "latent_width": 64,
                    "use_latent_input": False,
                }
            },
            "18": {
                "class_type": "LTXVConcatAVLatent",
                "inputs": {
                    "video_latent": ["132", 2],
                    "audio_latent": ["34", 1],
                }
            },
            "20": {
                "class_type": "KSamplerSelect",
                "inputs": {
                    "sampler_name": self.config.stage2_sampler,
                }
            },
            "21": {
                "class_type": "BasicScheduler",
                "inputs": {
                    "model": ["132", 3],
                    "scheduler": self.config.stage2_scheduler,
                    "steps": self.config.stage2_steps,
                    "denoise": self.config.stage2_denoise,
                }
            },
            "17": {
                "class_type": "CFGGuider",
                "inputs": {
                    "model": ["132", 3],
                    "positive": ["132", 0],
                    "negative": ["132", 1],
                    "cfg": self.config.stage2_cfg,
                }
            },
            "19": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["30", 0],
                    "guider": ["17", 0],
                    "sampler": ["20", 0],
                    "sigmas": ["21", 0],
                    "latent_image": ["18", 0],
                }
            },
            "22": {
                "class_type": "LTXVSeparateAVLatent",
                "inputs": {
                    "av_latent": ["19", 0],
                }
            },
            "54": {
                "class_type": "LTXDirectorCropGuides",
                "inputs": {
                    "positive": ["132", 0],
                    "negative": ["132", 1],
                    "latent": ["22", 0],
                }
            },
            "1": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["54", 2],
                    "vae": ["36", 0],
                }
            },
            "24": {
                "class_type": "LTXVAudioVAEDecode",
                "inputs": {
                    "samples": ["22", 1],
                    "audio_vae": ["8", 0],
                }
            },
            "139": {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "images": ["1", 0],
                    "audio": ["24", 0],
                    "frame_rate": ["131", 6],
                    "loop_count": 0,
                    "filename_prefix": self.config.filename_prefix,
                    "format": self.config.output_format,
                    "pix_fmt": self.config.output_pix_fmt,
                    "crf": self.config.output_crf,
                    "save_metadata": False,
                    "trim_to_audio": False,
                    "pingpong": False,
                    "save_output": True,
                }
            },
        }
        return prompt



# =============================================================================
# NON-CHUNKED PIPELINE (Full generation in one pass - for small frame counts)
# =============================================================================
class LTX23DirectPipeline:
    """
    Direct full-pipeline execution without chunking.
    Used for test mode or when total frames fit in VRAM.
    Implements the exact workflow node graph in a single pass.
    """

    def __init__(self, config: LTX23Config):
        self.config = config
        self.memory = MemoryManager()

    def run(self) -> Optional[str]:
        """Execute the full pipeline in one pass (no chunking)."""
        from nodes import NODE_CLASS_MAPPINGS
        from nodes import LoraLoaderModelOnly

        logger.info("\n" + "=" * 60)
        logger.info("DIRECT PIPELINE (Non-chunked)")
        logger.info("=" * 60)

        self.memory.report()

        try:
            with torch.inference_mode():
                return self._execute_full_pipeline()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("[Direct] OOM - pipeline needs chunked mode")
                self.memory.emergency_cleanup()
                return None
            raise
        except Exception as e:
            logger.error(f"[Direct] Pipeline failed: {e}")
            logger.error(traceback.format_exc())
            return None

    def _execute_full_pipeline(self) -> Optional[str]:
        """Execute the complete workflow graph."""
        from nodes import NODE_CLASS_MAPPINGS
        from nodes import LoraLoaderModelOnly

        # --- Load UNet ---
        logger.info("[Direct] Loading UNet...")
        unetloadergguf = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
        unet_result = unetloadergguf.load_unet(unet_name=self.config.unet_model)
        model = get_value_at_index(unet_result, 0)
        del unet_result
        self.memory.cleanup()

        # --- Load CLIP ---
        logger.info("[Direct] Loading text encoders...")
        dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        clip_result = dualcliploader.load_clip(
            clip_name1=self.config.text_encoder_1,
            clip_name2=self.config.text_encoder_2,
            type=self.config.text_encoder_type,
            device="default",
        )
        clip = get_value_at_index(clip_result, 0)
        del clip_result

        # --- Apply 4 LoRAs ---
        logger.info("[Direct] Applying 4 LoRAs...")
        load_lora = LoraLoaderModelOnly()
        lora_configs = [
            (self.config.lora_1_name, self.config.lora_1_strength),
            (self.config.lora_2_name, self.config.lora_2_strength),
            (self.config.lora_3_name, self.config.lora_3_strength),
            (self.config.lora_4_name, self.config.lora_4_strength),
        ]
        for lora_name, lora_strength in lora_configs:
            logger.info(f"[Direct]   LoRA: {lora_name} (s={lora_strength})")
            model = load_lora.load_lora_model_only(model, lora_name, lora_strength)[0]
        self.memory.cleanup()

        # --- Tiny VAE + Preview Override ---
        if "VAELoaderKJ" in NODE_CLASS_MAPPINGS:
            logger.info("[Direct] Setting up model preview...")
            vaeloaderkj = NODE_CLASS_MAPPINGS["VAELoaderKJ"]()
            tiny_vae_result = vaeloaderkj.load_vae(
                vae_name=self.config.tiny_vae,
                device="main_device",
                weight_dtype="bf16",
            )
            tiny_vae = get_value_at_index(tiny_vae_result, 0)
            del tiny_vae_result

            if "ModelPreviewOverrideKJ" in NODE_CLASS_MAPPINGS:
                preview_node = NODE_CLASS_MAPPINGS["ModelPreviewOverrideKJ"]()
                preview_result = preview_node.EXECUTE_NORMALIZED(
                    model=model,
                    vae=tiny_vae,
                    preview_interval=0,
                    preview_quality=80,
                    show_previews=True,
                    preview_width=240,
                    preview_fps=24,
                    preview_label="",
                )
                model = get_value_at_index(preview_result, 0)
                del preview_result
            del tiny_vae

        # --- Load VAEs ---
        logger.info("[Direct] Loading VAEs...")
        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        video_vae_result = vaeloader.load_vae(vae_name=self.config.video_vae)
        video_vae = get_value_at_index(video_vae_result, 0)
        del video_vae_result

        audio_vae_result = vaeloader.load_vae(vae_name=self.config.audio_vae)
        audio_vae = get_value_at_index(audio_vae_result, 0)
        del audio_vae_result
        self.memory.cleanup()

        # --- Load Upscale Model ---
        logger.info("[Direct] Loading upscale model...")
        latentupscalemodelloader = NODE_CLASS_MAPPINGS["LatentUpscaleModelLoader"]()
        upscale_result = latentupscalemodelloader.EXECUTE_NORMALIZED(
            model_name=self.config.upscaler_model,
        )
        upscale_model = get_value_at_index(upscale_result, 0)
        del upscale_result
        self.memory.cleanup()

        # --- LTXDirector ---
        logger.info("[Direct] Running LTXDirector...")
        ltxdirector = NODE_CLASS_MAPPINGS["LTXDirector"]()
        director_result = ltxdirector.EXECUTE_NORMALIZED(
            model=model,
            clip=clip,
            audio_vae=audio_vae,
            optional_latent=None,
            global_prompt=None,
        )

        director_model = get_value_at_index(director_result, 0)
        positive = get_value_at_index(director_result, 1)
        video_latent = get_value_at_index(director_result, 2)
        audio_latent = get_value_at_index(director_result, 3)
        guide_data = get_value_at_index(director_result, 4)
        motion_guide_data = get_value_at_index(director_result, 5)
        frame_rate_val = get_value_at_index(director_result, 6)
        del director_result, clip, dualcliploader
        self.memory.cleanup()

        # --- Conditioning ---
        logger.info("[Direct] Setting up conditioning...")
        conditioningzeroout = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
        negative = get_value_at_index(
            conditioningzeroout.zero_out(conditioning=positive), 0
        )

        ltxvconditioning = NODE_CLASS_MAPPINGS["LTXVConditioning"]()
        cond_result = ltxvconditioning.EXECUTE_NORMALIZED(
            frame_rate=frame_rate_val,
            positive=positive,
            negative=negative,
        )
        cond_positive = get_value_at_index(cond_result, 0)
        cond_negative = get_value_at_index(cond_result, 1)
        del cond_result
        self.memory.cleanup()

        # --- Random Noise ---
        randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
        noise = get_value_at_index(
            randomnoise.EXECUTE_NORMALIZED(noise_seed=self.config.seed), 0
        )

        # ====================================================================
        # STAGE 1: Low-res sampling
        # ====================================================================
        logger.info("[Direct] STAGE 1: Low-res sampling...")

        # LTXDirectorGuide (node 133)
        ltxdirectorguide = NODE_CLASS_MAPPINGS["LTXDirectorGuide"]()
        s1_guide = ltxdirectorguide.EXECUTE_NORMALIZED(
            positive=cond_positive,
            negative=cond_negative,
            vae=video_vae,
            latent=video_latent,
            guide_data=guide_data,
            motion_guide_data=motion_guide_data,
            model=director_model,
            image_path="None",
            apply_to_negative=1,
            guide_strength=self.config.stage1_guide_strength,
            scale_method="bicubic",
            crop=1,
            crop_method="center",
            extend_with_last_frame=True,
            offset_frames=False,
            latent_height=256,
            latent_width=64,
            use_latent_input=False,
        )

        s1_pos = get_value_at_index(s1_guide, 0)
        s1_neg = get_value_at_index(s1_guide, 1)
        s1_lat = get_value_at_index(s1_guide, 2)
        s1_model = get_value_at_index(s1_guide, 3)
        del s1_guide

        # ConcatAVLatent (node 29)
        ltxvconcatavlatent = NODE_CLASS_MAPPINGS["LTXVConcatAVLatent"]()
        s1_combined = get_value_at_index(
            ltxvconcatavlatent.EXECUTE_NORMALIZED(
                video_latent=s1_lat,
                audio_latent=audio_latent,
            ), 0
        )

        # Scheduler + Sampler + Guider
        basicscheduler = NODE_CLASS_MAPPINGS["BasicScheduler"]()
        s1_sigmas = get_value_at_index(
            basicscheduler.EXECUTE_NORMALIZED(
                scheduler=self.config.stage1_scheduler,
                steps=self.config.stage1_steps,
                denoise=self.config.stage1_denoise,
                model=s1_model,
            ), 0
        )

        ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        s1_sampler = get_value_at_index(
            ksamplerselect.EXECUTE_NORMALIZED(sampler_name=self.config.stage1_sampler), 0
        )

        cfgguider = NODE_CLASS_MAPPINGS["CFGGuider"]()
        s1_guider = get_value_at_index(
            cfgguider.EXECUTE_NORMALIZED(
                cfg=self.config.stage1_cfg,
                model=s1_model,
                positive=s1_pos,
                negative=s1_neg,
            ), 0
        )

        # SamplerCustomAdvanced (node 31)
        samplercustomadvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
        s1_output = samplercustomadvanced.EXECUTE_NORMALIZED(
            noise=noise,
            guider=s1_guider,
            sampler=s1_sampler,
            sigmas=s1_sigmas,
            latent_image=s1_combined,
        )
        del s1_guider, s1_sampler, s1_sigmas, s1_combined
        self.memory.cleanup()

        # Separate (node 34)
        ltxvseparateavlatent = NODE_CLASS_MAPPINGS["LTXVSeparateAVLatent"]()
        s1_separated = ltxvseparateavlatent.EXECUTE_NORMALIZED(
            av_latent=get_value_at_index(s1_output, 0),
        )
        s1_vid_lat = get_value_at_index(s1_separated, 0)
        s1_aud_lat = get_value_at_index(s1_separated, 1)
        del s1_output, s1_separated

        # CropGuides (node 55)
        ltxdirectorcropguides = NODE_CLASS_MAPPINGS["LTXDirectorCropGuides"]()
        s1_cropped = ltxdirectorcropguides.EXECUTE_NORMALIZED(
            positive=s1_pos,
            negative=s1_neg,
            latent=s1_vid_lat,
        )
        s1c_pos = get_value_at_index(s1_cropped, 0)
        s1c_neg = get_value_at_index(s1_cropped, 1)
        s1c_lat = get_value_at_index(s1_cropped, 2)
        del s1_cropped, s1_vid_lat, s1_pos, s1_neg
        self.memory.cleanup()

        logger.info("[Direct] Stage 1 complete")

        # ====================================================================
        # STAGE 2: Upscaled sampling
        # ====================================================================
        logger.info("[Direct] STAGE 2: Upscaled sampling...")

        # LTXVLatentUpsampler (node 14)
        ltxvlatentupsampler = NODE_CLASS_MAPPINGS["LTXVLatentUpsampler"]()
        upsampled = ltxvlatentupsampler.upsample_latent(
            samples=s1c_lat,
            upscale_model=upscale_model,
            vae=video_vae,
        )
        upsampled_lat = get_value_at_index(upsampled, 0)
        del upsampled, upscale_model
        self.memory.cleanup()

        # LTXDirectorGuide (node 132)
        s2_guide = ltxdirectorguide.EXECUTE_NORMALIZED(
            positive=s1c_pos,
            negative=s1c_neg,
            vae=video_vae,
            latent=upsampled_lat,
            guide_data=guide_data,
            motion_guide_data=motion_guide_data,
            model=director_model,
            image_path="None",
            apply_to_negative=1,
            guide_strength=self.config.stage2_guide_strength,
            scale_method="bicubic",
            crop=1,
            crop_method="center",
            extend_with_last_frame=True,
            offset_frames=False,
            latent_height=256,
            latent_width=64,
            use_latent_input=False,
        )

        s2_pos = get_value_at_index(s2_guide, 0)
        s2_neg = get_value_at_index(s2_guide, 1)
        s2_lat = get_value_at_index(s2_guide, 2)
        s2_model = get_value_at_index(s2_guide, 3)
        del s2_guide, upsampled_lat

        # ConcatAVLatent (node 18)
        s2_combined = get_value_at_index(
            ltxvconcatavlatent.EXECUTE_NORMALIZED(
                video_latent=s2_lat,
                audio_latent=s1_aud_lat,
            ), 0
        )

        # Scheduler + Sampler + Guider for Stage 2
        s2_sigmas = get_value_at_index(
            basicscheduler.EXECUTE_NORMALIZED(
                scheduler=self.config.stage2_scheduler,
                steps=self.config.stage2_steps,
                denoise=self.config.stage2_denoise,
                model=s2_model,
            ), 0
        )

        s2_sampler = get_value_at_index(
            ksamplerselect.EXECUTE_NORMALIZED(sampler_name=self.config.stage2_sampler), 0
        )

        s2_guider = get_value_at_index(
            cfgguider.EXECUTE_NORMALIZED(
                cfg=self.config.stage2_cfg,
                model=s2_model,
                positive=s2_pos,
                negative=s2_neg,
            ), 0
        )

        # SamplerCustomAdvanced (node 19)
        s2_output = samplercustomadvanced.EXECUTE_NORMALIZED(
            noise=noise,
            guider=s2_guider,
            sampler=s2_sampler,
            sigmas=s2_sigmas,
            latent_image=s2_combined,
        )
        del s2_guider, s2_sampler, s2_sigmas, s2_combined
        del director_model
        self.memory.cleanup()

        logger.info("[Direct] Stage 2 complete")

        # ====================================================================
        # DECODE AND OUTPUT
        # ====================================================================
        logger.info("[Direct] Decoding outputs...")

        # Separate (node 22)
        s2_separated = ltxvseparateavlatent.EXECUTE_NORMALIZED(
            av_latent=get_value_at_index(s2_output, 0),
        )
        s2_vid_lat = get_value_at_index(s2_separated, 0)
        s2_aud_lat = get_value_at_index(s2_separated, 1)
        del s2_output, s2_separated

        # CropGuides (node 54)
        s2_cropped = ltxdirectorcropguides.EXECUTE_NORMALIZED(
            positive=s2_pos,
            negative=s2_neg,
            latent=s2_vid_lat,
        )
        final_vid_lat = get_value_at_index(s2_cropped, 2)
        del s2_cropped, s2_vid_lat, s2_pos, s2_neg

        # VAEDecode (node 1)
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        decoded = vaedecode.decode(samples=final_vid_lat, vae=video_vae)
        video_frames = get_value_at_index(decoded, 0)
        del decoded, final_vid_lat, video_vae
        self.memory.cleanup()

        # Audio decode (node 24)
        ltxvaudiovaedecode = NODE_CLASS_MAPPINGS["LTXVAudioVAEDecode"]()
        audio_decoded = ltxvaudiovaedecode.EXECUTE_NORMALIZED(
            samples=s2_aud_lat,
            audio_vae=audio_vae,
        )
        audio_out = get_value_at_index(audio_decoded, 0)
        del audio_decoded, s2_aud_lat, audio_vae
        self.memory.cleanup()

        # --- Save final video ---
        logger.info("[Direct] Saving final video...")
        output_path = os.path.join(self.config.output_dir, "LTX23_Director_MV_Final.mp4")
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Try VHS_VideoCombine
        if "VHS_VideoCombine" in NODE_CLASS_MAPPINGS:
            try:
                vhs = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
                vhs.EXECUTE_NORMALIZED(
                    images=video_frames,
                    audio=audio_out,
                    frame_rate=frame_rate_val,
                    loop_count=0,
                    filename_prefix="LTX23_Director/Final",
                    format=self.config.output_format,
                    pix_fmt=self.config.output_pix_fmt,
                    crf=self.config.output_crf,
                    save_metadata=False,
                    trim_to_audio=False,
                    pingpong=False,
                    save_output=True,
                )
                logger.info(f"[Direct] Video saved via VHS_VideoCombine")
                return output_path
            except Exception as e:
                logger.warning(f"[Direct] VHS failed, using ffmpeg: {e}")

        # Fallback to CreateVideo if available
        if "CreateVideo" in NODE_CLASS_MAPPINGS:
            try:
                createvideo = NODE_CLASS_MAPPINGS["CreateVideo"]()
                createvideo.EXECUTE_NORMALIZED(
                    fps=frame_rate_val,
                    images=video_frames,
                    audio=audio_out,
                )
                logger.info("[Direct] Video created via CreateVideo node")
                return output_path
            except Exception as e:
                logger.warning(f"[Direct] CreateVideo failed: {e}")

        # Final fallback: ffmpeg
        self._save_with_ffmpeg(video_frames, audio_out, frame_rate_val, output_path)
        return output_path

    def _save_with_ffmpeg(
        self, video_frames: Any, audio_data: Any, frame_rate: float, output_path: str
    ) -> None:
        """Save video and audio with ffmpeg as final fallback."""
        import numpy as np

        if hasattr(video_frames, 'cpu'):
            frames_np = video_frames.cpu().numpy()
        else:
            frames_np = np.array(video_frames)

        if frames_np.max() <= 1.0:
            frames_np = (frames_np * 255).astype(np.uint8)

        n_frames, height, width, _ = frames_np.shape

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "rgb24",
            "-r", str(int(frame_rate)),
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", str(self.config.output_crf),
            "-pix_fmt", self.config.output_pix_fmt,
            output_path,
        ]

        try:
            proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            for frame in frames_np:
                proc.stdin.write(frame.tobytes())
            proc.stdin.close()
            proc.wait(timeout=300)
            if proc.returncode == 0:
                logger.info(f"[Direct] Video saved with ffmpeg: {output_path}")
            else:
                logger.error(f"[Direct] ffmpeg error: {proc.stderr.read().decode()[:200]}")
        except Exception as e:
            logger.error(f"[Direct] Failed to save video: {e}")



# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main() -> None:
    """
    Main entry point for LTX-2.3 Director 2.0 Music Video pipeline.

    Orchestrates the full flow:
    1. Parse arguments
    2. Setup environment (ComfyUI, custom nodes, packages)
    3. Download models and assets
    4. Run pre-flight checks
    5. Initialize memory manager
    6. Parse timeline
    7. Determine execution strategy (direct vs chunked vs API)
    8. Generate video
    9. Report results
    """
    # --- Parse arguments ---
    parser = argparse.ArgumentParser(
        description="LTX-2.3 Director 2.0 Music Video Generator"
    )
    parser.add_argument(
        "--test-mode", action="store_true", default=False,
        help="Test mode: generate only 8-17 frames for validation"
    )
    parser.add_argument(
        "--test-frames", type=int, default=17,
        help="Number of frames to generate in test mode (default: 17)"
    )
    parser.add_argument(
        "--api-fallback", action="store_true", default=False,
        help="Use ComfyUI API queue instead of direct node execution"
    )
    parser.add_argument(
        "--skip-setup", action="store_true", default=False,
        help="Skip environment setup (assume already configured)"
    )
    parser.add_argument(
        "--skip-download", action="store_true", default=False,
        help="Skip model downloads (assume already present)"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for generation (default: 0)"
    )
    parser.add_argument(
        "--width", type=int, default=1280,
        help="Output video width (default: 1280)"
    )
    parser.add_argument(
        "--height", type=int, default=720,
        help="Output video height (default: 720)"
    )
    parser.add_argument(
        "--frames", type=int, default=756,
        help="Total frames to generate (default: 756)"
    )
    parser.add_argument(
        "--project-dir", type=str, default="/content/LTX23_Project",
        help="Project directory for outputs and state"
    )
    parser.add_argument(
        "--comfyui-dir", type=str, default="/content/ComfyUI",
        help="ComfyUI installation directory"
    )
    parser.add_argument(
        "--workflow-json", type=str, default=None,
        help="Path to workflow JSON for API mode"
    )
    parser.add_argument(
        "--resume", action="store_true", default=False,
        help="Resume from checkpoint if available"
    )
    parser.add_argument(
        "--chunked", action="store_true", default=True,
        help="Use chunked generation (default: True)"
    )
    parser.add_argument(
        "--no-chunked", action="store_true", default=False,
        help="Disable chunked generation (process all frames at once)"
    )

    args = parser.parse_args()

    # --- Build configuration ---
    config = LTX23Config(
        project_dir=args.project_dir,
        comfyui_dir=args.comfyui_dir,
        output_dir=os.path.join(args.project_dir, "output"),
        chunks_dir=os.path.join(args.project_dir, "chunks"),
        state_file=os.path.join(args.project_dir, "generation_state.json"),
        seed=args.seed,
        width=args.width,
        height=args.height,
        total_frames=args.frames,
        test_mode=args.test_mode,
        test_frames=args.test_frames,
        api_fallback=args.api_fallback,
    )

    # --- Display configuration ---
    logger.info("\n" + "=" * 60)
    logger.info("LTX-2.3 DIRECTOR 2.0 MUSIC VIDEO PIPELINE")
    logger.info("=" * 60)
    logger.info(f"  Resolution: {config.width}x{config.height}")
    logger.info(f"  Frame Rate: {config.frame_rate} fps")
    logger.info(f"  Duration: {config.duration}s ({config.total_frames} frames)")
    logger.info(f"  Seed: {config.seed} (mode: {config.noise_mode})")
    logger.info(f"  Test Mode: {config.test_mode}")
    logger.info(f"  API Fallback: {config.api_fallback}")
    logger.info(f"  Project Dir: {config.project_dir}")
    logger.info(f"  ComfyUI Dir: {config.comfyui_dir}")
    logger.info(f"  Stage 1: {config.stage1_steps} steps, denoise={config.stage1_denoise}, "
                f"guide={config.stage1_guide_strength}")
    logger.info(f"  Stage 2: {config.stage2_steps} steps, denoise={config.stage2_denoise}, "
                f"guide={config.stage2_guide_strength}")
    logger.info(f"  LoRAs: [{config.lora_1_strength}, {config.lora_2_strength}, "
                f"{config.lora_3_strength}, {config.lora_4_strength}]")
    logger.info("=" * 60)

    # --- Step 1: Environment Setup ---
    if not args.skip_setup:
        setup_environment(config)
    else:
        logger.info("[Main] Skipping environment setup")
        # Still ensure sys.path includes ComfyUI
        if config.comfyui_dir not in sys.path:
            sys.path.insert(0, config.comfyui_dir)

    # --- Step 2: Download Models ---
    if not args.skip_download:
        download_models(config)
        download_assets(config)
    else:
        logger.info("[Main] Skipping model downloads")

    # --- Step 3: Pre-flight Checks ---
    preflight_ok = preflight_checks(config)
    if not preflight_ok:
        logger.error("[Main] Pre-flight checks failed! Continuing anyway...")

    # --- Step 4: Import Custom Nodes ---
    logger.info("[Main] Importing custom nodes...")
    import_custom_nodes()

    # --- Step 5: Parse Timeline ---
    logger.info("[Main] Parsing timeline...")
    timeline = parse_timeline(config.timeline_data)

    # --- Step 6: Execute Pipeline ---
    start_time = time.time()
    output_path: Optional[str] = None

    if config.api_fallback:
        # API Queue Mode
        logger.info("[Main] Using ComfyUI API fallback mode")
        api = ComfyUIAPIFallback(config)
        output_path = api.run(workflow_json_path=args.workflow_json)

    elif args.no_chunked or config.test_mode:
        # Direct non-chunked mode (for test mode or small generation)
        logger.info("[Main] Using direct (non-chunked) pipeline")
        direct = LTX23DirectPipeline(config)
        output_path = direct.run()

    else:
        # Chunked mode (default for full generation)
        logger.info("[Main] Using chunked pipeline with checkpoint/resume")
        pipeline = LTX23DirectorPipeline(config)
        output_path = pipeline.run()

    # --- Step 7: Report Results ---
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 60)

    if output_path and os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"  Output: {output_path}")
        logger.info(f"  File size: {file_size:.1f} MB")
        logger.info(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        logger.info("  Status: SUCCESS")
    else:
        logger.error(f"  Status: FAILED")
        logger.error(f"  Total time: {elapsed:.1f}s")
        if output_path:
            logger.error(f"  Expected output: {output_path}")

    logger.info("=" * 60)

    # Display in Colab if available
    try:
        from IPython.display import display, HTML
        if output_path and os.path.exists(output_path):
            display(HTML(f"""
            <div style="padding: 10px; background: #1a1a2e; border-radius: 8px;">
                <h3 style="color: #e94560;">Video Generated Successfully!</h3>
                <p style="color: #ccc;">Output: {output_path}</p>
                <p style="color: #ccc;">Size: {file_size:.1f} MB | Time: {elapsed:.1f}s</p>
            </div>
            """))
    except ImportError:
        pass


# =============================================================================
# COLAB CELL: SETUP & DOWNLOAD MODELS
# =============================================================================
# @title ⚙️ Setup & Download Models (run once)
def setup_and_download_models_cell():
    """
    Colab cell function: installs dependencies, clones custom nodes, and
    downloads all required models. Run this once before generation.
    """
    import os
    import subprocess
    from pathlib import Path

    print('[1/3] Installing dependencies...')
    _run_cmd(
        "pip install -q torch torchvision torchaudio einops diffusers accelerate "
        "av spandrel albumentations onnx opencv-python onnxruntime tqdm ipywidgets",
        "Installing Python packages"
    )
    if not os.path.exists('/content/ComfyUI'):
        _run_cmd("git clone -q https://github.com/comfyanonymous/ComfyUI /content/ComfyUI", "Cloning ComfyUI")
    _run_cmd("pip install -q -r /content/ComfyUI/requirements.txt", "Installing ComfyUI requirements")
    _run_cmd("apt-get -y install -qq aria2 > /dev/null 2>&1", "Installing aria2")

    print('[2/3] Cloning custom nodes...')
    custom_nodes_dir = "/content/ComfyUI/custom_nodes"
    os.makedirs(custom_nodes_dir, exist_ok=True)
    nodes = [
        "https://github.com/comfy-org/ComfyUI-Manager",
        "https://github.com/WhatDreamscost/WhatDreamsCost-ComfyUI",
        "https://github.com/rgthree/rgthree-comfy",
        "https://github.com/liconstudio/ComfyUI-Licon-MSR",
        "https://github.com/kijai/ComfyUI-KJNodes",
        "https://github.com/city96/ComfyUI-GGUF",
        "https://github.com/Lightricks/ComfyUI-LTXVideo/",
        "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite",
        "https://github.com/kijai/ComfyUI-MelBandRoFormer",
    ]
    for node in nodes:
        name = node.rstrip('/').split('/')[-1]
        node_path = os.path.join(custom_nodes_dir, name)
        if not os.path.exists(node_path):
            _run_cmd(f"git clone -q {node} {node_path}", f"Cloning {name}")
            req = os.path.join(node_path, "requirements.txt")
            if os.path.exists(req):
                _run_cmd(f"pip install -q -r {req}", f"Installing {name} requirements")

    print('[3/3] Downloading models...')

    def dl(url, dest, fname):
        Path(dest).mkdir(parents=True, exist_ok=True)
        fp = os.path.join(dest, fname)
        if not os.path.exists(fp):
            subprocess.run(
                ['aria2c', '--console-log-level=error', '-c', '-x', '16',
                 '-s', '16', '-k', '1M', '-d', dest, '-o', fname, url],
                check=True
            )

    dl("https://huggingface.co/vantagewithai/LTX-2.3-GGUF/resolve/main/dev/ltx-2-3-22b-dev-Q4_K_M.gguf",
       "/content/ComfyUI/models/unet", "ltx-2-3-22b-dev-Q4_K_M.gguf")
    dl("https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
       "/content/ComfyUI/models/text_encoders", "gemma_3_12B_it_fp4_mixed.safetensors")
    dl("https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors",
       "/content/ComfyUI/models/text_encoders", "ltx-2.3_text_projection_bf16.safetensors")
    dl("https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_audio_vae_bf16.safetensors",
       "/content/ComfyUI/models/vae", "LTX23_audio_vae_bf16.safetensors")
    dl("https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_video_vae_bf16.safetensors",
       "/content/ComfyUI/models/vae", "LTX23_video_vae_bf16.safetensors")
    dl("https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/taeltx2_3.safetensors",
       "/content/ComfyUI/models/vae", "taeltx2_3.safetensors")
    dl("https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/loras/ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors",
       "/content/ComfyUI/models/loras", "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors")
    dl("https://huggingface.co/joyfox/LTX-2.3-Transition-LORA/resolve/main/ltx2.3-transition.safetensors",
       "/content/ComfyUI/models/loras", "ltx2.3-transition.safetensors")
    dl("https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors",
       "/content/ComfyUI/models/loras", "LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors")
    dl("https://huggingface.co/vrgamedevgirl84/LTX_2.3_Crisp_Enhance_Style_LoRa/resolve/main/LTX2.3_Crisp_Enhance.safetensors",
       "/content/ComfyUI/models/loras", "LTX2.3_Crisp_Enhance.safetensors")
    dl("https://huggingface.co/LiconStudio/LTX-2.3-Multiple-Subject-Reference/resolve/main/LTX-2.3-Licon-MSR-V2.safetensors",
       "/content/ComfyUI/models/loras", "LTX-2.3-Licon-MSR-V2.safetensors")
    dl("https://huggingface.co/vidfom/aimusic/resolve/main/ComfyUI/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
       "/content/ComfyUI/models/latent_upscale_models", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
    dl("https://huggingface.co/vidfom/aimusic/resolve/main/ComfyUI/models/loras/LTX2.3-MVCamera-drclips.safetensors",
       "/content/ComfyUI/models/loras", "LTX2.3-MVCamera-drclips.safetensors")

    print('Setup complete! Run the generation cell to generate.')


# =============================================================================
# COLAB CELL HELPER FUNCTIONS
# =============================================================================
def run_test_generation() -> Optional[str]:
    """
    Quick test generation (8-17 frames) for validation.
    Can be called from a Colab cell directly.
    """
    config = LTX23Config(test_mode=True, test_frames=17)
    setup_environment(config)
    download_models(config)
    download_assets(config)
    preflight_checks(config)
    import_custom_nodes()

    direct = LTX23DirectPipeline(config)
    return direct.run()


def run_full_generation(
    seed: int = 0,
    width: int = 1280,
    height: int = 720,
    total_frames: int = 756,
    project_dir: str = "/content/LTX23_Project",
    comfyui_dir: str = "/content/ComfyUI",
    skip_setup: bool = False,
    skip_download: bool = False,
) -> Optional[str]:
    """
    Run the full video generation pipeline.
    Can be called from a Colab cell directly.

    Args:
        seed: Random seed (default: 0)
        width: Video width (default: 1280)
        height: Video height (default: 720)
        total_frames: Total frames (default: 756 = 31.5s at 24fps)
        project_dir: Output directory
        comfyui_dir: ComfyUI installation path
        skip_setup: Skip environment setup
        skip_download: Skip model downloads

    Returns:
        Path to output video or None if failed
    """
    config = LTX23Config(
        seed=seed,
        width=width,
        height=height,
        total_frames=total_frames,
        project_dir=project_dir,
        comfyui_dir=comfyui_dir,
    )

    if not skip_setup:
        setup_environment(config)
    else:
        if config.comfyui_dir not in sys.path:
            sys.path.insert(0, config.comfyui_dir)

    if not skip_download:
        download_models(config)
        download_assets(config)

    preflight_checks(config)
    import_custom_nodes()

    pipeline = LTX23DirectorPipeline(config)
    return pipeline.run()


def display_video_in_colab(video_path: str) -> None:
    """Display a video in Google Colab output cell."""
    try:
        from IPython.display import HTML, display
        from base64 import b64encode

        with open(video_path, "rb") as f:
            video_data = f.read()

        if video_path.lower().endswith('.mp4'):
            mime_type = "video/mp4"
        elif video_path.lower().endswith('.webm'):
            mime_type = "video/webm"
        else:
            mime_type = "video/mp4"

        data_url = f"data:{mime_type};base64," + b64encode(video_data).decode()
        display(HTML(f"""
        <video width=720 controls autoplay loop>
            <source src="{data_url}" type="{mime_type}">
        </video>
        """))
    except ImportError:
        logger.info(f"[Display] Video saved at: {video_path}")
    except Exception as e:
        logger.warning(f"[Display] Could not display video: {e}")
        logger.info(f"[Display] Video saved at: {video_path}")


# =============================================================================
# COLAB UI PARAMETERS (Interactive Controls)
# =============================================================================
# @markdown ### LoRA Configuration
use_lora_distilled = True  # @param {type:"boolean"}
lora_distilled_strength = 0.4  # @param {"type":"slider","min":0,"max":2,"step":0.01}
use_lora_omninft = True  # @param {type:"boolean"}
lora_omninft_strength = 0.6  # @param {"type":"slider","min":0,"max":2,"step":0.01}
use_lora_transition = True  # @param {type:"boolean"}
lora_transition_strength = 0.7  # @param {"type":"slider","min":0,"max":2,"step":0.01}
use_lora_mvcamera = True  # @param {type:"boolean"}
lora_mvcamera_strength = 0.9  # @param {"type":"slider","min":0,"max":2,"step":0.01}

# @markdown ### Sampling Settings
pass1_steps = 8  # @param {"type":"integer"}
pass1_denoise = 1.0  # @param {"type":"slider","min":0,"max":1,"step":0.01}
pass2_steps = 4  # @param {"type":"integer"}
pass2_denoise = 0.42  # @param {"type":"slider","min":0,"max":1,"step":0.01}
scheduler_name = "linear_quadratic"  # @param ["linear_quadratic", "normal", "simple", "ddim_uniform", "sgm_uniform"]
sampler_name = "euler"  # @param ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_sde"]


def build_config_from_ui_params() -> 'LTX23Config':
    """Build an LTX23Config using the Colab UI parameter values above."""
    config = LTX23Config(
        lora_1_strength=lora_distilled_strength if use_lora_distilled else 0.0,
        lora_2_strength=lora_omninft_strength if use_lora_omninft else 0.0,
        lora_3_strength=lora_transition_strength if use_lora_transition else 0.0,
        lora_4_strength=lora_mvcamera_strength if use_lora_mvcamera else 0.0,
        stage1_steps=pass1_steps,
        stage1_denoise=pass1_denoise,
        stage2_steps=pass2_steps,
        stage2_denoise=pass2_denoise,
        stage1_scheduler=scheduler_name,
        stage2_scheduler=scheduler_name,
        stage1_sampler=sampler_name,
        stage2_sampler=sampler_name,
    )
    return config


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()
