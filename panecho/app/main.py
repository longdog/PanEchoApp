#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.models import FrameTransformer, MultiTaskModel


PANECHO_WEIGHTS_URL = "https://github.com/CarDS-Yale/PanEcho/releases/download/v1.0/panecho.pt"


@dataclass
class Task:
    task_name: str
    task_type: str
    class_names: np.ndarray
    mean: float = np.nan


def run_cmd(cmd: List[str]) -> str:
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return proc.stdout.strip()


def read_video_duration(video_path: Path) -> float:
    out = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video_path),
        ]
    )
    data = json.loads(out)
    duration = float(data["format"]["duration"])
    if duration <= 0:
        raise RuntimeError(f"Could not determine valid video duration for: {video_path}")
    return duration


def preprocess_video_with_ffmpeg(video_path: Path, clip_len: int) -> torch.Tensor:
    duration = read_video_duration(video_path)
    target_fps = clip_len / duration

    with tempfile.TemporaryDirectory(prefix="panecho_frames_") as tmpdir:
        frame_pattern = str(Path(tmpdir) / "frame_%05d.png")
        vf = f"fps={target_fps:.8f},scale=256:256,crop=224:224"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video_path),
                "-vf",
                vf,
                "-frames:v",
                str(clip_len),
                frame_pattern,
            ],
            check=True,
        )

        frame_paths = sorted(Path(tmpdir).glob("frame_*.png"))
        if not frame_paths:
            raise RuntimeError(f"No frames were extracted from: {video_path}")

        frames = []
        for p in frame_paths:
            img = Image.open(p).convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            frames.append(arr)

        if len(frames) < clip_len:
            frames.extend([frames[-1]] * (clip_len - len(frames)))
        elif len(frames) > clip_len:
            frames = frames[:clip_len]

    arr = np.stack(frames, axis=0)  # L x H x W x C
    arr = np.transpose(arr, (3, 0, 1, 2))  # C x L x H x W
    x = torch.from_numpy(arr).unsqueeze(0)  # 1 x C x L x H x W

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype).view(1, 3, 1, 1, 1)
    x = (x - mean) / std

    return x


def load_tasks(tasks_path: Path) -> Dict[str, dict]:
    if not tasks_path.exists():
        raise FileNotFoundError(f"Could not find tasks file: {tasks_path}")
    return pd.read_pickle(tasks_path)


def build_model(task_dict: Dict[str, dict], clip_len: int, model_dir: Path, device: torch.device) -> MultiTaskModel:
    tasks = [
        Task(
            task_name=name,
            task_type=spec["task_type"],
            class_names=spec["class_names"],
            mean=spec.get("mean", np.nan),
        )
        for name, spec in task_dict.items()
    ]

    encoder = FrameTransformer(
        arch="convnext_tiny",
        n_heads=8,
        n_layers=4,
        transformer_dropout=0.0,
        pooling="mean",
        clip_len=clip_len,
    )
    model = MultiTaskModel(
        encoder=encoder,
        encoder_dim=encoder.encoder.n_features,
        tasks=tasks,
        fc_dropout=0.25,
        activations=True,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    weights_path = model_dir / "panecho.pt"
    if not weights_path.exists():
        torch.hub.download_url_to_file(PANECHO_WEIGHTS_URL, str(weights_path), progress=True)

    checkpoint = torch.load(weights_path, map_location="cpu")
    weights = checkpoint["weights"] if isinstance(checkpoint, dict) and "weights" in checkpoint else checkpoint
    weights.pop("encoder.time_encoder.pe", None)
    model.load_state_dict(weights, strict=False)

    model.eval().to(device)
    return model


def infer(model: MultiTaskModel, x: torch.Tensor, device: torch.device) -> Dict[str, np.ndarray]:
    with torch.no_grad():
        out = model(x.to(device))
    return {k: v.detach().cpu().numpy() for k, v in out.items()}


def format_results(task_dict: Dict[str, dict], outputs: Dict[str, np.ndarray], mode: str, video_path: str) -> str:
    lines = []
    lines.append(f"PanEcho Results")
    lines.append(f"timestamp_utc: {dt.datetime.utcnow().isoformat()}Z")
    lines.append(f"mode: {mode}")
    lines.append(f"video: {video_path if video_path else 'N/A'}")
    lines.append("")
    lines.append("predictions:")

    for task_name, spec in task_dict.items():
        task_type = spec["task_type"]
        values = outputs[task_name]

        if task_type == "multi-class_classification":
            probs = values[0].astype(float)
            class_names = [str(c) for c in spec["class_names"]]
            top_idx = int(np.argmax(probs))
            top_class = class_names[top_idx]
            top_prob = probs[top_idx]
            probs_str = ", ".join(f"{c}={p:.6f}" for c, p in zip(class_names, probs))
            lines.append(f"{task_name}: top_class={top_class}, top_prob={top_prob:.6f}, probs=[{probs_str}]")
        else:
            lines.append(f"{task_name}: {float(values[0][0]):.6f}")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PanEcho inference on echocardiogram video.")
    parser.add_argument("--video", type=Path, help="Path to input video file.")
    parser.add_argument("--demo", action="store_true", help="Demo mode: run model on random tensor input (no video).")
    parser.add_argument("--output", type=Path, default=Path("/output/results.txt"), help="Output text file path.")
    parser.add_argument("--model-dir", type=Path, default=Path("/models"), help="Directory for model weights/cache.")
    parser.add_argument("--clip-len", type=int, default=16, help="PanEcho clip length.")
    parser.add_argument(
        "--tasks-path",
        type=Path,
        default=Path("content/tasks.pkl"),
        help="Path to local PanEcho tasks.pkl file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.demo and args.video:
        raise SystemExit("Use either --video or --demo, not both.")
    if not args.demo and not args.video:
        raise SystemExit("Provide --video <path> or use --demo.")
    if args.video and not args.video.exists():
        raise SystemExit(f"Video does not exist: {args.video}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(args.model_dir / "torch_home"))

    task_dict = load_tasks(args.tasks_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.demo:
        model = build_model(task_dict, clip_len=args.clip_len, model_dir=args.model_dir, device=device)
        x = torch.rand(1, 3, args.clip_len, 224, 224)
        outputs = infer(model, x, device)
        report = format_results(task_dict, outputs, mode="demo", video_path="")
    else:
        x = preprocess_video_with_ffmpeg(args.video, clip_len=args.clip_len)
        model = build_model(task_dict, clip_len=args.clip_len, model_dir=args.model_dir, device=device)
        outputs = infer(model, x, device)
        report = format_results(task_dict, outputs, mode="inference", video_path=str(args.video))

    args.output.write_text(report, encoding="utf-8")
    print(f"Done. Results written to: {args.output}")


if __name__ == "__main__":
    main()
