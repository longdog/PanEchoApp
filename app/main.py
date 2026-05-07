#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image

from panecho.src.models import FrameTransformer, MultiTaskModel


PANECHO_WEIGHTS_URL = "https://github.com/CarDS-Yale/PanEcho/releases/download/v1.0/panecho.pt"

logger = logging.getLogger("panecho")


@dataclass
class Task:
    task_name: str
    task_type: str
    class_names: np.ndarray
    mean: float = np.nan


def run_cmd(cmd: List[str]) -> str:
    proc = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.stdout.strip()


def is_dicom_file(path: Path) -> bool:
    proc = subprocess.run(
        ["file", "-b", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        return False
    return "dicom medical imaging data" in proc.stdout.lower()


def dicom_dump(path: Path) -> str:
    proc = subprocess.run(
        ["dcmdump", str(path)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.stdout


def _first_ds_value_in_line(line: str) -> Optional[float]:
    m = re.search(r"\[([0-9.]+(?:[eE][+-]?[0-9]+)?)\]", line)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def parse_dicom_fps(dump_text: str) -> Optional[float]:
    """Return nominal FPS from DICOM dump when a standard timing tag is present."""
    for line in dump_text.splitlines():
        if "(0018,0040)" in re.sub(r"\s+", "", line):
            v = _first_ds_value_in_line(line)
            if v is not None and v > 0:
                return v
    for line in dump_text.splitlines():
        if "(0018,1063)" in re.sub(r"\s+", "", line):
            v = _first_ds_value_in_line(line)
            if v is not None and v > 0:
                return 1000.0 / v
    key_fps = re.compile(
        r"cine\s*rate|frames\s*/\s*sec|\bfps\b|recommended\s*frame",
        re.IGNORECASE,
    )
    key_ms = re.compile(r"frame\s*time|frame\s*delay", re.IGNORECASE)
    for line in dump_text.splitlines():
        if key_fps.search(line):
            v = _first_ds_value_in_line(line)
            if v is not None and v > 0:
                return v
    for line in dump_text.splitlines():
        if key_ms.search(line):
            v = _first_ds_value_in_line(line)
            if v is not None and v > 0:
                return 1000.0 / v
    return None


def _sorted_dcm2img_pngs(directory: Path, basename: str) -> List[Path]:
    patterns = [
        # dcm2img +on often emits basename.<frame>.png (e.g. frame.1398.png)
        re.compile(re.escape(basename) + r"\.(\d+)\.png$", re.IGNORECASE),
        re.compile(re.escape(basename) + r"_\.(\d+)\.png$", re.IGNORECASE),
        re.compile(re.escape(basename) + r"_(\d+)\.png$", re.IGNORECASE),
    ]
    pairs: List[tuple[int, Path]] = []
    for p in directory.iterdir():
        if not p.is_file() or p.suffix.lower() != ".png":
            continue
        for pat in patterns:
            m = pat.match(p.name)
            if m:
                pairs.append((int(m.group(1)), p))
                break
    pairs.sort(key=lambda t: t[0])
    return [p for _, p in pairs]


def _uniform_sample_frame_indices(n_frames: int, clip_len: int) -> np.ndarray:
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    raw = (np.arange(clip_len, dtype=np.float64) + 0.5) * (n_frames / clip_len)
    idx = np.clip(np.round(raw).astype(np.int64), 0, n_frames - 1)
    return idx


def _paths_for_dicom_frames(all_paths: Sequence[Path], clip_len: int) -> List[Path]:
    n = len(all_paths)
    if n == 0:
        return []
    idx = _uniform_sample_frame_indices(n, clip_len)
    return [all_paths[i] for i in idx]


def frame_paths_to_tensor(frame_paths: Sequence[Path], clip_len: int) -> torch.Tensor:
    """Resize to 256, center-crop 224 to match ffmpeg scale/crop preprocessing."""
    if not frame_paths:
        raise RuntimeError("No frame paths provided for tensor conversion")

    frames: List[np.ndarray] = []
    for p in frame_paths:
        img = Image.open(p).convert("RGB")
        img = img.resize((256, 256), Image.Resampling.BILINEAR)
        w, h = img.size
        l = (w - 224) // 2
        t = (h - 224) // 2
        img = img.crop((l, t, l + 224, t + 224))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        frames.append(arr)

    if len(frames) < clip_len:
        frames.extend([frames[-1]] * (clip_len - len(frames)))
    elif len(frames) > clip_len:
        frames = frames[:clip_len]

    arr = np.stack(frames, axis=0)
    arr = np.transpose(arr, (3, 0, 1, 2))
    x = torch.from_numpy(arr).unsqueeze(0)

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype).view(1, 3, 1, 1, 1)
    return (x - mean) / std


def preprocess_dicom_video(video_path: Path, clip_len: int) -> torch.Tensor:
    logger.info("Reading DICOM metadata (dcmdump) for timing tags…")
    dump_text = dicom_dump(video_path)
    fps = parse_dicom_fps(dump_text)
    if fps is not None:
        logger.info("Detected nominal frame rate from DICOM: %.4f fps", fps)
    else:
        logger.info(
            "No standard timing tag produced a frame rate; using uniform spacing across extracted frames"
        )

    with tempfile.TemporaryDirectory(prefix="panecho_dicom_") as tmpdir:
        tmp = Path(tmpdir)
        prefix_name = "frame"
        logger.info(
            "Converting DICOM to PNG frames (dcm2img +Fa +on), output basename %r …",
            prefix_name,
        )
        subprocess.run(
            ["dcm2img", "+Fa", "+on", str(video_path.resolve()), prefix_name],
            check=True,
            cwd=str(tmp),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        all_pngs = _sorted_dcm2img_pngs(tmp, basename=prefix_name)
        if not all_pngs:
            hint = f" Nominal FPS from DICOM metadata: {fps:.4g}." if fps else ""
            raise RuntimeError(
                f"dcm2img produced no PNG frames for DICOM input: {video_path}.{hint} "
                "Check that the file is a supported multi-frame image object."
            )
        logger.info(
            "Extracted %d frame image(s) from DICOM; selecting %d frames for model input "
            "(uniform temporal indices)",
            len(all_pngs),
            clip_len,
        )
        selected = _paths_for_dicom_frames(all_pngs, clip_len)
        tensor = frame_paths_to_tensor(selected, clip_len)
        logger.info("Preprocessed DICOM tensor shape: %s", tuple(tensor.shape))
        return tensor


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


def read_video_stream_fps(video_path: Path) -> Optional[float]:
    out = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate",
            "-of",
            "json",
            str(video_path),
        ]
    )
    data = json.loads(out)
    streams = data.get("streams") or []
    if not streams:
        return None

    def parse_rate(s: Optional[str]) -> Optional[float]:
        if not s or s in ("0/0", "N/A"):
            return None
        if "/" in s:
            num_s, den_s = s.split("/", 1)
            den = float(den_s)
            if den == 0:
                return None
            return float(num_s) / den
        try:
            return float(s)
        except ValueError:
            return None

    s0 = streams[0]
    for key in ("avg_frame_rate", "r_frame_rate"):
        fps = parse_rate(s0.get(key))
        if fps is not None and fps > 0:
            return fps
    return None


def preprocess_video_with_ffmpeg(video_path: Path, clip_len: int) -> torch.Tensor:
    duration = read_video_duration(video_path)
    stream_fps = read_video_stream_fps(video_path)
    target_fps = clip_len / duration

    if stream_fps is not None:
        logger.info(
            "Detected video stream frame rate: %.4f fps (ffprobe); duration: %.4f s",
            stream_fps,
            duration,
        )
    else:
        logger.info(
            "Could not detect video stream frame rate (ffprobe); duration: %.4f s",
            duration,
        )
    logger.info(
        "Converting video with ffmpeg: sampling %.6f fps to produce %d frames "
        "(uniform spread over clip), scale to 256px then center-crop 224px",
        target_fps,
        clip_len,
    )

    with tempfile.TemporaryDirectory(prefix="panecho_frames_") as tmpdir:
        frame_pattern = str(Path(tmpdir) / "frame_%05d.png")
        vf = f"fps={target_fps:.8f},scale=256:256"
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

        logger.info("ffmpeg extracted %d frame image(s); building normalized input tensor", len(frame_paths))
        tensor = frame_paths_to_tensor(frame_paths, clip_len)
        logger.info("Preprocessed video tensor shape: %s", tuple(tensor.shape))
        return tensor


def preprocess_video(video_path: Path, clip_len: int) -> torch.Tensor:
    if is_dicom_file(video_path):
        logger.info("Input detected as DICOM medical imaging: %s", video_path)
        return preprocess_dicom_video(video_path, clip_len)
    logger.info("Input detected as regular video file: %s", video_path)
    return preprocess_video_with_ffmpeg(video_path, clip_len)


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
        default=Path("panecho/content/tasks.pkl"),
        help="Path to local PanEcho tasks.pkl file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        force=True,
    )
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
        logger.info("Demo mode: random tensor input (clip_len=%d), device=%s", args.clip_len, device)
        model = build_model(task_dict, clip_len=args.clip_len, model_dir=args.model_dir, device=device)
        x = torch.rand(1, 3, args.clip_len, 224, 224)
        logger.info("Starting model inference …")
        outputs = infer(model, x, device)
        logger.info("Inference finished.")
        report = format_results(task_dict, outputs, mode="demo", video_path="")
    else:
        logger.info(
            "Starting preprocessing: clip_len=%d, device=%s, input=%s",
            args.clip_len,
            device,
            args.video,
        )
        x = preprocess_video(args.video, clip_len=args.clip_len)
        model = build_model(task_dict, clip_len=args.clip_len, model_dir=args.model_dir, device=device)
        logger.info("Starting model inference …")
        outputs = infer(model, x, device)
        logger.info("Inference finished.")
        report = format_results(task_dict, outputs, mode="inference", video_path=str(args.video))

    args.output.write_text(report, encoding="utf-8")
    print(f"Done. Results written to: {args.output}")


if __name__ == "__main__":
    main()
