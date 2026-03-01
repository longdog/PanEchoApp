# PanEcho Video Inference App

This repository provides a Dockerized app that runs the PanEcho model on echocardiogram videos and writes a text report.

Project code is in `panecho/`.

## What the app does

- Accepts an input video file and preprocesses it with `ffmpeg` to PanEcho format.
- Uses PanEcho input settings from the original project:
  - shape: `1 x 3 x 16 x 224 x 224` (default clip length 16)
  - ImageNet normalization
- Runs inference and saves predictions to a text file.
- Supports demo mode (`--demo`) that runs the model on random input tensor, without video.
- Stores downloaded model weights in an external mounted volume (`/models`).

## Folder layout

- `panecho/` - original PanEcho code + app wrapper + Dockerfile
- `README.md` - this usage guide

## Build

```bash
docker build -t panecho-app -f panecho/Dockerfile panecho
```

## Run with video

```bash
docker run --rm \
  -v /path/to/local/models:/models \
  -v /path/to/local/output:/output \
  -v /path/to/local/videos:/videos:ro \
  panecho-app \
  --video /videos/echo.mp4 \
  --output /output/results.txt
```

## Run demo (no video)

```bash
docker run --rm \
  -v /path/to/local/models:/models \
  -v /path/to/local/output:/output \
  panecho-app \
  --demo \
  --output /output/results_demo.txt
```

## Output

The container writes a text report (default: `/output/results.txt`) containing all PanEcho task predictions.
