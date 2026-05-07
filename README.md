# PanEcho Video Inference App

This repository provides a Dockerized app that runs the PanEcho model on echocardiogram videos and writes a text report.

App code is in `app/`. PanEcho source/model code is in `panecho/`.

## What the app does

- Accepts an input file via `--video`, preprocesses it to PanEcho format, runs inference, and writes a text report.
- Model input shape: `1 x 3 x 16 x 224 x 224` by default (16 frames; configurable with `--clip-len`), with ImageNet normalization.
- Supports demo mode (`--demo`) with random input (no file).
- Stores downloaded model weights in a mounted volume (`/models` by default).

## Supported input formats

- **Video**: common formats such as **MP4** and **AVI**.
- **DICOM**: a single **multiframe** DICOM file (for example echocardiography loops). Pass it with `--video` the same way as a video file.

## Folder layout

- `app/` - CLI app entrypoint and Python requirements
- `panecho/` - original PanEcho model/code/assets
- `Dockerfile` - container definition for this app
- `README.md` - this usage guide

## Build

```bash
docker build -t panecho-app .
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
