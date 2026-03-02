#!/usr/bin/env bash

sudo docker run --rm \
  -v ./models:/models \
  -v ./output:/output \
  -v ./video:/videos:ro \
  panecho-app \
  --video /videos/video.mp4 \
  --output /output/results.txt
