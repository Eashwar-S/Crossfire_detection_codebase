# VLM Integration Branch

This branch adds the initial logic for Vision-Language Model (VLM) integration within the DJI detection pipeline.

## Purpose
- Track YOLO detections across frames.
- After 10 consecutive positive YOLO detections, capture the last 10 RGB annotated frames.
- Send the frames to a placeholder VLM function and wait for a simulated response.

## Status
- VLM function is currently a placeholder and returns a mock response.
- Logging, detection pipeline, and MQTT logic remain unchanged.