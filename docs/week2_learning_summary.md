# Week 2 Learning Summary

## Main goal

The goal of Week 2 was to turn SignAssist CV from an empty project repo into a working computer vision prototype.

## What I built

This week I built:

- webcam test script
- MediaPipe hand landmark detection script
- landmark inspector script
- data collection script
- dataset summary script
- project documentation

## Current project pipeline

```text
webcam → OpenCV frame → MediaPipe Hands → 21 hand landmarks → 63 landmark features → CSV dataset → future classifier