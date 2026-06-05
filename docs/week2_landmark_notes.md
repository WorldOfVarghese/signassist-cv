# Week 2 Landmark Notes

## What I learned

MediaPipe Hands detects 21 landmarks on a hand.

Each landmark has:

- x coordinate
- y coordinate
- z coordinate

The x and y coordinates are normalized, meaning they are usually between 0 and 1 relative to the image size.

## Landmark map

- 0: wrist
- 1-4: thumb
- 5-8: index finger
- 9-12: middle finger
- 13-16: ring finger
- 17-20: pinky finger

## Why landmarks matter

Instead of training a model directly on raw webcam images, I can use hand landmarks as structured features.

For a beginner gesture recognition project, this is easier and more reliable than using raw pixels.

## Pixel conversion

MediaPipe gives normalized coordinates.

To draw them on an OpenCV image, I convert them into pixel coordinates:

```python
pixel_x = int(landmark.x * width)
pixel_y = int(landmark.y * height)