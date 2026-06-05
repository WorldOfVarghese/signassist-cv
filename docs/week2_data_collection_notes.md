# Week 2 Data Collection Notes

## Purpose

The goal of the data collection script is to collect labeled hand landmark data for the SignAssist CV gesture recognition project.

Instead of saving full webcam images, the script saves MediaPipe hand landmark coordinates.

## Gesture labels

The current planned labels are:

- 1: open_palm
- 2: closed_fist
- 3: point_index
- 4: peace_sign
- 5: thumbs_up

## What each sample contains

Each row in the CSV contains:

- 1 label
- 21 hand landmarks
- 3 values per landmark: x, y, z

Total feature values:

21 × 3 = 63

So one row contains:

label + 63 numeric features

## Why landmark data is useful

Using hand landmarks is simpler than training directly on raw images.

The project pipeline is:

webcam → MediaPipe hand detection → landmark coordinates → classifier → gesture label

## CSV output

The script saves data to:

```text
data/gesture_landmarks.csv