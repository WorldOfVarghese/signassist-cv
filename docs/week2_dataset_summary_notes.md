# Week 2 Dataset Summary Notes

## Purpose

Today I collected a small starter dataset for the SignAssist CV project.

This dataset is not final. It is only a test dataset to confirm that the data collection pipeline works.

## Gesture labels

The current gesture labels are:

- open_palm
- closed_fist
- point_index
- peace_sign
- thumbs_up

## Target sample count

Starter target:

- 10 samples per gesture
- 5 gestures total
- 50 samples total

## Dataset format

The CSV file contains:

- 1 label column
- 21 hand landmarks
- 3 values per landmark: x, y, z

Total columns:

1 + (21 × 3) = 64 columns

## Files used

Data collection script:

```text
src/collect_landmark_data.py