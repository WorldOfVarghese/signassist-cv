# Week 2 Project Status

## Current phase

The SignAssist CV project is currently in the prototype and data pipeline phase.

The project can:

- open the webcam
- detect one hand
- draw hand landmarks
- label landmarks by number
- print landmark coordinates
- collect labeled landmark samples
- save samples to a local CSV file
- summarize the local dataset

## Working scripts

### setup_test.py

Confirms that the Python environment and required packages work.

### webcam_test.py

Confirms that OpenCV can access the webcam.

### hand_landmark_test.py

Confirms that MediaPipe can detect hands and draw landmarks.

### landmark_inspector.py

Helps inspect the 21 MediaPipe hand landmarks.

### collect_landmark_data.py

Collects labeled landmark data for five gestures.

### dataset_summary.py

Summarizes the local CSV dataset and checks sample counts.

## Current gesture labels

- open_palm
- closed_fist
- point_index
- peace_sign
- thumbs_up

## Current dataset status

A small starter dataset was collected locally.

This dataset is not final.

The CSV is not pushed to GitHub yet because it may contain messy samples and uneven label counts.

## What I learned this week

- MediaPipe gives 21 hand landmarks.
- Each landmark has x, y, and z values.
- These landmarks can become ML features.
- A gesture sample can be represented as 63 numbers.
- Dataset balance matters.
- Data should be checked before training.
- Generated data should not be uploaded accidentally.

## Current pipeline

```text
webcam → OpenCV frame → MediaPipe Hands → 21 landmarks → CSV row → future classifier