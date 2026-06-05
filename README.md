# SignAssist CV

A beginner computer vision project that recognizes simple static hand gestures using webcam input, MediaPipe hand landmarks, and a lightweight machine learning classifier.

## Project goal

Build a webcam-based demo that recognizes 5 simple static hand gestures.

Planned gestures:

1. Open palm
2. Closed fist
3. Pointing index finger
4. Peace sign
5. Thumbs up

## Important note

This is not a full sign-language translator.

This is a beginner gesture recognition project connected to accessibility, human-computer interaction, and human-robot interaction learning.

## Why this project matters

This project helps me connect my accessibility/HCI/sign-language research background with real computer vision engineering skills.

The technical goal is to learn how to build a basic perception pipeline:

```text
webcam → MediaPipe hand landmarks → landmark features → classifier → gesture label