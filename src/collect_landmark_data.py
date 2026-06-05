# Week 2 Day 4: Landmark data collection for SignAssist CV
# Goal:
# 1. Open webcam
# 2. Detect one hand with MediaPipe
# 3. Press keys 1-5 to save labeled landmark samples
# 4. Save data to data/gesture_landmarks.csv
# 5. Press q to quit

import csv
import os

import cv2
import mediapipe as mp


GESTURE_LABELS = {
    ord("1"): "open_palm",
    ord("2"): "closed_fist",
    ord("3"): "point_index",
    ord("4"): "peace_sign",
    ord("5"): "thumbs_up",
}


OUTPUT_CSV = "data/gesture_landmarks.csv"


def create_csv_if_needed(file_path):
    """Create the CSV file with a header row if it does not already exist."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if os.path.exists(file_path):
        return

    header = ["label"]

    for landmark_index in range(21):
        header.append(f"x{landmark_index}")
        header.append(f"y{landmark_index}")
        header.append(f"z{landmark_index}")

    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)


def extract_landmark_features(hand_landmarks):
    """Convert 21 hand landmarks into a flat list of 63 values."""
    features = []

    for landmark in hand_landmarks.landmark:
        features.append(landmark.x)
        features.append(landmark.y)
        features.append(landmark.z)

    return features


def save_sample(file_path, label, features):
    """Save one labeled sample to the CSV file."""
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([label] + features)


def draw_instructions(frame, sample_count):
    """Draw instructions on the webcam frame."""
    instructions = [
        "Press 1 = open_palm",
        "Press 2 = closed_fist",
        "Press 3 = point_index",
        "Press 4 = peace_sign",
        "Press 5 = thumbs_up",
        "Press q = quit",
        f"Samples saved this session: {sample_count}",
    ]

    y = 30

    for text in instructions:
        cv2.putText(
            frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 35


def main():
    create_csv_if_needed(OUTPUT_CSV)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Landmark data collector started.")
    print("Show one hand clearly to the camera.")
    print("Press 1-5 to save a labeled sample.")
    print("Press q to quit.")
    print(f"Saving data to: {OUTPUT_CSV}")

    sample_count = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb_frame)

            detected_hand = None

            if results.multi_hand_landmarks:
                detected_hand = results.multi_hand_landmarks[0]

                mp_drawing.draw_landmarks(
                    frame,
                    detected_hand,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

            draw_instructions(frame, sample_count)

            display_frame = cv2.resize(frame, (960, 540))
            cv2.imshow("SignAssist CV - Data Collector", display_frame)

            raw_key = cv2.waitKey(1)
            key = raw_key & 0xFF

            if key == ord("q"):
                break

            if key in GESTURE_LABELS:
                if detected_hand is None:
                    print("No hand detected. Sample not saved.")
                    continue

                label = GESTURE_LABELS[key]
                features = extract_landmark_features(detected_hand)

                save_sample(OUTPUT_CSV, label, features)
                sample_count += 1

                print(f"Saved sample {sample_count}: {label}")

    cap.release()
    cv2.destroyAllWindows()
    print("Data collector closed.")
    print(f"Total samples saved this session: {sample_count}")


if __name__ == "__main__":
    main()