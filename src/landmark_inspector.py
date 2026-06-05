# Week 2 Day 3: Landmark inspector for SignAssist CV
# Goal:
# 1. Detect one hand with MediaPipe
# 2. Draw hand landmarks
# 3. Label each landmark from 0 to 20
# 4. Press 'p' to print landmark coordinates
# 5. Press 'q' to quit

import cv2
import mediapipe as mp


def print_landmark_coordinates(hand_landmarks):
    """Print normalized x, y, z coordinates for all 21 hand landmarks."""
    print("\nHand landmark coordinates:")

    for index, landmark in enumerate(hand_landmarks.landmark):
        print(
            f"Landmark {index:02d}: "
            f"x={landmark.x:.4f}, "
            f"y={landmark.y:.4f}, "
            f"z={landmark.z:.4f}"
        )


def draw_landmark_numbers(frame, hand_landmarks):
    """Draw landmark index numbers on the frame."""
    height, width, _ = frame.shape

    for index, landmark in enumerate(hand_landmarks.landmark):
        # Convert normalized coordinates to pixel coordinates.
        pixel_x = int(landmark.x * width)
        pixel_y = int(landmark.y * height)

        cv2.putText(
            frame,
            str(index),
            (pixel_x, pixel_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )


def main():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Landmark inspector started.")
    print("Show one hand to the camera.")
    print("Press 'p' to print landmark coordinates.")
    print("Press 'q' to quit.")

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

                draw_landmark_numbers(frame, detected_hand)

            display_frame = cv2.resize(frame, (960, 540))
            cv2.imshow("SignAssist CV - Landmark Inspector", display_frame)

            raw_key = cv2.waitKey(1)
            key = raw_key & 0xFF

            if key == ord("p"):
                if detected_hand is not None:
                    print_landmark_coordinates(detected_hand)
                else:
                    print("No hand detected. Try showing your full hand clearly.")

            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Landmark inspector closed.")


if __name__ == "__main__":
    main()