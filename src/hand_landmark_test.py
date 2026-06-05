# Week 2 Day 2: Hand landmark test for SignAssist CV
# Goal: Use MediaPipe to detect hands and draw landmarks on webcam feed.

import cv2
import mediapipe as mp


def main():
    # MediaPipe tools
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Open webcam using Windows-friendly backend
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Webcam opened successfully.")
    print("Show your hand to the camera.")
    print("Press 'q' to quit.")

    # Create MediaPipe Hands detector
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

            # Mirror frame for natural webcam behavior
            frame = cv2.flip(frame, 1)

            # MediaPipe expects RGB, but OpenCV uses BGR
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame with MediaPipe
            results = hands.process(rgb_frame)

            # If a hand is detected, draw landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

            # Resize display so window looks clean
            display_frame = cv2.resize(frame, (960, 540))

            cv2.imshow("SignAssist CV - Hand Landmark Test", display_frame)

            raw_key = cv2.waitKey(1)
            key = raw_key & 0xFF

            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")


if __name__ == "__main__":
    main()