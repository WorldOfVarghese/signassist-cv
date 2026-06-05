# Week 2 Day 1: Webcam test for SignAssist CV
# Goal: Open webcam, display video feed, and close safely with 'q'.

import cv2


def main():
    # On Windows, cv2.CAP_DSHOW often opens the webcam faster/more reliably.
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Webcam opened successfully.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)

        display_frame = cv2.resize(frame, (960, 540))

        cv2.imshow("SignAssist CV - Webcam Test", display_frame)


        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")


if __name__ == "__main__":
    main()