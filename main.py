from ultralytics import YOLO
import cv2
import pyttsx3
import time


def main():
    # Load the pretrained YOLOv8 nano model. Ultralytics downloads it if needed.
    model = YOLO("yolov8n.pt")

    # Set up text-to-speech and a five-second cooldown for each object class.
    speech_engine = pyttsx3.init()
    last_announced = {}
    announcement_cooldown = 5

    # Open the default webcam; camera index 0 is usually the built-in camera.
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        speech_engine.stop()
        raise RuntimeError(
            "Could not open the webcam. Check camera permissions or try camera index 1."
        )

    try:
        while True:
            # Capture one image from the webcam.
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a frame from the webcam.")

            # Run object detection on the current frame.
            results = model(frame, verbose=False)

            # Read the time once per frame to check announcement cooldowns.
            current_time = time.time()

            # Draw each detected object's bounding box and label on the frame.
            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    x1, y1, x2, y2 = (
                        int(value) for value in box.xyxy[0].tolist()
                    )
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    name = model.names[class_id]
                    label = f"{name} {confidence:.2f}"

                    # Announce each class at most once every five seconds.
                    if (
                        name not in last_announced
                        or current_time - last_announced[name] > announcement_cooldown
                    ):
                        message = f"{name} detected"
                        print(message)
                        speech_engine.say(message)
                        speech_engine.runAndWait()
                        last_announced[name] = time.time()

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

            # Display the live, annotated feed. Press Q or q to exit.
            cv2.imshow("YOLOv8 Webcam Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
    finally:
        # Always release the webcam and close the display window.
        camera.release()
        cv2.destroyAllWindows()
        speech_engine.stop()


if __name__ == "__main__":
    main()
