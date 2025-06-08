import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

def main():
    model = YOLO("yolov8n.pt")
    box_annotator = sv.BoxAnnotator(thickness=2)

    # Avoid flickering by disabling Ultralytics' internal display
    for result in model.track(source='C:/Users/HP/Downloads/UV.mp4', stream=True, show=False, persist=True):
        frame = result.orig_img

        # Track IDs
        tracker_ids = result.boxes.id
        if tracker_ids is not None:
            tracker_ids = tracker_ids.cpu().numpy().astype(int)
        else:
            tracker_ids = [-1] * len(result.boxes)

        # Create detections
        detections = sv.Detections(
            xyxy=result.boxes.xyxy.cpu().numpy(),
            confidence=result.boxes.conf.cpu().numpy(),
            class_id=result.boxes.cls.cpu().numpy().astype(int),
            tracker_id=np.array(tracker_ids)
        )

        # Labels
        labels = [
            f"#{tracker_id} {model.names[class_id]} {confidence:0.2f}"
            for confidence, class_id, tracker_id in zip(
                detections.confidence, detections.class_id, detections.tracker_id
            )
        ]

        detections.data["labels"] = labels
        frame = box_annotator.annotate(scene=frame, detections=detections)

        # Draw labels with background
        for box, label in zip(detections.xyxy, labels):
            x1, y1, x2, y2 = map(int, box)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            font_thickness = 3
            text_color = (0, 255, 255)   # Yellow
            bg_color = (0, 0, 0)         # Black

            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + th + 10

            # Background
            cv2.rectangle(
                frame,
                (text_x, text_y - th - baseline),
                (text_x + tw, text_y + baseline),
                bg_color,
                thickness=cv2.FILLED
            )

            # Text
            cv2.putText(
                frame,
                label,
                (text_x, text_y),
                font,
                font_scale,
                text_color,
                font_thickness
            )

        # Optional: resize for display
        frame = cv2.resize(frame, (960, 740))

        # Show frame smoothly
        cv2.imshow("YOLOv8 Object Tracking", frame)

        # Wait ~33 ms for ~30 FPS playback
        if cv2.waitKey(33) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
