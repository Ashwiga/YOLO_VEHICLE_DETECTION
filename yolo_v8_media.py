# USAGE:
# python yolo_v8_media.py --input 0 --output output/webcam.mp4 --model yolov8n.pt
# python yolo_v8_media.py --input videos/sample.mp4 --output output/video.mp4 --model yolov8n.pt
# python yolo_v8_media.py --input images/sample.jpg --output output/sample_output.jpg --model yolov8n.pt

import cv2
import argparse
import os
from ultralytics import YOLO

# Argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help="Path to input image/video, or 0 for webcam")
ap.add_argument("-o", "--output", required=True,
    help="Path to save output (image or video)")
ap.add_argument("-m", "--model", default="yolov8n.pt",
    help="Path to YOLOv8 model (.pt file)")
args = vars(ap.parse_args())

# Create output folder if not exists
os.makedirs(os.path.dirname(args["output"]), exist_ok=True)

# Load model
print("[INFO] Loading YOLOv8 model...")
model = YOLO(args["model"])

# Detect input type
input_path = args["input"]
is_webcam = input_path == "0" or input_path == 0
is_image = input_path.lower().endswith((".jpg", ".jpeg", ".png"))
is_video = input_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))

# -------------------- IMAGE INPUT --------------------
if is_image:
    print(f"[INFO] Processing image: {input_path}")
    image = cv2.imread(input_path)
    results = model(image)
    annotated = results[0].plot()
    cv2.imwrite(args["output"], annotated)
    print(f"[INFO] Saved output image to: {args['output']}")
    cv2.imshow("Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -------------------- VIDEO / WEBCAM INPUT --------------------
else:
    source = 0 if is_webcam else input_path
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        exit(1)

    writer = None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else -1
    print(f"[INFO] Resolution: {width}x{height} | FPS: {fps} | Frames: {total_frames if total_frames > 0 else 'Live'}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        print(f"[INFO] Processing frame {frame_count}", end="\r")

        results = model(frame, verbose=False)
        annotated = results[0].plot()

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(args["output"], fourcc, fps, (width, height))
            print(f"[INFO] Saving to: {args['output']}")

        writer.write(annotated)
        cv2.imshow("YOLOv8 Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[INFO] Interrupted by user.")
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Video processing complete.")
