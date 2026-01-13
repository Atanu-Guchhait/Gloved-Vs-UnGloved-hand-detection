import os
import cv2
import json
import argparse
from ultralytics import YOLO
from tqdm import tqdm

def detect_images(input_dir, output_dir, conf):
    model = YOLO("runs/detect/glove_detector/weights/best.pt")

    os.makedirs(output_dir, exist_ok=True)
    img_out = os.path.join(output_dir, "images")
    os.makedirs(img_out, exist_ok=True)

    all_results = []

    for img_name in tqdm(os.listdir(input_dir)):
        if not img_name.lower().endswith(".jpg"):
            continue

        img_path = os.path.join(input_dir, img_name)
        image = cv2.imread(img_path)

        results = model(image, conf=conf)[0]
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            score = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            detections.append({
                "label": label,
                "confidence": round(score, 2),
                "bbox": [x1, y1, x2, y2]
            })

            color = (0, 255, 0) if label == "gloved_hand" else (0, 0, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image,
                f"{label} {score:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        cv2.imwrite(os.path.join(img_out, img_name), image)

        all_results.append({
            "filename": img_name,
            "detections": detections
        })

    with open(os.path.join(output_dir, "detections.json"), "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="output")
    parser.add_argument("--confidence", type=float, default=0.5)

    args = parser.parse_args()
    detect_images(args.input, args.output, args.confidence)
