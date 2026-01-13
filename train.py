from ultralytics import YOLO

def train():
    model = YOLO("yolov8n.pt")

    model.train(
        data="data.yml",
        epochs=30,
        imgsz=640,
        batch=16,
        name="glove_detector"
    )

if __name__ == "__main__":
    train()
