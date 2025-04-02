from ultralytics import YOLO

# Load mô hình YOLOv11m pretrained
model_yolo = YOLO("yolov11m2-train.pt")

# Huấn luyện trên dataset của bạn (fine-tune)
results = model_yolo.train(
    data="data.yaml",
    epochs=100,  # Tăng epochs nếu có thể
    imgsz=640,
    lr0=0.002,  # Learning rate ban đầu, thử giảm xuống nếu model bị overfit
    lrf=0.007,  # Learning rate cuối
    plots=True
)


model_yolo.save('yolov11m3-train.pt')
