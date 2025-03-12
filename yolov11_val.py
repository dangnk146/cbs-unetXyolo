from ultralytics import YOLO

# Load mô hình YOLOv11m pretrained
model_yolo = YOLO("yolov11m-train.pt")

# Huấn luyện trên dataset của bạn (fine-tune)
results = model_yolo.val(data="data.yaml", imgsz=640, conf=0.25, iou=0.6, split="test", plots=True)
