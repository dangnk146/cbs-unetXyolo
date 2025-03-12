import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO
from attention_unet import build_attention_unet
import torch

# ------------------------------
# Cấu hình và load mô hình YOLOv11s và Attention UNet
# ------------------------------
model_yolo = YOLO("yolov11m2-train.pt")  # File mô hình detection đã được huấn luyện

class_names = {
    0: "TYPE 1 - P05",
    1: "TYPE 2 - IQA1425",
    2: "TYPE 2 - IQA1900",
    3: "TYPE 2 - IQA3800",
    4: "TYPE 2 - IQA950",
    5: "TYPE 7 - L1",
    6: "TYPE2-IQA2750A",
    7: "background"
}

# Sử dụng cùng kiến trúc như training: input_size = (256,256,3), num_classes = 8 (7 lớp đối tượng + background)
num_classes = 8
input_size = (256, 256, 3)
model_unet = build_attention_unet(input_size, num_classes)

if os.path.exists("attention_unet_coco.h5"):
    model_unet.load_weights("attention_unet_coco.h5")
    print("Loaded Attention UNet weights.")
else:
    print("Warning: Attention UNet weights not found.")

# ------------------------------
# Hàm apply_nms: Loại bỏ các bbox trùng lặp
# ------------------------------
def apply_nms(boxes, iou_threshold=0.3):
    if len(boxes) == 0:
        return []
    
    bboxes = np.array([b["bbox"] for b in boxes])
    scores = np.array([b["confidence"] for b in boxes])

    bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)

    keep_indices = torch.ops.torchvision.nms(bboxes_tensor, scores_tensor, iou_threshold)
    keep_indices = keep_indices.numpy()

    return [boxes[i] for i in keep_indices]

# ------------------------------
# Hàm xử lý kết quả YOLO
# ------------------------------
def process_yolo_results(results, conf_threshold=0.3):
    boxes = []
    for r in results:
        for box in r.boxes:
            if box.conf > conf_threshold:
                x_center, y_center, width, height = box.xywh[0]
                x1 = int(x_center - width/2)
                y1 = int(y_center - height/2)
                x2 = int(x_center + width/2)
                y2 = int(y_center + height/2)
                
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                boxes.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_id": class_id,
                    "confidence": confidence
                })
    return boxes

# ------------------------------
# Hàm load ground truth bounding boxes
# ------------------------------
def load_ground_truth_boxes(image_path, labels_folder="./datasets-origin/test/labels"):
    base = os.path.basename(image_path)
    name, _ = os.path.splitext(base)
    label_path = os.path.join(labels_folder, name + ".txt")
    gt_boxes = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()
        img = cv2.imread(image_path)
        H, W = img.shape[:2]
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                _, xc, yc, w, h = map(float, parts)
                x1 = int((xc - w/2) * W)
                y1 = int((yc - h/2) * H)
                x2 = int((xc + w/2) * W)
                y2 = int((yc + h/2) * H)
                gt_boxes.append([x1, y1, x2, y2])
    return gt_boxes

# ------------------------------
# Hàm tính IoU
# ------------------------------
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# ------------------------------
# Hàm pipeline_inference cho một ảnh
# ------------------------------
def pipeline_inference(image_path, margin=5):
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        print(f"Error: Không thể đọc ảnh {image_path}")
        return []
    
    results = model_yolo.predict(orig_img)
    boxes = apply_nms(process_yolo_results(results, conf_threshold=0.3), iou_threshold=0.3)

    final_results = []
    for box in boxes:
        x1, y1, x2, y2 = box["bbox"]
        # Thêm margin và clip tọa độ
        x1 = max(x1 - margin, 0)
        y1 = max(y1 - margin, 0)
        x2 = min(x2 + margin, orig_img.shape[1] - 1)
        y2 = min(y2 + margin, orig_img.shape[0] - 1)
        
        if x2 <= x1 or y2 <= y1:
            print(f"Warning: ROI không hợp lệ cho bbox {box['bbox']}")
            continue
        
        roi = orig_img[y1:y2, x1:x2]
        if roi.size == 0:
            print(f"Warning: ROI rỗng cho bbox {box['bbox']}")
            continue

        try:
            roi_resized = cv2.resize(roi, (256,256))
        except Exception as e:
            print(f"Error khi resize ROI: {e}")
            continue

        roi_input = roi_resized.astype(np.float32) / 255.0
        roi_input = np.expand_dims(roi_input, axis=0)
        pred_mask = model_unet.predict(roi_input)[0, ..., 0]
        mask_bin = (pred_mask > 0.6).astype(np.uint8) * 255

        # Tìm các vùng segmentation (contours)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Kiểm tra và gộp các contours gần nhau
        merged_contours = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if any(cv2.pointPolygonTest(merged, (x + w//2, y + h//2), False) >= 0 for merged in merged_contours):
                continue
            merged_contours.append(c)

        # Vẽ lại mask sau khi gộp
        mask_bin[:] = 0
        cv2.drawContours(mask_bin, merged_contours, -1, (255), thickness=cv2.FILLED)

        # Đếm lại số đối tượng trong mask sau khi gộp
        num_labels, _ = cv2.connectedComponents(mask_bin)
        count = num_labels - 1

        # Ghi thông tin count lên ROI (bạn có thể lưu riêng mask nếu muốn)
        cv2.putText(roi_resized, f"Count: {count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        final_results.append({
            'bbox': [x1, y1, x2, y2],
            'mask': mask_bin,
            'count': count,
            'class_id': box['class_id']
        })
    return final_results

# ------------------------------
# Hàm chạy inference trên tập test và so sánh với ground truth
# ------------------------------
def run_inference_on_test(test_images_folder, test_labels_folder="./datasets-origin/test/labels", output_folder="results", margin=20, iou_threshold=0.3):
    os.makedirs(output_folder, exist_ok=True)
    image_files = sorted(glob.glob(os.path.join(test_images_folder, "*.*")))
    # Thông số IoU threshold dùng cho tính accuracy
    print(f"[DEBUG] IoU Threshold: {iou_threshold}")

    total_accuracy = 0
    total_images = 0

    for idx, img_path in enumerate(image_files):
        print(f"\n[DEBUG] Xử lý ảnh: {img_path}")
        gt_boxes = load_ground_truth_boxes(img_path, labels_folder=test_labels_folder)
        if not gt_boxes:
            print("Không tìm thấy label cho ảnh này.")
            continue

        results_pipeline = pipeline_inference(img_path, margin=margin)
        # Tính tổng số object được phát hiện trên ảnh từ các ROI
        total_objects = sum([res['count'] for res in results_pipeline])
        print(f"[DEBUG] Tổng số object phát hiện trên ảnh: {total_objects}")
        predicted_boxes = [res['bbox'] for res in results_pipeline]

        true_positives = 0
        for gt in gt_boxes:
            for pred in predicted_boxes:
                if compute_iou(gt, pred) >= iou_threshold:
                    true_positives += 1
                    break
        image_accuracy = (true_positives / len(gt_boxes)) * 100
        print(f"[DEBUG] Độ chính xác (detection) cho ảnh {os.path.basename(img_path)}: {image_accuracy:.2f}%")
        total_accuracy += image_accuracy
        total_images += 1

        img = cv2.imread(img_path)
        # Vẽ bounding box và ghi count lên ảnh
        for i, res in enumerate(results_pipeline):
            x1, y1, x2, y2 = res['bbox']
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img, f"Count: {res['count']}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            # Vẽ loại object lên trên cùng, có thể thay đổi tọa độ theo ý muốn
            type_text = f"Type: {class_names.get(res['class_id'], 'Unknown')}"
            cv2.putText(img, type_text, (x1, y1 - 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        
        # Ghi tổng số object và độ chính xác lên ảnh
        cv2.putText(img, f"Total Objects: {total_objects}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, f"Detection Acc: {image_accuracy:.2f}%", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, f"IoU Thres: {iou_threshold}", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        output_img_path = os.path.join(output_folder, f"result_{idx}.jpg")
        cv2.imwrite(output_img_path, img)
        print(f"[DEBUG] Kết quả được lưu: {output_img_path}")

    if total_images > 0:
        avg_accuracy = total_accuracy / total_images
        print(f"[DEBUG] Độ chính xác trung bình trên tập test: {avg_accuracy:.2f}%")
    else:
        print("Không có ảnh có ground truth để tính độ chính xác.")

if __name__ == "__main__":
    test_images_folder = "./datasets-origin-preprocessed/test/images"
    test_labels_folder = "./datasets-origin-preprocessed/test/labels"
    run_inference_on_test(test_images_folder, test_labels_folder, output_folder="results", margin=20, iou_threshold=0.3)
