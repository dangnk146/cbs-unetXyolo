import os
import glob
import json
from PIL import Image

def yolo_to_coco(dataset_dir, images_dir, labels_dir, class_names, output_json):
    """
    Chuyển đổi dataset theo định dạng YOLO sang định dạng COCO.
    
    Parameters:
        dataset_dir (str): Đường dẫn đến thư mục chứa dataset (có thể dùng để lưu thông tin bổ sung).
        images_dir (str): Đường dẫn đến thư mục chứa ảnh (train, val, test).
        labels_dir (str): Đường dẫn đến thư mục chứa file annotation (YOLO .txt).
        class_names (list): Danh sách tên các lớp (vd: ['TYPE 1 - P05', 'TYPE 2 - IQA1425', ...]).
        output_json (str): Đường dẫn file JSON đầu ra chứa annotation theo COCO.
        
    Returns:
        coco (dict): Dictionary chứa annotation theo định dạng COCO.
    """
    # Khởi tạo cấu trúc COCO cơ bản
    coco = {
        "info": {
            "description": "Dataset converted from YOLO to COCO",
            "url": "",
            "version": "1.0",
            "year": 2025,
            "contributor": "",
            "date_created": ""
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Tạo danh mục categories
    for i, name in enumerate(class_names):
        coco["categories"].append({
            "id": i,
            "name": name,
            "supercategory": "none"
        })
    
    annotation_id = 0
    image_id = 0
    # Lấy danh sách ảnh từ thư mục images_dir
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.*")))
    
    for img_file in image_files:
        # Lấy thông tin kích thước ảnh sử dụng PIL
        with Image.open(img_file) as im:
            width, height = im.size
        
        # Thêm thông tin ảnh vào danh sách images
        coco["images"].append({
            "id": image_id,
            "file_name": os.path.basename(img_file),
            "width": width,
            "height": height
        })
        
        # Tìm file annotation tương ứng (giả sử cùng tên với ảnh, đuôi .txt)
        base = os.path.splitext(os.path.basename(img_file))[0]
        label_file = os.path.join(labels_dir, base + ".txt")
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                lines = f.readlines()
            # Mỗi dòng trong file YOLO có dạng:
            # class x_center y_center width height  (các giá trị relative)
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    cls = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                except Exception as e:
                    print(f"Lỗi parse file {label_file}: {e}")
                    continue
                
                # Chuyển từ relative sang absolute
                bbox_width = w * width
                bbox_height = h * height
                x1 = (x_center - w/2) * width
                y1 = (y_center - h/2) * height
                
                # COCO bounding box format: [x, y, width, height]
                bbox = [x1, y1, bbox_width, bbox_height]
                
                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": cls,
                    "bbox": bbox,
                    "area": bbox_width * bbox_height,
                    "iscrowd": 0,
                    "segmentation": []  # Không có thông tin segmentation, để trống
                })
                annotation_id += 1
        image_id += 1
        
    # Lưu kết quả dưới dạng file JSON
    with open(output_json, "w") as f:
        json.dump(coco, f, indent=4)
    
    return coco

# Ví dụ sử dụng:
if __name__ == "__main__":
    # Cấu hình đường dẫn
    dataset_dir = "./datasets-origin"  # hoặc bạn có thể không dùng tới nếu không cần thiết
    images_dir = "./datasets-origin/valid/images"  # có thể là valid, val hoặc test
    labels_dir = "./datasets-origin/valid/labels"
    
    # Số lượng lớp và tên lớp
    class_names = ['TYPE 1 - P05', 'TYPE 2 - IQA1425', 'TYPE 2 - IQA1900',
                   'TYPE 2 - IQA3800', 'TYPE 2 - IQA950', 'TYPE 7 - L1', 'TYPE2-IQA2750A']
    
    output_json = "coco_annotations_valid.json"
    
    coco_dataset = yolo_to_coco(dataset_dir, images_dir, labels_dir, class_names, output_json)
    print(f"Đã chuyển đổi dataset và lưu vào file: {output_json}")
