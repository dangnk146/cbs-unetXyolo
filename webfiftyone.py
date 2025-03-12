import fiftyone as fo

# Cấu hình đường dẫn
dataset_dir = "."  # Giả sử file coco_annotations.json nằm trong thư mục gốc của dự án
data_path = "datasets-origin/train/images"  # Thư mục chứa ảnh của tập train
labels_path = "coco_annotations.json"  # File annotation đã chuyển đổi sang COCO

# Import dataset COCO vào FiftyOne sử dụng Dataset.from_dir
dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=fo.types.COCODetectionDataset,
    labels_path=labels_path,
    data_path=data_path,
    name="coco_dataset",
    overwrite=True
)

# Khởi chạy ứng dụng web FiftyOne
session = fo.launch_app(dataset, address="0.0.0.0", port=5151)
input("Press Enter to exit...\n")
