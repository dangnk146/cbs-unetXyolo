import os
import cv2
import numpy as np
import shutil  # Thêm thư viện shutil để copy file
from tqdm import tqdm

def preprocess_image(image):
    """
    Làm nổi bật cạnh bằng Sobel và kết hợp với ảnh gốc.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel_normalized = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    alpha = 0.7
    beta = 0.3
    combined = cv2.addWeighted(image, alpha, cv2.cvtColor(sobel_normalized, cv2.COLOR_GRAY2BGR), beta, 0)
    return combined

def process_and_save_images(input_folder, output_folder):
    """
    Đọc ảnh, tiền xử lý và lưu.
    """
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for file_name in tqdm(image_files, desc=f"Processing images in {input_folder}"):
        image_path = os.path.join(input_folder, file_name)
        try:
            image = cv2.imread(image_path)
            if image is not None:
                processed_image = preprocess_image(image)
                output_path = os.path.join(output_folder, file_name)
                cv2.imwrite(output_path, processed_image)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

def copy_labels(input_folder, output_folder):
    """
    Sao chép nhãn từ thư mục gốc sang thư mục đã xử lý.
    """
    os.makedirs(output_folder, exist_ok=True)
    label_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.txt')]

    for file_name in tqdm(label_files, desc=f"Copying labels from {input_folder}"):
        label_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)
        try:
            shutil.copy2(label_path, output_path)  # copy2 giữ nguyên metadata
        except Exception as e:
            print(f"Error copying {label_path}: {e}")

def main():
    # Đường dẫn gốc
    base_input_dir = "/home/dangnguyen/Desktop/yolov11/datasets-origin"
    base_output_dir = "/home/dangnguyen/Desktop/yolov11/datasets-origin-preprocessed"

    sets = ["train", "valid", "test"]  # Các tập dữ liệu

    for dataset in sets:
        input_images_dir = os.path.join(base_input_dir, dataset, "images")
        output_images_dir = os.path.join(base_output_dir, dataset, "images")
        input_labels_dir = os.path.join(base_input_dir, dataset, "labels")
        output_labels_dir = os.path.join(base_output_dir, dataset, "labels")


        # Kiểm tra xem thư mục ảnh có tồn tại không trước khi xử lý
        if os.path.exists(input_images_dir):
            process_and_save_images(input_images_dir, output_images_dir)
        else:
            print(f"Warning: Image directory not found: {input_images_dir}")

        # Kiểm tra xem thư mục nhãn có tồn tại không trước khi copy
        if os.path.exists(input_labels_dir):
            copy_labels(input_labels_dir, output_labels_dir)
        else:
             print(f"Warning: Label directory not found: {input_labels_dir}")
if __name__ == "__main__":
    main()