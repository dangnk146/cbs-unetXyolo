import os
import glob
import cv2
import numpy as np

def load_mask_from_txt(txt_file, img_size=(256, 256)):
    """
    Đọc file .txt chứa annotation.
    - Nếu token đầu tiên có thể chuyển sang int được, giả sử định dạng: class x_center y_center width height.
    - Nếu không, giả sử định dạng: x_center y_center width height.
    Các giá trị đều ở dạng relative (0 đến 1).
    Hàm tạo mask nhị phân với các bounding box được vẽ đầy.
    Logging chi tiết được thêm vào để debug.
    """
    # Tạo mask trống với kích thước img_size
    mask = np.zeros(img_size, dtype=np.uint8)
    
    # Đọc nội dung file
    with open(txt_file, 'r') as f:
        content = f.read().strip()
    
    if not content:
        print(f"[DEBUG] Warning: {txt_file} rỗng.")
        # Đảm bảo trả về mask có shape (H, W, 1)
        return np.expand_dims(mask, axis=-1)

    tokens = content.split()
    num_tokens = len(tokens)
    print(f"[DEBUG] File: {txt_file} có {num_tokens} token.")
    print(f"[DEBUG] Nội dung token đầu tiên: {tokens[:10]}")
    
    # Xác định định dạng: thử parse token đầu tiên dưới dạng int
    try:
        _ = int(tokens[0])
        fmt = 5  # có class
        print(f"[DEBUG] Định dạng file {txt_file}: 5 token mỗi đối tượng (có class).")
    except ValueError:
        fmt = 4  # không có class
        print(f"[DEBUG] Định dạng file {txt_file}: 4 token mỗi đối tượng (không có class).")

    # Kiểm tra số token có chia hết cho fmt không
    if num_tokens % fmt != 0:
        print(f"[DEBUG] Warning: Số token trong {txt_file} ({num_tokens}) không chia hết cho {fmt}.")
        # Đảm bảo trả về mask có shape (H, W, 1)
        return np.expand_dims(mask, axis=-1)

    num_objects = num_tokens // fmt
    print(f"[DEBUG] File {txt_file} chứa {num_objects} object(s).")
    
    # Duyệt qua từng annotation
    for i in range(0, num_tokens, fmt):
        try:
            if fmt == 5:
                # Có class, bỏ qua thông tin class
                class_token = tokens[i]
                try:
                    cls = int(class_token)
                except Exception as e:
                    print(f"[DEBUG] Lỗi chuyển class token '{class_token}' sang int tại vị trí {i}: {e}")
                    continue
                x_center = float(tokens[i+1])
                y_center = float(tokens[i+2])
                w = float(tokens[i+3])
                h = float(tokens[i+4])
            else:
                x_center = float(tokens[i])
                y_center = float(tokens[i+1])
                w = float(tokens[i+2])
                h = float(tokens[i+3])
        except Exception as e:
            print(f"[DEBUG] Lỗi khi parse file {txt_file} tại token index {i}: {e}")
            continue

        # In log thông tin object để debug
        print(f"[DEBUG] Object {i//fmt+1}: x_center={x_center}, y_center={y_center}, w={w}, h={h}")
        
        # Chuyển đổi relative thành absolute dựa trên kích thước mong muốn
        W, H = img_size[1], img_size[0]
        x_center *= W
        y_center *= H
        w *= W
        h *= H
        
        # Tính tọa độ bounding box
        x1 = int(x_center - w / 2)
        y1 = int(y_center - h / 2)
        x2 = int(x_center + w / 2)
        y2 = int(y_center + h / 2)
        
        # Vẽ hình chữ nhật đầy vào mask
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    
    # Chuẩn hóa mask về [0,1] và thêm kênh (channels)
    mask = mask / 255.0
    mask = np.expand_dims(mask, axis=-1)
    return mask

def load_images_and_masks(images_path, masks_path, img_size=(256, 256), max_files=50):
    """
    Load ảnh và mask từ các thư mục đã cho.
    Nếu file mask có định dạng ảnh (png, jpg, jpeg) thì đọc trực tiếp.
    Nếu file mask là file .txt chứa annotation YOLO thì chuyển đổi thành mask nhị phân.
    Giả sử tên file ảnh và mask khớp với nhau.
    Thêm tham số max_files để giới hạn số file load (mặc định 50).
    """
    image_files = sorted(glob.glob(os.path.join(images_path, "*.*")))[:max_files]
    mask_files = sorted(glob.glob(os.path.join(masks_path, "*.*")))[:max_files]
    
    print(f"[DEBUG] Tìm thấy {len(image_files)} file ảnh và {len(mask_files)} file mask (giới hạn {max_files}).")
    
    images = []
    masks = []
    for img_file, mask_file in zip(image_files, mask_files):
        # Đọc ảnh (BGR -> RGB)
        img = cv2.imread(img_file)
        if img is None:
            print(f"[DEBUG] Không thể đọc ảnh: {img_file}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        images.append(img)
        
        # Xử lý file mask tùy theo định dạng
        ext = os.path.splitext(mask_file)[1].lower()
        if ext in ['.png', '.jpg', '.jpeg']:
            m = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            if m is None:
                print(f"[DEBUG] Không thể đọc mask ảnh: {mask_file}")
                continue
            m = cv2.resize(m, img_size)
            m = m / 255.0
            m = np.expand_dims(m, axis=-1)
        elif ext == '.txt':
            print(f"[DEBUG] Xử lý file mask dạng txt: {mask_file}")
            m = load_mask_from_txt(mask_file, img_size=img_size)
        else:
            print(f"[DEBUG] Định dạng file không được hỗ trợ: {mask_file}")
            continue
        
        masks.append(m)
    
    images = np.array(images, dtype=np.float32) / 255.0
    masks = np.array(masks, dtype=np.float32)
    return images, masks

if __name__ == "__main__":
    # Cập nhật đường dẫn đến dữ liệu của bạn
    train_images_path = "./datasets-origin/train/images"
    train_masks_path = "./datasets-origin/train/labels"
    
    print("Loading images and masks...")
    imgs, msks = load_images_and_masks(train_images_path, train_masks_path, img_size=(256, 256), max_files=50)
    print("Số lượng ảnh:", imgs.shape[0])
    print("Kích thước mảng ảnh:", imgs.shape)
    print("Kích thước mảng mask:", msks.shape)
    
    # Lưu lại file mask đầu tiên ra để kiểm tra (nhân lại 255 để lưu dưới dạng ảnh)
    if msks.shape[0] > 0:
        demo_mask = (msks[0] * 255).astype(np.uint8)
        cv2.imwrite("demo_mask.png", demo_mask)
        print("Đã lưu demo_mask.png")
