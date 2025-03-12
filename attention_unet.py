import os
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Activation, Add, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# -------------------------------
# Các hàm xây dựng kiến trúc mô hình
# -------------------------------

def conv_block(x, filters):
    """Khối Conv gồm 2 lớp Conv2D + BatchNormalization + ReLU"""
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def attention_gate(x, g, inter_channels):
    """
    Attention Gate đơn giản:
      - x: skip connection từ encoder.
      - g: tín hiệu từ decoder.
    Trả về phần feature của x được nhân với hệ số attention.
    """
    theta_x = Conv2D(inter_channels, (1, 1), padding='same')(x)
    phi_g = Conv2D(inter_channels, (1, 1), padding='same')(g)
    add_xg = Add()([theta_x, phi_g])
    act_xg = Activation('relu')(add_xg)
    psi = Conv2D(1, (1, 1), padding='same')(act_xg)
    psi = Activation('sigmoid')(psi)
    attn_coeff = Multiply()([x, psi])
    return attn_coeff

class SelfAttention(tf.keras.layers.Layer):
    """
    Khối Self-Attention kiểu SAGAN.
    Tính attention map trên feature map và kết hợp với input.
    """
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.f_conv = Conv2D(channels // 8, kernel_size=1, padding='same')
        self.g_conv = Conv2D(channels // 8, kernel_size=1, padding='same')
        self.h_conv = Conv2D(channels, kernel_size=1, padding='same')
        self.gamma = self.add_weight(name='gamma', shape=[1],
                                     initializer=tf.zeros_initializer(), trainable=True)

    def call(self, x):
        f = self.f_conv(x)  # (batch, H, W, C//8)
        g = self.g_conv(x)  # (batch, H, W, C//8)
        h = self.h_conv(x)  # (batch, H, W, C)
        shape = tf.shape(x)
        batch_size, h_dim, w_dim = shape[0], shape[1], shape[2]
        # Reshape các tensor: (batch, N, C')
        f_flat = tf.reshape(f, [batch_size, -1, self.channels // 8])
        g_flat = tf.reshape(g, [batch_size, -1, self.channels // 8])
        h_flat = tf.reshape(h, [batch_size, -1, self.channels])
        # Tính attention map
        s = tf.matmul(f_flat, g_flat, transpose_b=True)  # (batch, N, N)
        beta = tf.nn.softmax(s, axis=-1)  # attention map
        o = tf.matmul(beta, h_flat)  # (batch, N, C)
        o = tf.reshape(o, [batch_size, h_dim, w_dim, self.channels])
        x = self.gamma * o + x
        return x

def build_attention_unet(input_shape, num_classes):
    """
    Xây dựng mô hình Attention U-Net với self-attention ở bottleneck.
    input_shape: tuple, ví dụ (256,256,3)
    num_classes: số lớp đầu ra (bao gồm background)
    """
    inputs = Input(input_shape)

    # Encoder
    c1 = conv_block(inputs, 64)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 128)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 256)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, 512)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = conv_block(p4, 1024)
    # Áp dụng Self-Attention tại bottleneck
    sa = SelfAttention(1024)(c5)

    # Decoder với attention gate cho skip connection
    u6 = UpSampling2D((2, 2), interpolation='bilinear')(sa)
    att4 = attention_gate(c4, u6, inter_channels=512)
    u6 = Concatenate()([u6, att4])
    c6 = conv_block(u6, 512)

    u7 = UpSampling2D((2, 2), interpolation='bilinear')(c6)
    att3 = attention_gate(c3, u7, inter_channels=256)
    u7 = Concatenate()([u7, att3])
    c7 = conv_block(u7, 256)

    u8 = UpSampling2D((2, 2), interpolation='bilinear')(c7)
    att2 = attention_gate(c2, u8, inter_channels=128)
    u8 = Concatenate()([u8, att2])
    c8 = conv_block(u8, 128)

    u9 = UpSampling2D((2, 2), interpolation='bilinear')(c8)
    att1 = attention_gate(c1, u9, inter_channels=64)
    u9 = Concatenate()([u9, att1])
    c9 = conv_block(u9, 64)

    # Output: sử dụng softmax cho phân lớp pixel (sử dụng loss sparse_categorical_crossentropy)
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c9)

    model = Model(inputs, outputs)
    return model

# -------------------------------
# Các hàm xử lý dataset theo định dạng COCO
# -------------------------------

def load_annotations(json_file):
    """
    Load file coco_annotations.json và tạo mapping: file_name -> danh sách annotation.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    # Mapping image_id -> file_name
    image_id_to_file = {img['id']: img['file_name'] for img in data['images']}
    annotations_by_file = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        file_name = image_id_to_file.get(img_id)
        if file_name is None:
            continue
        if file_name not in annotations_by_file:
            annotations_by_file[file_name] = []
        annotations_by_file[file_name].append(ann)
    return annotations_by_file

def data_generator(image_dir, annotations_by_file, batch_size, input_size, num_classes):
    """
    Generator sinh batch dữ liệu:
      - Load ảnh từ image_dir.
      - Tạo mask từ annotation (dựa trên bbox). Mask được khởi tạo với giá trị 0 (background),
        và vùng bbox được điền giá trị = category_id + 1.
      - Resize cả ảnh và mask về kích thước input_size x input_size.
    """
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    while True:
        np.random.shuffle(image_files)
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i+batch_size]
            images = []
            masks = []
            for file in batch_files:
                # Load ảnh bằng cv2
                img = cv2.imread(file)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                orig_h, orig_w = img.shape[:2]
                # Resize ảnh về kích thước input_size x input_size và chuẩn hóa [0,1]
                img_resized = cv2.resize(img, (input_size, input_size))
                img_resized = img_resized / 255.0
                images.append(img_resized)
                
                # Tạo mask ban đầu kích thước (orig_h, orig_w) với background = 0
                mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                file_name = os.path.basename(file)
                if file_name in annotations_by_file:
                    for ann in annotations_by_file[file_name]:
                        # bbox theo định dạng [x, y, width, height]
                        bbox = ann['bbox']
                        x, y, w_box, h_box = bbox
                        x1 = int(round(x))
                        y1 = int(round(y))
                        x2 = int(round(x + w_box))
                        y2 = int(round(y + h_box))
                        # Điền label (category_id + 1; vì 0 dành cho background)
                        category = ann['category_id'] + 1
                        mask[y1:y2, x1:x2] = category
                # Resize mask với interpolation NEAREST để không làm thay đổi giá trị label
                mask_resized = cv2.resize(mask, (input_size, input_size), interpolation=cv2.INTER_NEAREST)
                masks.append(mask_resized)
            if images and masks:
                yield np.array(images), np.array(masks)

# -------------------------------
# Main: Huấn luyện mô hình
# -------------------------------

if __name__ == "__main__":
    # Cấu hình tham số
    batch_size = 4
    epochs = 50
    input_size = 256
    # Số lớp: giả sử có 7 lớp đối tượng, cộng thêm background -> tổng 8 lớp
    num_classes = 8

    # Đường dẫn dataset
    train_image_dir = "./datasets-origin/train/images"
    annotations_file = "coco_annotations.json"
    annotations_by_file = load_annotations(annotations_file)

    # Đường dẫn dataset validation
    val_image_dir = "./datasets-origin/valid/images"
    val_annotations_file = "coco_annotations_valid.json"  # Đảm bảo file này tồn tại
    val_annotations_by_file = load_annotations(val_annotations_file)

    # Xây dựng mô hình
    model_unet = build_attention_unet((input_size, input_size, 3), num_classes)
    model_unet.compile(optimizer=Adam(learning_rate=1e-4),
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
    model_unet.summary()

    # Tạo generator dữ liệu
    num_images = len([f for f in os.listdir(train_image_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    steps_per_epoch = num_images // batch_size if num_images >= batch_size else 1
    train_gen = data_generator(train_image_dir, annotations_by_file, batch_size, input_size, num_classes)

    # Tạo generator dữ liệu validation
    num_val_images = len([f for f in os.listdir(val_image_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    val_steps = num_val_images // batch_size if num_val_images >= batch_size else 1
    val_gen = data_generator(val_image_dir, val_annotations_by_file, batch_size, input_size, num_classes)


    # Lưu weights với tên file kết thúc bằng .weights.h5 để tránh lỗi
    checkpoint_callback = ModelCheckpoint(
        "attention_unet_coco.h5",
        monitor='val_loss',      # hoặc một metric khác bạn muốn theo dõi
        verbose=1,
        save_best_only=True,     # chỉ lưu khi model cải thiện
        mode='min'
    )

    # Huấn luyện mô hình
    model_unet.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=[checkpoint_callback]
    )

    # Lưu weights
    model_unet.save("attention_unet_coco.h5")
    print("Training finished. Weights saved to attention_unet_coco.h5")
