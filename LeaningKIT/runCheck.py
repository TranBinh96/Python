import cv2
import numpy as np
from ultralytics import YOLO

# Hàm giảm nhiễu ảnh (Denoising) với Bộ lọc Bilateral
def bilateral_denoising(image):
    denoised = cv2.bilateralFilter(image, d=5, sigmaColor=40, sigmaSpace=40)  # Giảm thông số lọc
    return denoised

# Hàm làm sắc nét ảnh (Sharpening)
def sharpen_image(image):
    kernel = np.array([[0, -0.3, 0], [-0.3, 2, -0.3], [0, -0.3, 0]])  # Làm sắc nét rất nhẹ
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

# Hàm điều chỉnh độ sáng và độ tương phản
def adjust_brightness_contrast(image, alpha=1.1, beta=10):  # Giảm độ tương phản và sáng
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

# Hàm tiền xử lý ảnh (Preprocessing)
def preprocess_image(image):
    # Giảm nhiễu bằng Bilateral Filter
    image = bilateral_denoising(image)

    # Điều chỉnh độ sáng và độ tương phản
    image = adjust_brightness_contrast(image)

    # Làm sắc nét ảnh
    image = sharpen_image(image)

    return image



# Tải mô hình YOLO đã huấn luyện
model = YOLO('./databaseKIT/best.pt')  # Đảm bảo đường dẫn mô hình chính xác

# Đường dẫn tới bức ảnh bạn muốn kiểm tra
image_path = './dataset/images/train/53_142913.jpg'

# Đọc ảnh
image = cv2.imread(image_path)

# Tiền xử lý ảnh
processed_image = preprocess_image(image)

# Dự đoán trên ảnh đã xử lý
results = model.predict(source=image, conf=0.5)

# Truy cập kết quả dự đoán
result = results[0]  # Lấy đối tượng đầu tiên trong danh sách kết quả

# Kiểm tra xem có đối tượng nào được phát hiện không
if len(result.boxes) == 0:
    print("Không phát hiện đối tượng nào.")
else:
    # Hiển thị ảnh với bounding boxes và nhãn của đối tượng
    result.show()

    # Lưu ảnh kết quả với bounding boxes
    result.save()  # Lưu kết quả vào thư mục 'runs/detect/exp'
