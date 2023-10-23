import cv2
import numpy as np
from PIL import Image
import os

# Đường dẫn đến thư mục chứa ảnh khuôn mặt
path = 'dataset'

# Tạo một đối tượng nhận diện khuôn mặt
recognizer = cv2.face_LBPHFaceRecognizer.create()

# Sử dụng bộ phát hiện khuôn mặt
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Hàm để lấy ảnh và dữ liệu nhãn
def getImagesAndLabels(path):

    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L')  # Chuyển ảnh sang định dạng grayscale
        img_numpy = np.array(PIL_img, 'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)

    return faceSamples, ids

print("\n [INFO] Đang huấn luyện mô hình nhận diện khuôn mặt. Vui lòng đợi ...")
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Lưu mô hình vào thư mục "trainer/trainer.yml"
recognizer.save('trainer/trainer.yml')

# In ra số lượng khuôn mặt đã được huấn luyện và kết thúc chương trình
print("\n [INFO] {0} khuôn mặt đã được huấn luyện. Kết thúc chương trình".format(len(np.unique(ids))))
