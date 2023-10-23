import cv2
import numpy as np
import os 

# Tạo một đối tượng nhận diện khuôn mặt
recognizer = cv2.face_LBPHFaceRecognizer.create()
recognizer.read('trainer/trainer.yml')

# Sử dụng bộ phát hiện khuôn mặt
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

# Khởi tạo ID ban đầu
id = 0

# Danh sách tên tương ứng với ID: Ví dụ Marcelo: id=1, v.v.
names = ['None', 'dat', 'hieu']  

# Khởi tạo và bắt đầu quá trình quay video trực tiếp
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Thiết lập chiều rộng video
cam.set(4, 480)  # Thiết lập chiều cao video

# Xác định kích thước tối thiểu để nhận diện khuôn mặt
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:

    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        if confidence < 100:
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)

    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xff  # Ấn 'ESC' để thoát video
    if k == 27:
        break

# Dọn dẹp tài nguyên
print("\n [INFO] Kết thúc chương trình và dọn dẹp tài nguyên")
cam.release()
cv2.destroyAllWindows()
