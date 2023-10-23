import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image

# Khởi tạo biến đếm số lượng mẫu khuôn mặt
count = 0

# Hàm để chụp khuôn mặt
def capture_faces():
    global count
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_id = id_entry.get()

    if not face_id:
        messagebox.showerror("Lỗi", "Vui lòng nhập ID của người dùng.")
        return

    print("\n [INFO] Đang khởi tạo việc chụp khuôn mặt. Hãy nhìn vào camera và đợi ...")

    while True:
        ret, img = cam.read()
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])
            cv2.imshow('image', img)

        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break
        elif count >= 60:
            break

    print("\n [INFO] Kết thúc việc chụp khuôn mặt và đóng camera")
    cam.release()
    cv2.destroyAllWindows()

# Hàm để nhận diện khuôn mặt
def recognize_faces():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    recognizer.read('trainer/trainer.yml')

    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)

    font = cv2.FONT_HERSHEY_SIMPLEX

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

# Tạo cửa sổ Tkinter
root = tk.Tk()
root.title("Face Capture and Recognition")

# Tạo label và entry cho ID người dùng
id_label = tk.Label(root, text="Nhập ID của người dùng:")
id_label.pack()
id_entry = tk.Entry(root)
id_entry.pack()

# Tạo nút "Chụp khuôn mặt"
capture_button = tk.Button(root, text="Chụp khuôn mặt", command=capture_faces)
capture_button.pack(pady=20)

# Tạo nút "Nhận diện khuôn mặt"
recognize_button = tk.Button(root, text="Nhận diện khuôn mặt", command=recognize_faces)
recognize_button.pack()

# Tạo nút "Thoát"
exit_button = tk.Button(root, text="Thoát", command=root.destroy)
exit_button.pack()

# Bắt đầu chạy ứng dụng Tkinter
root.mainloop()
