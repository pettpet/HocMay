import cv2
import os

# Khởi tạo camera
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Đặt chiều rộng video
cam.set(4, 480)  # Đặt chiều cao video

# Sử dụng bộ phát hiện khuôn mặt
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Nhập ID cho người dùng
face_id = input('\n Nhập ID của người dùng và nhấn <Enter> ==>  ')

print("\n [INFO] Đang khởi tạo việc chụp khuôn mặt. Hãy nhìn vào camera và đợi ...")
# Khởi tạo biến đếm số lượng mẫu khuôn mặt
count = 0

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)  # Lật ảnh video theo chiều dọc
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        # Lưu ảnh đã chụp vào thư mục "datasets"
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff  # Nhấn 'ESC' để thoát khỏi video
    if k == 27:
        break
    elif count >= 60:  # Lấy 60 mẫu khuôn mặt và dừng video
        break

# Dọn dẹp tài nguyên
print("\n [INFO] Kết thúc chương trình và dọn dẹp tài nguyên")
cam.release()
cv2.destroyAllWindows()
