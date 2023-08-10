import cv2
import numpy as np
import sqlite3
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf

# model = tf.keras.models.load_model('./model/CNN')
model = tf.keras.models.load_model('./model/CNN')
# Check its architecture
model.summary()

# Khởi tạo bộ phát hiện khuôn mặt
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

id = 0

# Kết nối tới cơ sở dữ liệu
conn = sqlite3.connect('FaceBaseNew.db')
cur = conn.cursor()

# Khởi tạo camera
cam = cv2.VideoCapture(0)
while (True):
    # Đọc ảnh từ camera
    ret, img = cam.read()

    # Lật ảnh cho đỡ bị ngược
    img = cv2.flip(img, 1)

    # Chuyển ảnh về xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Phát hiện các khuôn mặt trong ảnh camera
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)

    id_to_name = {}
    cur.execute("SELECT * FROM People")
    rows = cur.fetchall()
    for row in rows:
        person_id = row[0]
        person_name = row[1]
        id_to_name[person_id] = person_name

    # Lặp qua các khuôn mặt nhận được để hiện thông tin
    for (x, y, w, h) in faces:
        # Vẽ hình chữ nhật quanh mặt
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Chuyển đổi ảnh về định dạng phù hợp và thực hiện dự đoán

        roi_color = img[y:y+h, x:x+w]
        roi_color = cv2.resize(roi_color, (150, 150))
        roi_color = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)  # Chuyển đổi sang định dạng RGB
        preds = model.predict(np.array([roi_color]))
        person_id = np.argmax(preds)
        # person_name = id_to_name[person_id]
        print(preds)
        print(person_id)
        print("Đây là: {}".format(id_to_name[person_id+1]))
        cv2.putText(img, str(person_id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow('Face', img)
    # Nếu nhấn q thì thoát
    if cv2.waitKey(1) == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
