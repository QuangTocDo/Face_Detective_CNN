from flask import Flask, render_template, request,redirect, url_for, flash, get_flashed_messages, session
import cv2
import numpy as np
import sqlite3
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import time

app = Flask(__name__)
app.secret_key = 'ronansibappe'

# model = tf.keras.models.load_model('./model/CNN')
model = tf.keras.models.load_model('./model/CNN')

# kiến trúc của mô hình
# model.summary()

# Khởi tạo bộ phát hiện khuôn mặt
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Tạo thông tin người dùng
users = [
    {
        "id": 0,
        "name": "khoa",
        "password": "12345"
    },
    {
        "id": 1,
        "name": "Quang",
        "password": "12345"
    },
]
def get_user(username, password):
    for user in users:
        if user['name'] == username and user['password'] == password:
            return user
    return None

# // GET home
@app.route('/')
def index():
    return render_template('index.html')

# //GET login
@app.route('/login', methods=['GET'])
def login():
    messages = get_flashed_messages()
    return render_template('login.html', messages=messages)

# //POST login
@app.route('/login', methods=['POST'])
def do_login():
    username = request.form['username']
    password = request.form['password']
    user = get_user(username,password)
    if user is not None:
        return redirect(url_for('face_detection', user_id=user['id'], username=user['name']))
    else:
        flash('Đăng nhập thất bại!')
        return redirect(url_for('login'))

@app.route('/register')
def register():
    return render_template('registerForm.html')

# Route để nhận diện khuôn
@app.route('/face_detection')
def face_detection():
    user_id = request.args.get('user_id')
    username = request.args.get('username')
    if user_id is None or username is None:
        return redirect(url_for('login'))
    else:
        # Khởi tạo bộ phát hiện khuôn mặt
        faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        # Khởi tạo camera
        cam = cv2.VideoCapture(0)

        while (True):
            # khởi tạo bộ đếm số lần giống với dự đoán
            count = 0

            # Đọc ảnh từ camera
            ret, img = cam.read()

            # Lật ảnh cho đỡ bị ngược
            img = cv2.flip(img, 1)

            # Chuyển ảnh về xám
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Phát hiện các khuôn mặt trong ảnh camera
            faces = faceDetect.detectMultiScale(gray, 1.3, 5)

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
                print(preds)
                print(person_id)
                print(user_id)
                if (str(user_id) == str(person_id)):
                    print("Đây là: {}".format(username))
                cv2.putText(img, str(person_id)+"_Press_q_to_break", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                # xử lí xác thực hay không:
                if (str(user_id) == str(person_id)):
                    if preds[0][person_id] == 1:
                        cam.release()
                        cv2.destroyAllWindows()
                        return 'Đăng nhập thành công!'

            cv2.imshow('Face', img)
            # Nếu nhấn q thì thoát
            if cv2.waitKey(1) == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()
        time.sleep(2000)
        return 'không thành công!'


if __name__ == '__main__':
    app.run(debug=True)
