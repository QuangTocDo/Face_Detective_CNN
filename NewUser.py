import cv2
import sqlite3
cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Hàm cập nhật tên và ID vào CSDL
def insertOrUpdate(name):
    conn = sqlite3.connect('FaceBaseNew.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS People
                (ID INTEGER PRIMARY KEY AUTOINCREMENT,
                Name TEXT NOT NULL);''')
    conn.commit()

    cmd = "INSERT INTO People (Name) VALUES (?)"
    conn.execute(cmd, (name,))

    conn.commit()

    # Lấy ID của người dùng vừa được thêm vào
    cursor = conn.execute('SELECT last_insert_rowid()')
    id = cursor.fetchone()[0]

    conn.close()

    return id


# id=input('Nhập mã nhân viên:')
name=input('Nhập tên:')
print("Bắt đầu chụp ảnh nhân viên, nhấn q để thoát!")

id = insertOrUpdate(name)

sampleNum=0
sizeboxW = 300
sizeboxH = 400

while(True):

    ret, img = cam.read()

    # Lật ảnh cho đỡ bị ngược
    img = cv2.flip(img,1)

    # Kẻ khung giữa màn hình để người dùng đưa mặt vào khu vực này
    centerH = img.shape[0] // 2
    centerW = img.shape[1] // 2
    cv2.rectangle(img, (centerW - sizeboxW // 2, centerH - sizeboxH // 2),
                  (centerW + sizeboxW // 2, centerH + sizeboxH // 2), (255, 255, 255), 5)

    # Đưa ảnh về ảnh xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Nhận diện khuôn mặt
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # Vẽ hình chữ nhật quanh mặt nhận được
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        sampleNum = sampleNum + 1
        import os

        # Tạo đường dẫn cho thư mục dataSet/{name}
        path = f"dataSet/{name}"

        # Tạo thư mục nếu nó chưa tồn tại
        if not os.path.exists(path):
            os.makedirs(path)

        # Lưu hình ảnh vào thư mục dataSet/{name}
        cv2.imwrite(f"{path}/User.{id}.{sampleNum}.jpg", gray[y:y + h, x:x + w])

    cv2.imshow('frame', img)
    # Check xem có bấm q hoặc trên 100 ảnh sample thì thoát
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    elif sampleNum>=2000:
        break

cam.release()
cv2.destroyAllWindows()
