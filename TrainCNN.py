import cv2
import numpy as np
import os
# import sqlite3
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Đọc dữ liệu từ cơ sở dữ liệu và lưu vào mảng và nhãn tương ứng
#     labels = []
#     cur.execute("SELECT * FROM People")
#     rows = cur.fetchall()
#     data = []
#     resData=[]
#     for row in rows:
#         person_id = row[0]
#         person_name = row[1].strip()
#         person_path = os.path.join(path, str(person_name).strip())
#         print(row)
#         print(person_name)
#         for file in os.listdir(person_path):
#             img = cv2.imread(os.path.join(person_path, file))
#             img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             img_gray_resized = cv2.resize(img_gray, (150, 150))
#             data.append(img_gray_resized.reshape(150, 150, 1))
#             labels.append(person_id)
#         resData.append(data)
#         data=[]
#     return data, labels
def get_data(data_dir,target_size=(150, 150)):
    clss = os.listdir(data_dir)
    imgs = []
    img_names = []
    labels = []
    error_img = 0
    error_img_name = []
    for person_name in tqdm(clss):
        person_path = os.path.join(data_dir, person_name)
        for img_name in os.listdir(person_path):
            try:
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, target_size)
                imgs.append(img.astype(float))
                img_names.append(img_name)
                labels.append(person_name)
            except:
                error_img += 1
                error_img_name.append(img_name)
                pass
    imgs = np.asarray(imgs, dtype=float)
    print("\n[INFO] Danh sách nhân viên: {}".format(clss))
    print('[INFO] The number of images = {}'.format(imgs.shape))
    print('[INFO]Error img count = {}, error img namems = {}\n'.format(error_img, error_img_name))
    return imgs, img_names, labels, clss

# lấy dữ liệu
imgs, img_names, labels, clss = get_data('dataset')

# convert label to one-hot-coding
dict_labels = {}
for i, cls in enumerate(clss):
    dict_labels[cls] = i
print(dict_labels)

ys = [dict_labels[label] for label in labels]
ys = np.asarray(ys)
ys = tf.keras.utils.to_categorical(ys-1, num_classes=len(clss))
print(ys)
print('[INFO] Y shape = {}\n'.format(ys.shape))

# Phân chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(imgs, ys, test_size=0.2, random_state=42)
print('[INFO] Train set = {}, {}'.format(X_train.shape, y_train.shape))
print('[INFO] Test set = {}, {}\n'.format(X_test.shape, y_test.shape))


# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# mô hình với siêu tham số
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(clss), activation='softmax'))

# Compile và huấn luyện mô hình với siêu tham số tốt hơn
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),epochs=50, validation_data=(X_test, y_test),callbacks=[EarlyStopping(monitor='val_loss', patience=1)])

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", loss)
print("Test accuracy:", acc)



# Lưu mô hình
model.save('./model/CNN')



print("Trained!")