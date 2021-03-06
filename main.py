import tensorflow as tf
import glob
from tensorflow.python.ops.confusion_matrix import confusion_matrix
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.python.client.session import Session
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv1D, Dropout, Dense, Flatten, MaxPooling1D
import os

# Load data into memory
labels = ['stationary', 'walking', 'running', 'shake']
x = []
y = []
for data_lable, label in enumerate(labels):
    print(data_lable)
    # находим все файлы в каталоге datasets с расширением csv
    filenames = glob.glob('dataset/' + label + '/*.csv')
    for filename in filenames:
        data = np.loadtxt(filename, delimiter=',')
        x.append(data)
        y.append(data_lable)

x2 = np.array(x).reshape(len(x), 1000, 3)
y2 = np.array(y)


def make_sampels_for_plot(labels, answers, n_samples=2):
    samples = []
    y3 = list(answers.copy())
    y3.append(len(labels))
    last_index = 0
    for i, label_name in enumerate(labels):
        samples.append(random.sample(range(last_index, y3.index(i + 1)), n_samples))
        last_index = y3.index(i + 1)
    return samples


def show_samples(labels, data, answers, n_samples=1):
    """
    :param labels: список заголовков
    :param data: сами данные
    :param n_samples: количество отображаемых примеров на один заголовок
    """
    n_types = len(labels)
    samples = make_sampels_for_plot(labels, answers, n_samples)
    for i, n in enumerate(samples):
        for j in range(n_samples):
            plt.subplot(n_types, n_samples, (i * n_samples) + j + 1)
            plt.plot(data[n[j]])
            plt.title(labels[i])
            plt.ylim(-4000, 4000)  # 4000 mg acc. range
    plt.tight_layout()
    plt.show()


# show_samples(labels, x2, y2, 3)


def frame(x, frame_len, hop_len):
    # Slice a 3D data array into (overlapping) frames.

    assert (x.shape == (len(x), 3))
    assert (x.shape[0] >= frame_len)
    assert (hop_len >= 1)

    n_frames = 1 + (x.shape[0] - frame_len) // hop_len
    shape = (n_frames, frame_len, x.shape[1])
    strides = ((hop_len * x.strides[0],) + x.strides)
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


x3 = []
y3 = []
for i in range(x2.shape[0]):
    # frames = frame(x2[i], 26, 26)  # no overlap
    frames = frame(x2[i], 26, 16)  # 50% overlap
    x3.append(frames)
    y3.append(np.full(frames.shape[0], y[i]))

x3 = np.concatenate(x3)
y3 = np.concatenate(y3)
# print("x3.shape", x3.shape)
# print("y3.shape", y3.shape)

x4 = x3 / 4000

x_train, x_test, y_train, y_test = train_test_split(x4, y3, test_size=0.25)
print(np.shape(x_test))
print("Trainning samples:", x_train.shape)
print("Testing samples:", x_test.shape)

model = Sequential()
model.add(Conv1D(filters=8, kernel_size=3, activation='relu', input_shape=(26, 3)))
model.add(MaxPooling1D(pool_size=4))
model.add(Conv1D(filters=4, kernel_size=3, activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(4, activation='softmax'))
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(26, 3)),
#   tf.keras.layers.Conv1D(filters=8, kernel_size=3, activation='relu'),
#   tf.keras.layers.Dropout(0.5),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(64, activation='relu'),
#   tf.keras.layers.Dense(3, activation='softmax')
# ])
model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print("Train start")

# model.fit(x_train, y_train, epochs=20, verbose=0)
# test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
#
# print("Test loss:", test_loss)
# print("Test acc:", test_acc)
pwd = os.getcwd()
path = pwd + '/h5'
print(path)
if not os.path.exists(path):
    os.mkdir(path)
os.chdir(path)
for a in range(10):
    model.fit(x_train, y_train, epochs=20, verbose=0)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)

    print("Test loss:", test_loss)
    print("Test acc:", test_acc)

    # Save the model into an HDF5 file ‘model.h5’
    model.save('model_' + str(round(test_acc, 5)) + '.h5')
    model_json = model.to_json()
    json_file = open("model.json", "w")
    # Записываем архитектуру сети в файл
    json_file.write(model_json)
    json_file.close()

os.chdir(pwd)

y_pred = model.predict_classes(x_test)
confusion_matrix = confusion_matrix(y_test, y_pred)
sess = Session()

plt.figure()
sns.heatmap(sess.run(confusion_matrix),
            annot=True,
            xticklabels=labels,
            yticklabels=labels,
            cmap=plt.cm.Blues,
            fmt='d', cbar=False)
# plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
