import numpy as np
from tensorflow.python.keras.models import model_from_json

# Загружаем из файла обученную сверточную нейронную сеть:
json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

# Компилируем модель перед использованием
loaded_model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

data = np.array([[[1605, 1276, 7949], [1580, 1266, 7990], [1595, 1286, 7923], [1604, 1292, 7966], [1735, 1433, 7631],
                  [1560, 1282, 7859], [1606, 1263, 7968], [1607, 1287, 7943], [1601, 1260, 8002], [1613, 1292, 7983],
                  [1598, 1276, 7972], [1610, 1283, 7979], [1594, 1267, 7963], [1597, 1287, 7964], [1593, 1280, 8032],
                  [1600, 1291, 7991], [1603, 1278, 7966], [1610, 1262, 7962], [1601, 1281, 7960], [1592, 1278, 7989],
                  [1600, 1291, 7987], [1619, 1284, 7992], [1598, 1272, 7982], [1619, 1278, 7991], [1592, 1288, 7991],
                  [1600, 1277, 7990]]])
data2 = data / 32000

d = np.loadtxt("C:/Users/u/PycharmProjects/PyUart/np_test.txt")
d = d / 16000
print(np.shape(d))
a = np.expand_dims(d, axis=0)
print(np.shape(a))

print(loaded_model.predict(a))
