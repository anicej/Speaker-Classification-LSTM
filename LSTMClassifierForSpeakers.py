import librosa
import json
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Conv2D, Dropout
import tensorflow as tf
import matplotlib.pyplot as plt

trainF = open('E:\projects\\amerandishan\data\\train_data.json', )
trainData = json.load(trainF)

testF = open('E:\projects\\amerandishan\data\\test_data.json', )
basePath="E:\projects\\amerandishan\data\\"
audioPath=basePath+ "wavs\\"
testData = json.load(testF)

trainF = open(basePath+'train_data.json', )
trainData = json.load(trainF)
testF = open(basePath+'test_data.json', )
testData = json.load(testF)

trainX = np.zeros((128, 0))
trainy = np.zeros((0, 38))
temp = np.zeros((38))

testX = np.zeros((128, 0))
testy = np.zeros((0, 38))
a = np.array([1], dtype="S8")
b = np.array([38], dtype="S8")
lookupTable = ["" for x in range(38)]

for key in testData:
    a[0] = key
    b = np.append(b, a, axis=0)
lookupTable, indexedTrainy = np.unique(b, return_inverse=True)
for key in testData:
    print(key)
    for i in testData[key]:
        y, sr = librosa.load(basePath+'wavs\\' + i, sr=16000)
        x = librosa.feature.melspectrogram(y=y, sr=sr)
        testX = np.append(testX, x, axis=1)
        for i in range(x.shape[1]):
            for x in range(len(lookupTable)):
                if lookupTable[x].decode('UTF-8') == key:
                    temp[indexedTrainy[x]] = 1
                    break

            testy = np.append(testy, temp)
            temp = np.zeros((38))

print(testy.shape)
testF.close()

np.savetxt('testX.csv', testX, delimiter=',')
np.savetxt('testy.csv', testy, delimiter=',')
#data
for key in trainData:
    print(key)
    for i in trainData[key]:
        y, sr = librosa.load(audioPath + i, sr=16000)
        x = librosa.feature.melspectrogram(y=y, sr=sr)
        trainX = np.append(trainX, x, axis=1)
        for i in range(x.shape[1]):

            for x in range(len(lookupTable)):
                if lookupTable[x].decode('UTF-8') == key:
                    temp[indexedTrainy[x]] = 1
                    break

            trainy = np.append(trainy, temp)
            temp = np.zeros((38))
trainF.close()
np.savetxt('trainX.csv', trainX, delimiter=',')
np.savetxt('trainy.csv', trainy, delimiter=',')
# trainX = np.loadtxt('trainX.csv', delimiter=',')
# trainy = np.loadtxt('trainy.csv', delimiter=',')
# testX = np.loadtxt('testX.csv', delimiter=',')
# testy = np.loadtxt('testy.csv', delimiter=',')

optimizer = tf.keras.optimizers.SGD(lr=1e-12, momentum=0.1)

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(None, 128)))
model.add(LSTM(30, input_shape=(128, 1)))
model.add(Dense(10))
model.add(Dense(38, activation='softmax'))
model.compile(
              optimizer='adam',
              loss='mean_squared_error',
              metrics=[
                  tf.metrics.MeanSquaredError(name='my_mse'),
                  tf.metrics.AUC(name='my_auc'),
              ])
trainX = trainX.reshape(trainX.shape[0], 1, trainX.shape[1])
testX  = testX.reshape(testX.shape[0], 1,testX.shape[1])
trainX = np.array(trainX).transpose()
testX = np.array(testX).transpose()

trainy=trainy.reshape(148572,38)
testy=np.array(testy.reshape(29329,38))
hist = model.fit(trainX,
                 trainy,
                 batch_size=24,
                 epochs=3,
                 validation_data=(testX, testy),
                 )

plt.plot(hist.history['my_auc'])
plt.plot(hist.history['val_my_auc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()