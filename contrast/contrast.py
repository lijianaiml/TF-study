import numpy as np
from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
# from keras.optimizers import SGD
from keras.optimizers import Adam
import cv2


img2 = cv2.imread('./data/2.BMP')
img2 = cv2.resize(img2, (2592, 1728))
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
x_train = np.expand_dims(img2, axis=0)
x_train = np.expand_dims(x_train, axis=3)
x_train = x_train / 255

img1 = cv2.imread('./data/1.BMP')
img1 = cv2.resize(img1, (2592, 1728))
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
y_train = np.expand_dims(img1, axis=0)
y_train = np.expand_dims(y_train, axis=3)
y_train = y_train / 255

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu',
                 input_shape=(1728, 2592, 1), padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

model.add(Conv2D(1, (1, 1), activation='relu'))
# model.add(Flatten())
model.summary()
# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
Adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
            epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss='mean_squared_error', optimizer=Adam)

model.fit(x_train, y_train, batch_size=1, epochs=1000)
model.save('model1.h5')
