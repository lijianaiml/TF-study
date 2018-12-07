from keras.models import *
import cv2
import numpy as np

model = load_model('./model1.h5')


img2 = cv2.imread('./data/3.BMP')
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img2_resize = cv2.resize(img2_gray, (2592, 1728))
x_train = np.expand_dims(img2_resize, axis=0)
x_train = np.expand_dims(x_train, axis=3)
x_train = x_train / 255

img1 = cv2.imread('./data/1.BMP')
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img1_resize = cv2.resize(img1_gray, (2592, 1728))

output = model.predict(x_train)
output = output * 255
output = output[0, :, :, 0]


cv2.imwrite('./data/output.jpg', output)
cv2.imwrite('./data/truth.jpg', img1_resize)
cv2.imwrite('./data/input.jpg', img2_resize)
