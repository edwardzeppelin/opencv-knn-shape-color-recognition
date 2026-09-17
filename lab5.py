import cv2
import numpy as np
import imutils

samples = np.loadtxt('generalsamples.data', np.float32)
responses = np.loadtxt('generalresponses.data', np.float32)
responses = responses.reshape((responses.size, 1))

model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)

colors = {
    "Yellow": ((20, 100, 100), (30, 255, 255)),
    "Green": ((60, 50, 50), (90, 255, 255)),
    "Red": ((0, 100, 100), (10, 255, 255)),
    "Blue": ((95, 100, 100), (120, 255, 255)),
    "Purple": ((125, 50, 50), (150, 255, 255)),
}

relatives = {
    0: ('Trapeze'),
    1: ('Pentagon'),
    2: ('Tri'),
    3: ('Rect'),
    4: ('Circle'),
    5: ('CHTO'),
    6: ('Square'),
    7: ('Lighting'),
    8: ('Rhomb')
}

image = cv2.imread('C:\Projects\Intellect\lab5\image.jpg')
im = imutils.resize(image, width=600)
im2 = im.copy()


def color_check():
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    cv2.imshow('Красная жара', hsv)

    for col in colors:
        mask = cv2.inRange(hsv, colors.get(col)[0], colors.get(col)[1])
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        contours = [i for i in contours if cv2.contourArea(i) > 50]
        find_fig(contours, mask, col)


def find_fig(contours, mask, col):
    for cnt in contours:
        [x, y, w, h] = cv2.boundingRect(cnt)
        if h > 28:
            try:
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi = mask[y:y + h, x:x + w]
                l = float(w) / h
                roismall = cv2.resize(roi, (10, 10))
                roismall = roismall.reshape((1, 100))
                roismall = np.float32(roismall)
                retval, results, neigh_resp, dists = model.findNearest(roismall, k=1)
                num = int(results.ravel()[0])
                result = relatives[num]
                text = "{} {}".format(col, result)
                cv2.putText(im2, text, (x + w // 2, y + h // 2), 0, 0.6, (0, 0, 0))
            except cv2.Error as e:
                print('Invalid')


color_check()

cv2.imshow('Result', im2)
cv2.imshow('Source', im)
cv2.waitKey(0)
cv2.destroyAllWindows()