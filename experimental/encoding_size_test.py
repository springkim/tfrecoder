import cv2
import numpy as np
from stl10.stl10_recoder import read_all_images, read_labels

train_X = read_all_images("stl10/train_X.bin")
train_y = read_labels("stl10/train_y.bin")

print(train_X.shape)
print(train_y.shape)

for i in range(len(train_X)):
    original = cv2.imread("iu.png")
    original = cv2.resize(original, (512, 512))
    h, w, c = original.shape
    print(h * w * c * 4)

    img = cv2.cvtColor(original, cv2.COLOR_BGR2YCrCb)
    YCbCr = cv2.split(img)
    Y = YCbCr[0].astype(np.float32)
    Cb = cv2.resize(YCbCr[1], (w // 2, h // 2)).astype(np.float32)
    Cr = cv2.resize(YCbCr[2], (w // 2, h // 2)).astype(np.float32)

    Y = cv2.dct(Y)
    Cb = cv2.dct(Cb)
    Cr = cv2.dct(Cr)
    Y = Y[0:h // 4, 0:w // 4]
    Cb = Cb[0:h // 8, 0:w // 8]
    Cr = Cr[0:h // 8, 0:w // 8]
    print(len(Y.tobytes()) + len(Cb.tobytes()) + len(Cr.tobytes()))
    # img = train_X[i].astype(np.float32)
    # buf = cv2.dct(img)
    # print(len(buf))
    dY = np.zeros((h, w), np.float32)
    dY[0:h // 4, 0:w // 4] = Y
    Y = dY

    dCb = np.zeros((h // 2, w // 2), np.float32)
    dCb[0:h // 8, 0:w // 8] = Cb
    Cb = dCb

    dCr = np.zeros((h // 2, w // 2), np.float32)
    dCr[0:h // 8, 0:w // 8] = Cr
    Cr = dCr

    Y = cv2.idct(Y)
    Y[Y < 0] = 0
    Y[Y > 255] = 255
    Y = Y.astype(np.uint8)

    Cb = cv2.idct(Cb)
    Cb[Cb < 0] = 0
    Cb[Cb > 255] = 255
    Cb = Cb.astype(np.uint8)

    Cr = cv2.idct(Cr)
    Cr[Cr < 0] = 0
    Cr[Cr > 255] = 255
    Cr = Cr.astype(np.uint8)

    Cb = cv2.resize(Cb, (w, h))
    Cr = cv2.resize(Cr, (w, h))
    decoded = cv2.merge([Y, Cb, Cr])
    decoded = cv2.cvtColor(decoded, cv2.COLOR_YCrCb2BGR)
    cv2.imshow("img", cv2.hconcat([original, decoded]))
    cv2.waitKey(0)
