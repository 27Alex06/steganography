import cv2
import numpy as np
import math

secret = []

def frombits(bits):
    chars = []
    for b in range(int(len(bits) / 8)):
        byte = bits[b * 8:(b + 1) * 8]
        chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
    return ''.join(chars)

def inttobitarray(_int):
    _bit = [0, 0, 0, 0, 0, 0, 0, 0]
    _conv = bin(_int)[2:]
    _convL = len(_conv)
    for i in range(_convL):
        _bit[i] = int(_conv[_convL - i - 1])
    return _bit


def image1converter(_bites):
    value = 0
    for i in range(2, 8):
        value += _bites[i] * 2 ** (i - 2)
    return value


def image2converter(_bites):
    value = 0
    for i in range(0, 2):
        value += _bites[i] * 2 ** i
    return value

def get_secret(_int, length):
    _bit = []
    for i in range(length):
        _bit.append(0)
    _conv = bin(_int)[2:]
    _convL = len(_conv)
    for i in range(_convL):
        _bit[i] = int(_conv[_convL - i - 1])
    return _bit

def append_secret(_bites):
    for i in range(len(_bites)):
        secret.append(_bites[i])

img = cv2.imread('output.png', cv2.IMREAD_GRAYSCALE)

height = img.shape[0]
width = img.shape[1]

image1 = np.ndarray(shape=(height, width), dtype=int)
image2 = np.ndarray(shape=(height, width), dtype=int)

heightT = int(height / 3)
widthT = int(width / 2)

for row in range(width):
    for col in range(height):
        _byte = int(img[col, row])
        _bites = inttobitarray(_byte)
        image1[col, row] = image1converter(_bites)
        image2[col, row] = image2converter(_bites)

        # DEBUG
        # print(_byte, _bites, image1[col, row], image2[col, row])


secret = []

for ht in range(heightT):
    for wt in range(widthT):
        _matrix = [
            [ht * 3, wt * 2],
            [ht * 3 + 2, wt * 2],
            [ht * 3, wt * 2 + 1],
            [ht * 3 + 2, wt * 2 + 1],
        ]
        pu = image1[ht * 3 + 1, wt * 2]
        pb = image1[ht * 3 + 1, wt * 2 + 1]

        #print(ht, wt, [pu, pb])

        l = 0
        u = 0

        for i in range(4):
            pi = image1[_matrix[i][0], _matrix[i][1]]
            du = (pi - pu)
            db = (pi - pb)

            if du > 0 and db > 0:
                l = max(pu + 1, pb + 1)
                u = 63

            if du <= 0 and db <= 0:
                l = 0
                u = min(pu, pb)

            if du > 0 and db <= 0:
                l = pu + 1
                u = pb

            if du <= 0 and db > 0:
                l = pb + 1
                u = pu

            n = math.floor(min(math.log2(abs(u-l+1)), 3))

            if n > 0:
                _mod = 2 ** n
                #if abs(pi - pu) % _mod != 0:
                _bit = get_secret(abs(pi - pu) % _mod, n)
                append_secret(_bit)

        _matrix2 = [
            [ht * 3, wt * 2],
            [ht * 3, wt * 2 + 1],
            [ht * 3 + 1, wt * 2],
            [ht * 3 + 1, wt * 2 + 1],
            [ht * 3 + 2, wt * 2],
            [ht * 3 + 2, wt * 2 + 1],
        ]
        for i in range(6):
            append_secret(get_secret(image2[_matrix2[i][0], _matrix2[i][1]], 2))




print(frombits(secret))
print("Done!")