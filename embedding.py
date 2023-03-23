import cv2
import numpy as np
import math
import random


img = cv2.imread('picture.png', cv2.IMREAD_GRAYSCALE)
secret = "Pixel value differencing steganography (PVDS) technique efficiently identifies the edge and smooth regions from an image, therefore, the PVDS technique is more suitable for concealing the secret information in an image. Notwithstanding the advantages, the PVDS techniques give rise to some major issues. Most of the PVDS techniques suffer from boundary issue (BI). Further, the majority of the PVDS techniques are exposed to pixel difference histogram (PDH) analysis. In this paper, two improved PVDS based techniques, such as (1) indicator-based PVDS (IPVDS) and (2) multi-directional overlapped PVDS (MDOPVDS) has been proposed to address the BI as well as PDH issue. The IPVDS technique utilizes the concept of indicator pixel in a 2 × 3 pixel block. On the other hand, the MDOPVDS technique utilizes the pixel overlap strategy in a 3 × 3 pixel block using all three directions, such as (1) horizontal, (2) vertical, and (3) diagonal. Results of the proposed technique are evaluated in terms of the conflicting metrics like peak signal-to-noise ratio (PSNR), capacity, and security. Further, it is observed that the proposed technique maintains a good symmetry between the aforementioned conflicting metrics. Additionally, both the proposed technique successfully avoids the BI."
secret_index = 0
secret_bites_capability = 0

def tobits(s):
    result = []
    for c in s:
        bits = bin(ord(c))[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result


def frombits(bits):
    chars = []
    for b in range(int(len(bits) / 8)):
        byte = bits[b * 8:(b + 1) * 8]
        chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
    return ''.join(chars)


secret_bits = tobits(secret)
secret_lenght = len(secret_bits)


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


def image12cmp(_bites1, _bites2):
    _value = 0
    for i in range(0, 2):
        _value += _bites2[i] * 2 ** i
    for i in range(2, 8):
        _value += _bites1[i - 2] * 2 ** i
    return _value


def get_bits_from_secret(count):
    global secret_index
    res = []
    if secret_index>secret_lenght:
        return [0]
    if secret_index+count>secret_lenght:
        count = secret_lenght-secret_index
    for i in range(count):
        res.append(secret_bits[secret_index])
        secret_index+=1
    return res


def bits_to_int(bits):
    res = 0
    for i in range(len(bits)):
        res+=bits[i]*2**i
    return res



height = img.shape[0]
width = img.shape[1]

image1 = np.ndarray(shape=(height, width), dtype=int)
image2 = np.ndarray(shape=(height, width), dtype=int)

for row in range(width):
    for col in range(height):
        _byte = int(img[col, row])
        _bites = inttobitarray(_byte)
        image1[col, row] = image1converter(_bites)
        image2[col, row] = image2converter(_bites)

        # DEBUG
        # print(_byte, _bites, image1[col, row], image2[col, row])

heightT = int(height / 3)
widthT = int(width / 2)

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

            n = math.floor(min(math.log2(abs(u-l+1)), 4))

            #print(ht, wt, i, n, pi, [pu, pb], [u , l], n)
            # find e
            if n > 0:
                _min = 1000
                _bit = get_bits_from_secret(n)
                data = bits_to_int(_bit)
                # for i in range(n):
                #     if(secret_index < secret_max):
                #         data += secret_bits[secret_index]*2**i
                #         secret_index += 1

                _mod = 2 ** n
                value = pi
                for e in range(l, u+1):
                    if (abs(e - pu) - data) % _mod == 0:
                        #print((abs(e - pu) - data), _mod)
                        if _min > abs(e - pi):
                            _min = abs(e - pi)
                            value = e
                #print(pi, pu, _mod, value, data)
                secret_bites_capability+=n
                image1[_matrix[i][0], _matrix[i][1]] = value
                #print(l, value, u)
                #print(ht, wt, i, pi, value, [pu, pb], [u, l], n)

        _matrix2 = [
            [ht * 3, wt * 2],
            [ht * 3, wt * 2 + 1],
            [ht * 3 + 1, wt * 2],
            [ht * 3 + 1, wt * 2 + 1],
            [ht * 3 + 2, wt * 2],
            [ht * 3 + 2, wt * 2 + 1],
        ]
        for i in range(6):
            secret_bites_capability += 2
            image2[_matrix2[i][0], _matrix2[i][1]] = bits_to_int(get_bits_from_secret(2))

output = np.ndarray(shape=(height, width), dtype=int)

for row in range(width):
    for col in range(height):
        _bites1 = inttobitarray(int(image1[col, row]))
        _bites2 = inttobitarray(int(image2[col, row]))
        output[col, row] = image12cmp(_bites1, _bites2)
        #if row % 3 == 1:
            #print(row, col, output[col, row], img[col, row])
        #print(img[col, row], _bites2, output[col, row])

#cv2.imwrite("image1.png", image1)
#cv2.imwrite("image2.png", image2)
cv2.imwrite("gray.png", img)

sum = 0
for row in range(width):
    for col in range(height):
        sum += (output[col, row]-img[col, row])**2
MSE = sum/(width*height)
PNSR=10*math.log10(255*255/MSE)

cv2.imwrite("output.png", output)
print("PNSR:", PNSR)
print(secret_bites_capability)
print("capability:", secret_bites_capability/(width*height))


print("Done!")
