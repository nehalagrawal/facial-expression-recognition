from __future__ import print_function
import numpy as np
import csv
import cv2
import scipy
import scipy.misc
import imutils

to_write = list()
to_write.append(['emotion', 'pixels', 'Usage'])
data_class = 'Training'


def Flip(data):
    dataFlipped = data[..., ::-1].reshape(2304).tolist()
    return dataFlipped


def Roated15Left(data):
    num_rows, num_cols = data.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), -30, 1)
    img_rotation = cv2.warpAffine(data, rotation_matrix, (num_cols, num_rows))
    return img_rotation.reshape(2304).tolist()


def Roated15Right(data):
    num_rows, num_cols = data.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), -30, 1)
    img_rotation = cv2.warpAffine(data, rotation_matrix, (num_cols, num_rows))
    return img_rotation.reshape(2304).tolist()


def Zoomed(data):
    datazoomed = scipy.misc.imresize(data, (60, 60))
    datazoomed = datazoomed[5:53, 5:53]
    datazoomed = datazoomed.reshape(2304).tolist()
    return datazoomed


def shiftedUp20(data):
    translated = imutils.translate(data, 0, -5)
    translated2 = translated.reshape(2304).tolist()
    return translated2


def shiftedDown20(data):
    translated = imutils.translate(data, 0, 5)
    translated2 = translated.reshape(2304).tolist()
    return translated2


def shiftedLeft20(data):
    translated = imutils.translate(data, -5, 0)
    translated2 = translated.reshape(2304).tolist()
    return translated2


def shiftedRight20(data):
    translated = imutils.translate(data, 5, 0)
    translated2 = translated.reshape(2304).tolist()
    return translated2


def flatten_matrix(matrix):
    vector = matrix.flatten(1)
    vector = vector.reshape(1, len(vector))
    return vector


def ZeroCenter(data):
    data = data - np.mean(data, axis=0)
    return data


def zca_whitening(inputs):
    sigma = np.dot(inputs, inputs.T) / inputs.shape[1]  # Correlation matrix
    U, S, V = np.linalg.svd(sigma)  # Singular Value Decomposition
    epsilon = 0.1  # Whitening constant, it prevents division by zero
    ZCAMatrix = np.dot(np.dot(U, np.diag(1.0 / np.sqrt(np.diag(S) + epsilon))), U.T)  # ZCA Whitening matrix
    return np.dot(ZCAMatrix, inputs)  # Data whitening


def global_contrast_normalize(X, scale=1., subtract_mean=True, use_std=True, sqrt_bias=10, min_divisor=1e-8):
    assert X.ndim == 2, "X.ndim must be 2"
    scale = float(scale)
    assert scale >= min_divisor

    mean = X.mean(axis=1)
    if subtract_mean:
        X = X - mean[:, np.newaxis]
    else:
        X = X.copy()
    if use_std:
        ddof = 1
        if X.shape[1] == 1:
            ddof = 0
        normalizers = np.sqrt(sqrt_bias + X.var(axis=1, ddof=ddof)) / scale
    else:
        normalizers = np.sqrt(sqrt_bias + (X ** 2).sum(axis=1)) / scale
    normalizers[normalizers < min_divisor] = 1.
    X /= normalizers[:, np.newaxis]  # Does not make a copy.
    return X


def Zerocenter_ZCA_whitening_Global_Contrast_Normalize(list):
    Intonumpyarray = np.asarray(list)
    data = Intonumpyarray.reshape(48, 48)
    data2 = ZeroCenter(data)
    data3 = zca_whitening(flatten_matrix(data2)).reshape(48, 48)
    data4 = global_contrast_normalize(data3)
    data5 = np.rot90(data4, 3)
    return data5


# get the data
filname = '/content/drive/My Drive/ECS 289 Deep Learning/Dataset/all/fer2013/fer2013.csv'
label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def getData(filname):
    # images are 48x48
    # N = 35887
    Y = []
    X = []
    first = True

    with open("/content/drive/My Drive/ECS 289 Deep Learning/badtrainingdata.txt", "r") as text:
        ToBeRemovedTrainingData = []
        for line in text:
            ToBeRemovedTrainingData.append(int(line))

    number = 0

    for line in open(filname):
        number += 1
        if number not in ToBeRemovedTrainingData:
            if first:
                first = False
            else:
                row = line.split(',')
                to_write.append([row[0], row[1], data_class])

                x = row[1].split(' ')
                x = np.asarray(x, dtype=np.uint8)

                flipped = Flip(x)
                to_write.append([row[0], ' '.join(map(str, flipped)), data_class])

                rotated_left = Roated15Left(np.reshape(x, (-1, 48)))
                to_write.append([row[0], ' '.join(map(str, rotated_left)), data_class])

                rotated_right = Roated15Right(np.reshape(x, (-1, 48)))
                to_write.append([row[0], ' '.join(map(str, rotated_right)), data_class])

                zoomed = Zoomed(np.reshape(x, (-1, 48)))
                to_write.append([row[0], ' '.join(map(str, zoomed)), data_class])

#                 shifted_up = shiftedUp20(np.reshape(x, (-1, 48)))
#                 to_write.append([row[0], ' '.join(map(str, shifted_up)), data_class])

#                 shifted_down = shiftedDown20(np.reshape(x, (-1, 48)))
#                 to_write.append([row[0], ' '.join(map(str, shifted_down)), data_class])

#                 shifted_left = shiftedLeft20(np.reshape(x, (-1, 48)))
#                 to_write.append([row[0], ' '.join(map(str, shifted_left)), data_class])

#                 shifted_right = shiftedRight20(np.reshape(x, (-1, 48)))
#                 to_write.append([row[0], ' '.join(map(str, shifted_right)), data_class])

                # temp_list = list()
                # for pixel in row[1].split(' '):
                #     temp_list.append(int(pixel))
                # data = Zerocenter_ZCA_whitening_Global_Contrast_Normalize(temp_list)
                # X.append(map(int, data.reshape(2304).tolist()))
                # to_write.append([row[0], ' '.join(map(str, X[len(X) - 1])), data_class])

    with open('/content/drive/My Drive/ECS 289 Deep Learning/newfer2013.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(to_write)


getData(filname)